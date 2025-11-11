# src/talaria/hermes_service/services/utterance_service.py
import json
import logging
from typing import List, Dict, Any
from ..schemas.utterance import UtteranceProcessIn, UtteranceProcessOut
from ...core.database import get_db_cursor
from ...hermes_service.graph import app as langgraph_app # LangGraph 앱 import
from ...hermes_service.state import TranslateState # Graph State import
import time

log = logging.getLogger(__name__)

def _get_chat_history(meeting_id: int, limit: int = 10) -> List[str]:
    """DB에서 최근 대화 기록을 가져옵니다."""
    history = []
    try:
        with get_db_cursor(dictionary=True) as cursor:
            # (참고) text 컬럼에 보정된 텍스트가 있다면 그것을 가져오는 것이 더 좋습니다.
            # 지금은 원본 STT 텍스트를 가져옵니다.
            sql = """
                SELECT text 
                FROM tl_utterances 
                WHERE meeting_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """
            cursor.execute(sql, (meeting_id, limit))
            results = cursor.fetchall()
            # 최신순(DESC)으로 가져왔으니, 시간순(ASC)으로 뒤집어 줍니다.
            history = [row['text'] for row in reversed(results)]
    except Exception as e:
        log.error(f"❌ Chat History 조회 실패: {e}")
    return history

def _run_contextual_translation(state_input: TranslateState) -> TranslateState:
    """LangGraph를 실행하여 문맥 보정 번역을 수행합니다."""
    try:
        # hermes_service/graph.py의 'app'을 실행합니다.
        final_state = langgraph_app.invoke(state_input)
        return final_state
    except Exception as e:
        log.error(f"❌ LangGraph 실행 실패: {e}")
        # 실패 시, 입력 state를 그대로 반환 (answer가 비어있음)
        return state_input

def _save_utterance(payload: UtteranceProcessIn, final_answer: str, ai_tags: Dict[str, Any]):
    """발화 내용을 tl_utterances 테이블에 저장합니다."""
    try:
        with get_db_cursor(commit=True) as cursor:
            sql = """
                INSERT INTO tl_utterances (
                    meeting_id, speaker_id, track_sid, seq, 
                    started_ms, ended_ms, lang, text, 
                    confidence, ai_tags, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, NOW()
                )
            """

            seq_val = payload.ended_ms or int(time.time() * 1000)

            # (참고) seq는 DDL에 AUTO_INCREMENT가 없으므로 임시로 0 처리
            params = (
                payload.meeting_id,
                payload.speaker_id,
                payload.track_sid,
                seq_val,
                payload.started_ms,
                payload.ended_ms,
                payload.lang,
                payload.stt_text,
                payload.confidence,
                json.dumps(ai_tags) # ai_tags를 JSON 문자열로 변환
            )
            cursor.execute(sql, params)
    except Exception as e:
        log.error(f"❌ Utterance DB 저장 실패: {e}")


def process_and_save_utterance(payload: UtteranceProcessIn) -> UtteranceProcessOut:
    """
    (Service) 발화 처리의 메인 로직
    1. History 조회
    2. Graph 실행 (보정)
    3. DB 저장
    4. 결과 반환
    """
    log.info(f"발화 처리 시작 (Meeting: {payload.meeting_id}, Speaker: {payload.speaker_id}): {payload.stt_text}")

    # --- 1. DB에서 Chat History 조회 ---
    chat_history = _get_chat_history(payload.meeting_id)

    # --- 2. Graph 실행 (보정) ---
    # TODO: lang(ko-KR)을 target(en-US)으로 변환하는 로직 필요
    # 우선 simple_translations에서 첫 번째 값(영어)을 사용
    target_lang = "en-US" # (임시)
    simple_translation = payload.simple_translations.get(target_lang, "")

    # LangGraph에 넣을 State 객체 준비
    graph_input_state = TranslateState(
        stt_text=payload.stt_text,
        translated_text=simple_translation, # 1차 번역본
        context=[], # RAG용 context (일단 비워둠)
        meeting_id=payload.meeting_id,
        docs_id=None, # (중요) RAG를 위해 이 값도 채워야 함 (node_check_conditions.py 로직 참고)
        chat_history=chat_history,
        found_terms=[], # Graph가 채워줄 것
        should_context_translate=False, # Graph가 채워줄 것
        answer="" # Graph가 채워줄 것
    )

    # 그래프 실행
    final_state = _run_contextual_translation(graph_input_state)
    
    # 그래프가 보정한 최종 번역 (보정에 실패하면 'answer'가 비어있음)
    corrected_answer = final_state.get("answer") or simple_translation

    # --- 3. DB 저장 ---
    # ai_tags에 모든 데이터를 저장
    ai_tags = {
        "simple_translations": payload.simple_translations,
        "contextual_translation": {
            target_lang: corrected_answer
        },
        # (참고) manner_checker는 현재 agent.py에 있습니다. 
        # 이 로직을 hermes_service로 옮기면, 여기서 manner_alerts도 추가 가능
        "manner_alerts": [] 
    }
    
    _save_utterance(payload, corrected_answer, ai_tags)

    # --- 4. livekit_agent로 결과 반환 ---
    # (중요) agent가 UI로 보낼 수 있도록 보정된 최종 번역본을 반환
    final_translations = payload.simple_translations.copy()
    final_translations[target_lang] = corrected_answer # 영어 번역본을 덮어쓰기

    return UtteranceProcessOut(
        stt_text=payload.stt_text,
        lang=payload.lang,
        contextual_translations=final_translations,
        ai_tags=ai_tags
    )
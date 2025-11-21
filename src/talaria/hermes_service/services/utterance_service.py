# src/talaria/hermes_service/services/utterance_service.py
import json
import logging
from typing import List, Dict, Any
from ..schemas.utterance import UtteranceProcessIn, UtteranceProcessOut
from ...core.database import get_db_cursor
from ...hermes_service.graph import app as langgraph_app 
from ...hermes_service.state import TranslateState 
import time

log = logging.getLogger(__name__)

def _get_chat_history(meeting_id: int, limit: int = 10) -> List[str]:
    """DB에서 최근 대화 기록을 가져옵니다."""
    history = []
    try:
        with get_db_cursor(dictionary=True) as cursor:
            sql = """
                SELECT text 
                FROM tl_utterances 
                WHERE meeting_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """
            cursor.execute(sql, (meeting_id, limit))
            results = cursor.fetchall()
            history = [row['text'] for row in reversed(results)]
    except Exception as e:
        log.error(f"❌ Chat History 조회 실패: {e}")
    return history

def _run_contextual_translation(state_input: TranslateState) -> TranslateState:
    """LangGraph를 실행하여 문맥 보정 번역을 수행합니다."""
    try:
        final_state = langgraph_app.invoke(state_input)
        return final_state
    except Exception as e:
        log.error(f"❌ LangGraph 실행 실패: {e}")
        state_input['final_answer'] = state_input['translated_text']
        return state_input

def _save_utterance(payload: UtteranceProcessIn, final_text_content: str, ai_tags: Dict[str, Any]):
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
            params = (
                payload.meeting_id,
                payload.speaker_id,
                payload.track_sid,
                seq_val,
                payload.started_ms,
                payload.ended_ms,
                payload.lang,
                final_text_content,
                payload.confidence,
                json.dumps(ai_tags) 
            )
            cursor.execute(sql, params)
    except Exception as e:
        log.error(f"❌ Utterance DB 저장 실패: {e}")


def process_and_save_utterance(payload: UtteranceProcessIn) -> UtteranceProcessOut:
    """
    (Service) 발화 처리의 메인 로직 (⭐️ 듀얼 모드 지원)
    """
    log.info(f"발화 처리 시작 (Mode: {payload.translation_mode}, STT: {payload.stt_text})")

    # --- 1. DB에서 Chat History 조회 ---
    chat_history = _get_chat_history(payload.meeting_id)

    # --- 2. Graph State 준비 ---
    
    # ⭐️ [핵심 수정] 
    # 1차 번역본과 대상 언어를 Agent가 보낸 payload에서 직접 가져옵니다.
    # 더 이상 "en-US"를 하드코딩하거나 추측하지 않습니다.
    
    target_lang = payload.target_lang
    simple_translation = payload.simple_translations.get(target_lang, "") # Vibe 모드이거나 B-모드 FastPath가 아니면 비어있음

    if payload.translation_mode == "business" and not simple_translation:
        log.warning(f"Business 모드이지만 1차 번역본(target_lang: {target_lang})이 비어있습니다.")


    # LangGraph에 넣을 State 객체 준비
    graph_input_state = TranslateState(
        stt_text=payload.stt_text,
        translated_text=simple_translation, 
        
        stt_lang=payload.lang,
        target_lang=target_lang,

        context=[], 
        meeting_id=payload.meeting_id,
        docs_id=None, 
        chat_history=chat_history,
        found_terms=[], 
        
        translation_mode=payload.translation_mode, 
        should_correct_business=False, 
        vibe_emotion_tag=None,         
        vibe_triggers=None,            
        final_answer="",               
        final_explanation=None,        
        final_emoji=None,              
    )

    # --- 3. 그래프 실행 ---
    final_state = _run_contextual_translation(graph_input_state)
    
    # 번역 결과 (Target Lang)
    corrected_answer = final_state.get("final_answer")
    if not corrected_answer:
        corrected_answer = simple_translation 

    # ⭐️ [핵심 수정] DB에 저장할 '본문 텍스트' 결정 로직
    # 1순위: LangGraph가 보정한 원문 (corrected_stt) - 주로 Vibe 모드
    # 2순위: 원본 STT (stt_text) - Business 모드(현재 보정 로직 없음) 또는 보정 실패 시
    corrected_source_text = final_state.get("corrected_stt")
    
    if corrected_source_text and corrected_source_text.strip():
        text_to_save = corrected_source_text
        log.info(f"📝 DB 저장: 보정된 STT를 사용합니다. ('{payload.stt_text}' -> '{text_to_save}')")
    else:
        text_to_save = payload.stt_text
        log.info(f"📝 DB 저장: 원본 STT를 그대로 사용합니다. ('{text_to_save}')")

    # --- 4. DB 저장 ---
    ai_tags = {
        "mode": payload.translation_mode,
        "target_lang": target_lang, 
        "simple_translations": payload.simple_translations,
        "final_answer": corrected_answer, # 번역 결과는 태그에 저장
        "original_stt": payload.stt_text, # 🟢 [추가] 원본 STT는 ai_tags에 백업해둠 (디버깅용)
        # ... (기존 태그들)
        "vibe_emotion": final_state.get("vibe_emotion_tag"),
        "vibe_triggers": final_state.get("vibe_triggers"),
        "vibe_explanation": final_state.get("final_explanation"),
        "vibe_emoji": final_state.get("final_emoji"),
        "biz_found_terms": final_state.get("found_terms"),
        "biz_rag_context_count": len(final_state.get("context", [])),
    }
    
    # 수정된 _save_utterance 호출 (text_to_save 전달)
    _save_utterance(payload, text_to_save, ai_tags)

    # --- 5. livekit_agent로 결과 반환 ---
    final_translations = payload.simple_translations.copy()
    
    # [수정] LLM이 보정한 'target_lang'의 값만 덮어씁니다.
    final_translations[target_lang] = corrected_answer 

    return UtteranceProcessOut(
        stt_text=payload.stt_text,
        lang=payload.lang,
        contextual_translations=final_translations,
        final_explanation=final_state.get("final_explanation"),
        final_emoji=final_state.get("final_emoji"),
        corrected_stt=final_state.get("corrected_stt"),
        vibe_triggers=final_state.get("vibe_triggers"),
        ai_tags=ai_tags
    )
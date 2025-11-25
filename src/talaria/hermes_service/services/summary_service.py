# src/talaria/hermes_service/services/summary_service.py

import logging
import time
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple  # 🟢 Tuple 추가

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..schemas.summary import MeetingSummaryIn, MeetingSummaryOut
from ..summary_graph import app as summary_graph
from ...core.database import get_db_cursor
from ..utils.llm_factory import get_llm_factory

log = logging.getLogger(__name__)

# --- [1] 날짜 포맷팅 헬퍼 (LLM용) ---
def _format_friendly_date(iso_date_str: str) -> str:
    """ISO 8601 문자열을 LLM용 'YYYY년 MM월 DD일 (요일)'로 변환"""
    if not iso_date_str:
        return datetime.now().strftime("%Y년 %m월 %d일")
    try:
        dt = datetime.fromisoformat(iso_date_str)
    except ValueError:
        try:
            dt = datetime.strptime(iso_date_str, "%Y-%m-%d")
        except ValueError:
            return iso_date_str

    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    day_str = weekdays[dt.weekday()]
    return f"{dt.year}년 {dt.month}월 {dt.day}일 ({day_str})"

# --- [2] DB: 대상 언어 조회 ---
def _get_target_locales(meeting_id: int) -> List[str]:
    """
    tl_meeting_participants + tl_users 테이블을 조인하여
    해당 회의 참여자들의 언어(locale) 목록을 중복 없이 가져옵니다.
    """
    target_locales = set()
    try:
        with get_db_cursor(dictionary=True) as cursor:
            sql = """
                SELECT DISTINCT u.locale
                FROM tl_meeting_participants mp
                JOIN tl_users u ON mp.user_id = u.user_id
                WHERE mp.meetings_id = %s
            """
            cursor.execute(sql, (meeting_id,))
            results = cursor.fetchall()
            
            for row in results:
                loc = row.get('locale', 'ko')
                # 한국어(ko)는 이미 원본이 있으므로 번역 대상에서 제외
                if loc != 'ko': 
                    target_locales.add(loc)
                    
    except Exception as e:
        log.error(f"[{meeting_id}] 참여자 언어 조회 실패: {e}")
        return []

    return list(target_locales)

# --- [3] LLM: JSON 구조 유지 번역 (제목 포함) ---
async def _translate_summary_json(
    original_json: Dict[str, Any], 
    meeting_title: str,          # 🟢 1. 인자로 제목 받음
    target_lang_code: str
) -> Tuple[Dict[str, Any], str]: # 🟢 2. 리턴 타입 변경 (JSON, Title)
    """
    LLM을 사용하여 요약 결과 JSON의 '값(Value)'과 '회의 제목'을 타겟 언어로 번역합니다.
    """
    # 언어 코드 매핑
    lang_map = {
        "en": "English (US)",
        "ja": "Japanese (Business Tone)",
        "zh": "Simplified Chinese (zh-CN)", 
        "vi": "Vietnamese",
        "th": "Thai"
    }
    target_lang_name = lang_map.get(target_lang_code, target_lang_code)
    
    log.info(f"🌍 번역 시작: {target_lang_code} ({target_lang_name})")

    try:
        llm = get_llm_factory().get_llm(model_name="gpt-4o-mini", temperature=0.0)
        
        # 🟢 3. JSON에 제목을 임시로 포함시켜 한 번에 번역 요청
        input_data = original_json.copy()
        input_data["meeting_title"] = meeting_title 

        prompt = ChatPromptTemplate.from_template("""
        You are a professional translator for business meeting minutes.
        Translate the VALUES of the following JSON into {target_lang}.
        
        [Rules]
        1. KEEP all JSON Keys exactly the same. Do NOT translate keys.
        2. Translate only the string values.
        3. Maintain the professional nuance of the original text.
        4. Output MUST be a valid JSON.

        [Source JSON]
        {json_data}
        """)
        
        chain = prompt | llm | JsonOutputParser()
        
        translated_full_json = await chain.ainvoke({
            "target_lang": target_lang_name,
            "json_data": json.dumps(input_data, ensure_ascii=False)
        })
        
        # 🟢 4. 번역된 결과에서 제목 분리
        # (만약 번역 중 키가 사라졌으면 원본 제목 사용)
        translated_title = translated_full_json.pop("meeting_title", meeting_title)
        
        log.info(f"✅ 번역 완료: {target_lang_code} (Title: {translated_title})")
        return translated_full_json, translated_title

    except Exception as e:
        log.error(f"❌ 번역 실패 ({target_lang_code}): {e}")
        # 실패 시 원본 그대로 반환
        return original_json, meeting_title 

# --- [4] DB: 요약본 저장 (메인 로직) ---
def _save_summary_to_db(
    meeting_id: int,
    lang: str,
    iso_date_str: str,
    meeting_title: str,           # 🟢 5. 번역된 제목을 인자로 받음
    participants_list: List[str], # 🟢 6. (중요) 얘는 번역 안 된 원본 리스트 그대로 씀
    summary_data: Dict[str, Any]
):
    """
    tl_meeting_summaries 테이블에 저장합니다.
    """
    try:
        # 1. 데이터 파싱
        try:
            if iso_date_str:
                meeting_date = datetime.fromisoformat(iso_date_str).date()
            else:
                meeting_date = datetime.now().date()
        except ValueError:
            meeting_date = datetime.now().date()

        # 구조 분해
        ai_summary = summary_data.get("aiSummary", {})
        ai_coach = summary_data.get("aiCoach", {})
        topics = summary_data.get("topics", [])

        # 2. DB Insert
        with get_db_cursor(commit=True) as cursor:
            sql = """
                INSERT INTO tl_meeting_summaries (
                    meeting_id, lang, meeting_date, meeting_title, 
                    participants, participant_count,
                    summary, meeting_score, risk_level,
                    topics, decisions, action_items, 
                    risks, improvements, next_steps,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, 
                    %s, %s, 
                    %s, %s, %s, 
                    %s, %s, %s, 
                    %s, %s, %s,
                    NOW(), NOW()
                )
                ON DUPLICATE KEY UPDATE
                    meeting_title = VALUES(meeting_title),
                    participants = VALUES(participants),
                    participant_count = VALUES(participant_count),
                    summary = VALUES(summary),
                    meeting_score = VALUES(meeting_score),
                    risk_level = VALUES(risk_level),
                    topics = VALUES(topics),
                    decisions = VALUES(decisions),
                    action_items = VALUES(action_items),
                    risks = VALUES(risks),
                    improvements = VALUES(improvements),
                    next_steps = VALUES(next_steps),
                    updated_at = NOW()
            """
            
            params = (
                meeting_id,
                lang,
                meeting_date,
                meeting_title, # 🟢 번역된 제목 저장
                json.dumps(participants_list, ensure_ascii=False), # 🟢 원본 참가자 (번역 X)
                len(participants_list),
                
                ai_coach.get("summary", ""),
                ai_coach.get("meetingScore", 0),
                ai_coach.get("riskLevel", "보통"),
                
                json.dumps(topics, ensure_ascii=False),
                json.dumps(ai_summary.get("decisions", []), ensure_ascii=False),
                json.dumps(ai_summary.get("actionItems", []), ensure_ascii=False),
                
                json.dumps(ai_coach.get("risks", []), ensure_ascii=False),
                json.dumps(ai_coach.get("improvements", []), ensure_ascii=False),
                json.dumps(ai_coach.get("nextSteps", []), ensure_ascii=False)
            )
            
            cursor.execute(sql, params)
            log.info(f"💾 DB 저장 완료 (Meeting: {meeting_id}, Lang: {lang})")

    except Exception as e:
        log.error(f"❌ DB 저장 실패 (Meeting: {meeting_id}, Lang: {lang}): {e}", exc_info=True)


# --- [Main] 메인 프로세스 함수 ---
async def process_meeting_summary(payload: MeetingSummaryIn) -> MeetingSummaryOut:
    
    participants_str = payload.participants
    
    formatted_text = f"""[회의 정보]
회의 제목: {payload.meeting_title}
참석자: {participants_str}

[스크립트 본문]
{payload.transcript}"""

    if payload.participants:
        participant_list = [p.strip() for p in payload.participants.split(",") if p.strip()]
    else:
        participant_list = []
    
    friendly_date_str = _format_friendly_date(payload.current_date)
    log.info(f"[{payload.meeting_id}] 날짜 매핑: '{payload.current_date}' -> '{friendly_date_str}'")

    initial_state = {
        "transcript": formatted_text, 
        "current_date": friendly_date_str, 
        "participants": participant_list,
        "summary_intro": "", "keywords": [], "decisions": [], "track_a_feedback": "", "track_a_retry": 0,
        "action_items": [], "track_b_feedback": "", "track_b_retry": 0,
        "content_score": 0.0, "content_risks": [], "content_improvements": [],
        "process_score": 0.0, "process_risks": [], "coach_next_steps": [],
        "final_output": {}
    }

    # 2. [실행] LangGraph 실행 (한국어 원본 생성)
    log.info(f"[{payload.meeting_id}] 🇰🇷 한국어 요약 생성 시작...")
    final_state = await summary_graph.ainvoke(initial_state)
    korean_result = final_state.get("final_output", {})
    
    # 3. [저장] 한국어 원본 DB 저장 (제목 원본 사용)
    _save_summary_to_db(
        meeting_id=payload.meeting_id,
        lang="ko",
        iso_date_str=payload.current_date,
        meeting_title=payload.meeting_title, # 🟢 원본 제목
        participants_list=participant_list,  # 🟢 원본 참가자
        summary_data=korean_result
    )

    # 4. [번역] 타겟 언어 조회 및 병렬 번역 실행
    target_langs = _get_target_locales(payload.meeting_id)
    
    if target_langs:
        log.info(f"[{payload.meeting_id}] 🌍 다국어 번역 대상: {target_langs}")
        
        # 🟢 [수정] 각 언어별 번역 Task (제목도 인자로 전달)
        translation_tasks = [
            _translate_summary_json(korean_result, payload.meeting_title, lang) 
            for lang in target_langs
        ]
        
        # 병렬 실행 (Wait All) -> (Data, Title) 튜플 리스트 반환됨
        translated_results = await asyncio.gather(*translation_tasks)
        
        # 5. [저장] 번역본 DB 저장
        for lang, (translated_data, translated_title) in zip(target_langs, translated_results):
            _save_summary_to_db(
                meeting_id=payload.meeting_id,
                lang=lang,
                iso_date_str=payload.current_date,
                meeting_title=translated_title, # 🟢 번역된 제목
                participants_list=participant_list, # 🟢 원본 참가자 (번역X)
                summary_data=translated_data
            )
    else:
        log.info(f"[{payload.meeting_id}] 번역 대상 언어가 없습니다.")

    meta_info = {
        "meeting_id": payload.meeting_id,
        "status": "success",
        "generated_at": str(int(time.time())),
        "saved_languages": ["ko"] + target_langs
    }

    return MeetingSummaryOut(
        meta=meta_info,
        result=korean_result
    )
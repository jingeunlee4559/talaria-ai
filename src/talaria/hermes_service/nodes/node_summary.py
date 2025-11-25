# src/talaria/hermes_service/nodes/node_summary.py
import logging
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any

# 🟢 [변경] 팩토리 임포트
from ..utils.llm_factory import get_llm_factory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..state import SummaryState

log = logging.getLogger(__name__)

# ==========================================
# 🟢 Track A: 요약 & 키워드 (Summary)
# ==========================================
def node_track_a_generate(state: SummaryState):
    log.info("🟢 [Track A] 요약 및 결정사항 생성 시작...")
    
    # 팩토리를 통해 LLM 가져오기 (온도 0.0)
    try:
        llm = get_llm_factory().get_llm(temperature=0.0)
    except Exception:
        return {"track_a_feedback": "LLM Load Failed"}

    prompt = ChatPromptTemplate.from_template("""
    당신은 전문 회의 서기입니다. 회의록을 분석하여 다음 JSON 포맷으로 출력하세요.
    
    [요구사항]
    1. summary_intro: 회의 전체를 관통하는 한 줄 요약 (명확하게).
    2. keywords: 핵심 주제 명사 4개.
    3. decisions: '확정된' 의사결정 사항만 3개 이내로 추출 (논의 중인 것은 제외).
    4. 언어: 한국어
    
    [회의록]
    {transcript}
    
    [피드백 반영(있을 경우)]: {feedback}
    """)
    
    chain = prompt | llm | JsonOutputParser()
    try:
        res = chain.invoke({
            "transcript": state["transcript"],
            "feedback": state.get("track_a_feedback", "")
        })
        return {
            "summary_intro": res.get("summary_intro", ""),
            "keywords": res.get("keywords", []),
            "decisions": res.get("decisions", [])
        }
    except Exception as e:
        log.error(f"Track A Error: {e}")
        return {"track_a_feedback": "JSON Formatting Error", "track_a_retry": state.get("track_a_retry", 0) + 1}

def node_track_a_eval_structure(state: SummaryState):
    """평가 1: 구조/누락 검사"""
    if not state.get("summary_intro") or not state.get("decisions"):
        return {"track_a_feedback": "필수 필드 누락", "track_a_retry": state.get("track_a_retry", 0) + 1}
    return {"track_a_feedback": ""}

def node_track_a_eval_quality(state: SummaryState):
    """평가 2: 퀄리티(길이 등) 검사"""
    if state.get("track_a_feedback"): return {}
    
    if len(state["summary_intro"]) > 150:
        return {"track_a_feedback": "요약이 너무 깁니다. 100자 이내로 줄이세요.", "track_a_retry": state.get("track_a_retry", 0) + 1}
    
    log.info(f"✅ [Track A] 요약 완료 (결정사항 {len(state['decisions'])}개)")
    return {"track_a_feedback": None}

# ==========================================
# 🔵 Track B: 액션 아이템 (Action Items)
# ==========================================
def node_track_b_extract(state: SummaryState):
    log.info("🔵 [Track B] 액션 아이템 추출 시작...")
    
    try:
        llm = get_llm_factory().get_llm(temperature=0.0)
    except Exception:
        return {"track_b_feedback": "LLM Load Failed"}
    
    # 🟢 [수정] 프롬프트에 priority 판단 로직 추가
    prompt = ChatPromptTemplate.from_template("""
    회의록에서 '실행 과제(Action Item)'를 추출하여 JSON 리스트로 반환하세요.
    
    [매우 중요: 날짜 계산]                                         
    [기준 날짜] {current_date}
    - 본문에서 "내일", "다음 주 수요일" 같은 표현이 나오면 위 기준 날짜를 바탕으로 정확한 날짜(YYYY-MM-DD)로 계산해 적으세요.
    
    [추출 가이드]
    1. owner: 담당자 이름 (없으면 '미정')
    2. task: 할 일 (명사형 종결)
    3. deadline: 마감일 (문맥상 추론하여 'YYYY-MM-DD' 또는 'MM월 DD일' 포맷)
    4. priority: 업무의 시급성과 중요도를 판단하여 "HIGH", "MEDIUM", "LOW" 중 하나 선택.
       - 당장 이번 주 내로 해야 하거나, 장애/보안 이슈면 "HIGH"
       - 일반적인 개발/기획 업무는 "MEDIUM"
       - 나중에 해도 되는 것은 "LOW"
    
    [회의록]
    {transcript}
    
    [피드백 반영]: {feedback}
    """)
    
    chain = prompt | llm | JsonOutputParser()
    try:
        res = chain.invoke({
            "transcript": state["transcript"],
            "current_date": state["current_date"],
            "feedback": state.get("track_b_feedback", "")
        })
        items = res if isinstance(res, list) else res.get("action_items", [])
        return {"action_items": items}
    except Exception as e:
        log.error(f"Track B Error: {e}")
        return {"track_b_feedback": "JSON Parsing Error", "track_b_retry": state.get("track_b_retry", 0) + 1}
    
def node_track_b_eval_schema(state: SummaryState):
    items = state.get("action_items", [])
    if not items: return {"track_b_feedback": None}
    
    for item in items:
        if "owner" not in item or "task" not in item:
            return {"track_b_feedback": "필드 누락 (owner, task 필수)", "track_b_retry": state.get("track_b_retry", 0) + 1}
    return {"track_b_feedback": ""}

def node_track_b_eval_logic(state: SummaryState):
    if state.get("track_b_feedback"): return {}
    log.info(f"✅ [Track B] 액션 아이템 {len(state['action_items'])}개 추출 완료.")
    return {"track_b_feedback": None}

# ==========================================
# 🟣 Track C: 리스크 분석 (2-Way Hybrid)
# ==========================================

# --- C-1: 내용 분석 (LLM) ---
def node_analyze_content(state: SummaryState):
    log.info("🟣 [Track C-1] 내용 기반 리스크 분석 (Clarity, Resolution)...")
    try:
        llm = get_llm_factory().get_llm(temperature=0.0)
    except Exception:
        return {"content_score": 0}

    actions_str = json.dumps(state.get("action_items", []), ensure_ascii=False)
    
    prompt = ChatPromptTemplate.from_template("""
    당신은 '조직 행동론 전문가'입니다. 회의 내용과 도출된 액션 아이템을 평가하세요.
    
    [평가 기준]
    1. 실행 명확성(40점): 액션 아이템({actions})에 담당자와 기한이 명확한가? 모호한 표현("나중에", "검토")이 있으면 감점.
    2. 갈등 해결(20점): 회의록 내 충돌이 합의로 끝났는가? 유야무야 넘어갔으면 감점.
    
    [출력 포맷 JSON]
    {{
        "clarity_score": (0~40점),
        "resolution_score": (0~20점),
        "risks": ["감지된 리스크1", "리스크2"],
        "improvements": ["개선책1", "개선책2"],
        "next_steps": ["전략적 제안1", "제안2"]
    }}
    
    [회의록]
    {transcript}
    """)
    
    chain = prompt | llm | JsonOutputParser()
    try:
        res = chain.invoke({"transcript": state["transcript"], "actions": actions_str})
        return {
            "content_score": res.get("clarity_score", 30) + res.get("resolution_score", 15),
            "content_risks": res.get("risks", []),
            "content_improvements": res.get("improvements", []),
            "coach_next_steps": res.get("next_steps", [])
        }
    except Exception as e:
        log.error(f"Content Analysis Error: {e}")
        return {"content_score": 45, "content_risks": ["분석 중 오류 발생"]}

# --- C-2: 프로세스 분석 (Python) ---
def node_analyze_process(state: SummaryState):
    log.info("🟣 [Track C-2] 프로세스 데이터 분석 (Balance, Efficiency)...")
    
    transcript = state["transcript"]
    
    # 1. 소통 균형 (Balance, 20점) - 단순 글자수/화자 카운트
    # (참가자 정보가 없으면 기본값)
    balance_score = 18 
    
    # 2. 시간 효율 (Efficiency, 20점) - 길이 기반
    efficiency_score = 17
    
    process_risks = []
    if len(transcript) < 50:
        process_risks.append("회의 내용이 너무 짧아 심도 있는 논의가 부족했습니다.")
        efficiency_score = 10

    return {
        "process_score": balance_score + efficiency_score,
        "process_risks": process_risks
    }

# ==========================================
# 🏁 Final Merger
# ==========================================
def node_final_merger(state: SummaryState):
    log.info("🏁 [Final] 최종 결과 병합 및 필터링 중...")
    
    # 1. 점수 합산 및 리스크 레벨 산정 (기존 로직)
    total_score = int(state.get("content_score", 0) + state.get("process_score", 0))
    total_score = min(100, max(0, total_score))
    
    if total_score >= 80: level = "낮음"
    elif total_score >= 60: level = "보통"
    else: level = "높음"
    
    # 2. 🟢 [추가] 액션 아이템 정렬 및 필터링 (Top 3)
    raw_items = state.get("action_items", [])
    
    # 우선순위 매핑 (정렬을 위해 숫자 부여)
    priority_map = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    
    def sort_key(item):
        # 1순위: Priority (High < Medium < Low)
        p_score = priority_map.get(item.get("priority", "MEDIUM").upper(), 1)
        # 2순위: Deadline (문자열이라 완벽하진 않지만, 없는 것보단 나음. 필요 시 날짜 파싱 로직 추가 가능)
        d_score = item.get("deadline", "9999") 
        return (p_score, d_score)

    # 정렬 후 상위 3개만 슬라이싱
    sorted_items = sorted(raw_items, key=sort_key)
    top_3_items = sorted_items[:3]
    
    log.info(f"📋 액션 아이템 필터링: 전체 {len(raw_items)}개 -> 상위 {len(top_3_items)}개 선택됨.")

    # 3. 최종 JSON 조립
    all_risks = state.get("content_risks", []) + state.get("process_risks", [])
    
    result = {
        "topics": state.get("keywords", []),
        "aiSummary": {
            "decisions": state.get("decisions", []),
            "actionItems": top_3_items  # 🟢 필터링된 리스트 사용
        },
        "aiCoach": {
            "summary": state.get("summary_intro", ""),
            "meetingScore": total_score,
            "riskLevel": level,
            "risks": all_risks[:3],
            "improvements": state.get("content_improvements", [])[:2],
            "nextSteps": state.get("coach_next_steps", [])[:3]
        }
    }
    
    return {"final_output": result}
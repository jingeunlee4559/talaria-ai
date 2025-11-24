import logging
import json
import re
import ast
from datetime import datetime
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from ..state import SummaryState

log = logging.getLogger(__name__)

# --- LLM 초기화 (OpenAI) ---
try:
    # gpt-4o-mini 사용 (비용 효율성 및 성능 고려)
    llm = ChatOpenAI(model="gpt-5", temperature=0.1, max_retries=2)
    log.info("✅ 요약용 LLM (gpt-5) 로드 완료.")
except Exception as e:
    log.critical(f"❌ 요약용 LLM 로드 실패: {e}")
    llm = None

# --- 유틸리티: JSON 파서 ---
def clean_and_parse_json(text: str) -> dict:
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return ast.literal_eval(match.group(1))
        except Exception:
            pass
    raise ValueError(f"JSON parsing failed: {text[:50]}...")

# --- 프롬프트 템플릿 정의 ---
SUMMARY_PROMPT = PromptTemplate(
    template="""
당신은 AI 회의 서기입니다. 아래 회의록을 바탕으로 [1. 핵심 요약, 2. 주요 결정 사항] 형식의 한국어 보고서를 작성하세요. 
참석자 목록이나 세부 할 일(To-do 리스트)은 작성하지 마세요.

[입력 데이터]
{transcript}

[작성 가이드]
1. 명확하고 간결한 비즈니스 톤을 사용하세요.
{feedback_section}
반드시 한국어로 작성하세요.
""",
    input_variables=["transcript", "feedback_section"]
)

EXTRACTION_PROMPT = PromptTemplate(
    template="""
회의록을 분석하여 [액션 아이템(action_items)]과 [다음 회의 제안(smart_trigger)] 정보를 JSON 형식으로 추출하세요. 
날짜는 '기준 날짜'({current_date})를 바탕으로 계산한 YYYY-MM-DD 형식으로 작성하세요.

[회의록]
{transcript}

[추출 가이드]
1. action_items: owner_name, text, due_date(YYYY-MM-DD), priority(HIGH/MEDIUM/LOW)
2. smart_trigger: next_meeting_intent(bool), suggested_schedule(topic, date, time, reason)
3. 오직 순수한 JSON 문자열만 출력하세요.

{feedback_section}
""",
    input_variables=["transcript", "current_date", "feedback_section"]
)

# 평가용 스키마
class EvalScore(BaseModel):
    reasoning: str = Field(description="이유")
    score: int = Field(description="점수 (1:통과, 0:실패)")

FACT_CHECK_PROMPT = PromptTemplate(
    template="""[팩트 체크] 원문과 요약문을 비교하여 거짓 정보가 있으면 score 0, 없으면 1을 주세요.
원문: {transcript}
요약: {summary}
JSON 형식: {{"reasoning": "...", "score": 0/1}}""",
    input_variables=["transcript", "summary"]
)

COMPLETENESS_PROMPT = PromptTemplate(
    template="""[누락 검사] 요약문에 핵심 결정 사항이 빠졌으면 score 0, 충분하면 1을 주세요.
원문: {transcript}
요약: {summary}
JSON 형식: {{"reasoning": "...", "score": 0/1}}""",
    input_variables=["transcript", "summary"]
)

CONSISTENCY_PROMPT = PromptTemplate(
    template="""[정합성 검사] JSON 데이터의 날짜와 담당자가 원문(기준일:{current_date})과 일치하는지 확인하세요.
원문: {transcript}
JSON: {json_output}
오류가 있으면 score 0, 없으면 1.
JSON 형식: {{"reasoning": "...", "score": 0/1}}""",
    input_variables=["transcript", "json_output", "current_date"]
)


# --- Node Functions ---

def node_generate_summary(state: SummaryState):
    log.info("📝 [Track A] 요약문 생성 중...")
    transcript = state["input_transcript"]
    feedback = state.get("summary_feedback", "")
    
    feedback_section = f"\n[이전 피드백 반영]: {feedback}\n" if feedback else ""
    
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    response = chain.invoke({"transcript": transcript, "feedback_section": feedback_section})
    return {"summary_text": response}

def node_fact_check(state: SummaryState):
    log.info("🕵️ [Track A] 팩트 체크 중...")
    parser = JsonOutputParser(pydantic_object=EvalScore)
    chain = FACT_CHECK_PROMPT | llm | parser
    
    try:
        res = chain.invoke({"transcript": state["input_transcript"], "summary": state["summary_text"]})
        if res["score"] == 0:
            return {"summary_feedback": f"팩트 체크 실패: {res['reasoning']}", "summary_retry_count": state["summary_retry_count"] + 1}
    except Exception:
        pass # 파싱 에러 시 통과
    return {"summary_feedback": ""}

def node_completeness_check(state: SummaryState):
    if state.get("summary_feedback"): return {}
    log.info("⚖️ [Track A] 내용 충실도 검사 중...")
    parser = JsonOutputParser(pydantic_object=EvalScore)
    chain = COMPLETENESS_PROMPT | llm | parser
    try:
        res = chain.invoke({"transcript": state["input_transcript"], "summary": state["summary_text"]})
        if res["score"] == 0:
            return {"summary_feedback": f"내용 누락: {res['reasoning']}", "summary_retry_count": state["summary_retry_count"] + 1}
    except Exception:
        pass
    return {"summary_feedback": ""}

def node_extract_json(state: SummaryState):
    log.info("⛏️ [Track B] JSON 데이터 추출 중...")
    transcript = state["input_transcript"]
    current_date = state["metadata"].get("current_date", "")
    feedback = state.get("extraction_feedback", "")
    
    feedback_section = f"\n[피드백]: {feedback} 형식을 수정하세요.\n" if feedback else ""
    
    chain = EXTRACTION_PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "transcript": transcript, 
        "current_date": current_date, 
        "feedback_section": feedback_section
    })
    return {"json_output": response}

def node_validate_syntax(state: SummaryState):
    log.info("🔍 [Track B] JSON 문법 검사 중...")
    raw = state["json_output"]
    try:
        parsed = clean_and_parse_json(raw) if isinstance(raw, str) else raw
        return {"json_output": parsed, "extraction_feedback": ""}
    except Exception as e:
        return {"extraction_feedback": f"JSON 파싱 실패: {e}", "extraction_retry_count": state["extraction_retry_count"] + 1}

def node_consistency_check(state: SummaryState):
    if state.get("extraction_feedback"): return {}
    log.info("🧐 [Track B] 데이터 정합성 검사 중...")
    parser = JsonOutputParser(pydantic_object=EvalScore)
    chain = CONSISTENCY_PROMPT | llm | parser
    try:
        json_str = json.dumps(state["json_output"], ensure_ascii=False)
        res = chain.invoke({
            "transcript": state["input_transcript"], 
            "json_output": json_str,
            "current_date": state["metadata"].get("current_date")
        })
        if res["score"] == 0:
            return {"extraction_feedback": f"데이터 불일치: {res['reasoning']}", "extraction_retry_count": state["extraction_retry_count"] + 1}
    except Exception:
        pass
    return {"extraction_feedback": ""}

def node_final_merger(state: SummaryState):
    log.info("🏁 [Merger] 최종 결과 병합 중...")
    json_data = state.get("json_output", {})
    if isinstance(json_data, str): json_data = {}
    
    final_response = {
        "meta": {
            "status": "success",
            "model": "gpt-4o-mini",
            "generated_at": datetime.now().isoformat()
        },
        "result": {
            "summary": {"report_text": state.get("summary_text", "요약 실패")},
            "action_items": json_data.get("action_items", []),
            "smart_trigger": json_data.get("smart_trigger", {})
        }
    }
    return {"final_response": final_response}
# src/talaria/hermes_service/graph.py
from langgraph.graph import StateGraph, END
from .state import TranslateState
import logging 
from typing import Literal

log = logging.getLogger(__name__)

# --- 1. 모든 노드 함수 임포트 ---

# [Flow A] Business Mode Nodes
from .nodes.node_check_conditions import check_conditions_node as analyze_business_context
from .nodes.node_context_translate import context_translate_node as correction_agent

# ⭐️ [수정] [Flow B] Vibe Mode Nodes (2-LLM Call 아키텍처)
from .nodes.node_vibe_extract import vibe_extract_agent # (LLM 1)
from .nodes.node_vibe_enrich import context_enrich_agent, emoji_agent # (Rules)
from .nodes.node_vibe_translate import vibe_translate_agent # (LLM 2)

# --- 2. 라우터 및 Fallback 노드 정의 (동일) ---

def fallback_translate_node(state: TranslateState) -> TranslateState:
    log.info(f"Biz 모드 (Fast Path): LLM 보정 건너뜀 (STT: '{state['stt_text']}')")
    state['final_answer'] = state['translated_text'] 
    return state

def mode_router(state: TranslateState) -> str:
    mode = state.get("translation_mode", "business") 
    log.info(f"Mode Router: '{mode}' 플로우로 분기")
    return mode 

def business_correction_router(state: TranslateState) -> str:
    if state.get("should_correct_business"): 
        log.info("Biz Router: 보정 필요(correct) -> correction_agent")
        return "correct"
    else:
        log.info("Biz Router: 보정 불필요(fallback) -> standard_fallback_agent")
        return "fallback"

# --- 3. 그래프 조립 ---

workflow = StateGraph(TranslateState)

# [A] Business Mode 노드 추가 (동일)
workflow.add_node("analyze_business_context", analyze_business_context)
workflow.add_node("correction_agent", correction_agent)
workflow.add_node("standard_fallback_agent", fallback_translate_node) 

# ⭐️ [수정] [B] Vibe Mode 노드 추가 (4개 노드)
workflow.add_node("vibe_extract_agent", vibe_extract_agent)
workflow.add_node("context_enrich_agent", context_enrich_agent)
workflow.add_node("emoji_agent", emoji_agent)
workflow.add_node("vibe_translate_agent", vibe_translate_agent)

# --- 4. 엣지(흐름) 설정 ---

# [1. 진입점 설정] (⭐️ Vibe 모드 진입점을 'vibe_extract_agent'로 변경)
workflow.set_conditional_entry_point(
    mode_router, 
    {
        "business": "analyze_business_context",
        "vibe": "vibe_extract_agent" # ⭐️ (수정) Vibe 모드의 첫 시작
    }
)

# [3. Flow A: Business Mode 흐름] (동일)
workflow.add_conditional_edges(
    "analyze_business_context", 
    business_correction_router, 
    {
        "correct": "correction_agent",
        "fallback": "standard_fallback_agent"
    }
)
workflow.add_edge("correction_agent", END)
workflow.add_edge("standard_fallback_agent", END)

# ⭐️ [4. Flow B: Vibe Mode 흐름] (신규 2-LLM Call 플로우)
# (LLM 1) -> (규칙 1) -> (규칙 2) -> (LLM 2) -> (종료)
workflow.add_edge("vibe_extract_agent", "context_enrich_agent")
workflow.add_edge("context_enrich_agent", "emoji_agent")
workflow.add_edge("emoji_agent", "vibe_translate_agent")
workflow.add_edge("vibe_translate_agent", END)

# --- 5. 그래프 컴파일 ---
app = workflow.compile()
log.info("✅ (수정) 8-Node 듀얼 모드(Business/Vibe 2-LLM) LangGraph 컴파일 완료.")
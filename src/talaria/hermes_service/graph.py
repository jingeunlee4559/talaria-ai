from langgraph.graph import StateGraph, END
from .state import TranslateState
import logging 

log = logging.getLogger(__name__)

# --- 1. 모든 노드 함수 임포트 ---
from .nodes.node_check_conditions import check_conditions_node as analyze_business_context
from .nodes.node_context_translate import context_translate_node as correction_agent

# [Flow B] Vibe Mode Nodes (자체 모델 + VectorDB)
from .nodes.node_vibe_extract import vibe_extract_agent
from .nodes.node_vibe_enrich import context_enrich_agent
from .nodes.node_vibe_translate import vibe_translate_agent
# emoji_agent는 제거됨

# --- 2. 라우터 및 Fallback 노드 정의 ---
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
        return "correct"
    else:
        return "fallback"

# --- 3. 그래프 조립 ---
workflow = StateGraph(TranslateState)

# [A] Business Mode 노드
workflow.add_node("analyze_business_context", analyze_business_context)
workflow.add_node("correction_agent", correction_agent)
workflow.add_node("standard_fallback_agent", fallback_translate_node) 

# [B] Vibe Mode 노드 (Emoji 제거됨)
workflow.add_node("vibe_extract_agent", vibe_extract_agent)
workflow.add_node("context_enrich_agent", context_enrich_agent)
workflow.add_node("vibe_translate_agent", vibe_translate_agent)

# --- 4. 엣지(흐름) 설정 ---

# [진입점]
workflow.set_conditional_entry_point(
    mode_router, 
    {
        "business": "analyze_business_context",
        "vibe": "vibe_extract_agent"
    }
)

# [Flow A]
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

# [Flow B] 수정된 흐름 (Extract -> Enrich -> Translate -> END)
workflow.add_edge("vibe_extract_agent", "context_enrich_agent")
workflow.add_edge("context_enrich_agent", "vibe_translate_agent") # Emoji 건너뜀
workflow.add_edge("vibe_translate_agent", END)

# --- 5. 그래프 컴파일 ---
app = workflow.compile()
log.info("✅ LangGraph 컴파일 완료 (Vibe: Local Model + VectorDB)")
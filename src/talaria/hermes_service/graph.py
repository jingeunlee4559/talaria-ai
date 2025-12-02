# src/talaria/hermes_service/graph.py
from langgraph.graph import StateGraph, END
from .state import TranslateState
from .nodes.node_check_conditions import check_conditions_node as analyze_business_context
from .nodes.node_context_translate import context_translate_node as correction_agent
# ⭐️ 분리된 함수들 임포트
from .nodes.node_vibe_extract import vibe_extract_kor_agent, vibe_extract_foreign_agent
from .nodes.node_vibe_enrich import context_enrich_agent
from .nodes.node_vibe_translate import vibe_translate_kor_agent, vibe_translate_foreign_agent
from .nodes.node_manner_guard import manner_guard_agent

def main_router(state: TranslateState) -> str:
    if state.get("translation_mode") == "business": return "business"
    # User Locale로 분기
    return "vibe_kor" if state.get("stt_lang", "en").startswith("ko") else "vibe_foreign"

def business_router(state):
    return "correct" if state.get("should_correct_business") else "fallback"

def fallback_node(state):
    state['final_answer'] = state['translated_text']
    return state

workflow = StateGraph(TranslateState)

# Nodes
workflow.add_node("analyze_business_context", analyze_business_context)
workflow.add_node("correction_agent", correction_agent)
workflow.add_node("standard_fallback_agent", fallback_node)

workflow.add_node("vibe_extract_kor", vibe_extract_kor_agent)
workflow.add_node("context_enrich", context_enrich_agent)
workflow.add_node("vibe_translate_kor", vibe_translate_kor_agent)

workflow.add_node("vibe_extract_foreign", vibe_extract_foreign_agent)
workflow.add_node("vibe_translate_foreign", vibe_translate_foreign_agent)

workflow.add_node("manner_guard", manner_guard_agent)

# Entry & Edges
workflow.set_conditional_entry_point(
    main_router,
    {
        "business": "analyze_business_context",
        "vibe_kor": "vibe_extract_kor",
        "vibe_foreign": "vibe_extract_foreign"
    }
)

# Business Flow
workflow.add_conditional_edges("analyze_business_context", business_router, {"correct": "correction_agent", "fallback": "standard_fallback_agent"})
workflow.add_edge("correction_agent", END)
workflow.add_edge("standard_fallback_agent", END)

# Vibe Flow (Korean)
workflow.add_edge("vibe_extract_kor", "context_enrich")
workflow.add_edge("context_enrich", "vibe_translate_kor")
workflow.add_edge("vibe_translate_kor", "manner_guard")

# Vibe Flow (Foreign) - Enrich Skip
workflow.add_edge("vibe_extract_foreign", "vibe_translate_foreign")
workflow.add_edge("vibe_translate_foreign", "manner_guard")

# Final
workflow.add_edge("manner_guard", END)

app = workflow.compile()
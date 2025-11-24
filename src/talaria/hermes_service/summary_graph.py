# src/talaria/hermes_service/summary_graph.py
from langgraph.graph import StateGraph, START, END
from .state import SummaryState
from .nodes import node_summary as nodes

MAX_RETRIES = 3

# --- 조건부 라우팅 로직 ---
def route_track_a(state: SummaryState):
    if state.get("track_a_feedback") and state.get("track_a_retry", 0) < MAX_RETRIES:
        return "retry"
    return "pass"

def route_track_b(state: SummaryState):
    if state.get("track_b_feedback") and state.get("track_b_retry", 0) < MAX_RETRIES:
        return "retry"
    return "pass"

# --- 그래프 빌드 ---
workflow = StateGraph(SummaryState)

# 1. Nodes 등록
# Track A
workflow.add_node("gen_summary", nodes.node_track_a_generate)
workflow.add_node("eval_a_struct", nodes.node_track_a_eval_structure)
workflow.add_node("eval_a_quality", nodes.node_track_a_eval_quality)

# Track B
workflow.add_node("ext_action", nodes.node_track_b_extract)
workflow.add_node("eval_b_schema", nodes.node_track_b_eval_schema)
workflow.add_node("eval_b_logic", nodes.node_track_b_eval_logic)

# Track C (2-Way)
workflow.add_node("analyze_content", nodes.node_analyze_content)
workflow.add_node("analyze_process", nodes.node_analyze_process)

# Final
workflow.add_node("final_merger", nodes.node_final_merger)


# 2. Edges 연결 (흐름 정의)

# [Start] -> [Track A] & [Track B] (병렬 시작)
workflow.add_edge(START, "gen_summary")
workflow.add_edge(START, "ext_action")

# --- Track A Flow (평가 분리) ---
workflow.add_edge("gen_summary", "eval_a_struct")
workflow.add_conditional_edges(
    "eval_a_struct", 
    route_track_a, 
    {"retry": "gen_summary", "pass": "eval_a_quality"}
)
workflow.add_conditional_edges(
    "eval_a_quality",
    route_track_a,
    {"retry": "gen_summary", "pass": "analyze_content"} # A 완료 -> C 시작
)

# --- Track B Flow (평가 분리) ---
workflow.add_edge("ext_action", "eval_b_schema")
workflow.add_conditional_edges(
    "eval_b_schema", 
    route_track_b, 
    {"retry": "ext_action", "pass": "eval_b_logic"}
)
workflow.add_conditional_edges(
    "eval_b_logic",
    route_track_b,
    {"retry": "ext_action", "pass": "analyze_content"} # B 완료 -> C 시작
)
# 주의: LangGraph에서 병렬 노드가 모이는 지점(analyze_content)은 
# 모든 상위 노드가 완료될 때까지 기다립니다.

# --- Track C Flow (2-Way Fan-out) ---
# analyze_content는 위에서 연결됨.
# analyze_process는 A, B와 상관없이 독립적으로 돌 수도 있지만, 
# 구조상 A/B 완료 후 함께 실행되도록 설정 (Context Merging 시점)
workflow.add_edge("eval_a_quality", "analyze_process")
workflow.add_edge("eval_b_logic", "analyze_process")

# --- Final Merge ---
workflow.add_edge("analyze_content", "final_merger")
workflow.add_edge("analyze_process", "final_merger")

workflow.add_edge("final_merger", END)

app = workflow.compile()
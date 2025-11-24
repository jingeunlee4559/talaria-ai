from langgraph.graph import StateGraph, START, END
from .state import SummaryState
from .nodes import node_summary as nodes

MAX_RETRIES = 3

def route_summary(state: SummaryState):
    if state.get("summary_feedback") and state.get("summary_retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "pass"

def route_extraction(state: SummaryState):
    if state.get("extraction_feedback") and state.get("extraction_retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "pass"

workflow = StateGraph(SummaryState)

# Nodes
workflow.add_node("generate_summary", nodes.node_generate_summary)
workflow.add_node("fact_check", nodes.node_fact_check)
workflow.add_node("completeness_check", nodes.node_completeness_check)

workflow.add_node("extract_json", nodes.node_extract_json)
workflow.add_node("validate_syntax", nodes.node_validate_syntax)
workflow.add_node("consistency_check", nodes.node_consistency_check)

workflow.add_node("final_merger", nodes.node_final_merger)

# Edges - Track A (Summary)
workflow.add_edge(START, "generate_summary")
workflow.add_edge("generate_summary", "fact_check")
workflow.add_conditional_edges("fact_check", route_summary, {"retry": "generate_summary", "pass": "completeness_check"})
workflow.add_conditional_edges("completeness_check", route_summary, {"retry": "generate_summary", "pass": "final_merger"})

# Edges - Track B (Extraction)
workflow.add_edge(START, "extract_json")
workflow.add_edge("extract_json", "validate_syntax")
workflow.add_conditional_edges("validate_syntax", route_extraction, {"retry": "extract_json", "pass": "consistency_check"})
workflow.add_conditional_edges("consistency_check", route_extraction, {"retry": "extract_json", "pass": "final_merger"})

# End
workflow.add_edge("final_merger", END)

app = workflow.compile()
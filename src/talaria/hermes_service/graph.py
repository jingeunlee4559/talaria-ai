# LangGraph 워크플로우 (엣지, 노드 연결)

from langgraph.graph import StateGraph, END
from .state import TranslateState
from .nodes.node_check_conditions import check_conditions_node
from .nodes.node_context_translate import context_translate_node

# --- 노드 함수 정의 ---

def fallback_translate_node(state: TranslateState) -> TranslateState:
    """
    (ELSE) 'should_context_translate'가 False일 때 실행되는 노드.
    단순히 파파고 원본 번역을 'answer'로 복사합니다.
    """
    state['answer'] = state['translated_text']
    return state

# --- 라우터 함수 정의 ---

def route_translation(state: TranslateState) -> str:
    """
    'should_context_translate' 플래그를 기반으로 
    'context_translate' 노드로 갈지 'fallback_translate' 노드로 갈지 결정합니다.
    """
    if state["should_context_translate"]:
        return "context_translate"  # IF 노드로 가는 엣지 이름
    else:
        return "fallback_translate" # ELSE 노드로 가는 엣지 이름

# --- 그래프 조립 ---

# 1. 그래프 빌더 생성
workflow = StateGraph(TranslateState)

# 2. 노드 추가
workflow.add_node("check_conditions", check_conditions_node)       # 1. 조건 검사 노드
workflow.add_node("context_translate", context_translate_node)    # 2. (IF) LLM 보정 노드
workflow.add_node("fallback_translate", fallback_translate_node)  # 3. (ELSE) 폴백 노드

# 3. 엣지(흐름) 설정
# 시작점 설정
workflow.set_entry_point("check_conditions")

# 조건부 분기 설정
workflow.add_conditional_edges(
    "check_conditions", # 'check_conditions' 노드가 끝나면
    route_translation,  # 'route_translation' 함수를 실행해서
    {
        # 그 결과가 "context_translate"이면 'context_translate' 노드로 이동
        "context_translate": "context_translate",
        # 그 결과가 "fallback_translate"이면 'fallback_translate' 노드로 이동
        "fallback_translate": "fallback_translate"
    }
)

# 4. 종료점 설정
workflow.add_edge("context_translate", END)  # LLM 보정 후 종료
workflow.add_edge("fallback_translate", END) # 폴백 후 종료

# 5. 그래프 컴파일
# 이제 'app'을 FastAPI에서 호출할 수 있습니다.
app = workflow.compile()
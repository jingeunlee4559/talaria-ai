import logging
from ..state import TranslateState, FoundTerm
from ..utils.model_loader import get_retriever, get_meme_dictionary # ⭐️ 족보 로더 추가

log = logging.getLogger(__name__)

def context_enrich_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-2] (Hybrid RAG)
    1. 족보(JSON) 매칭 (Exact Match)
    2. 모델 추출 기반 VectorDB 검색
    3. 문장 기반 VectorDB 검색 (Fallback)
    """
    retriever = get_retriever()
    meme_dict = get_meme_dictionary() # ⭐️ 족보 가져오기
    
    stt_text = state.get('stt_text', "")
    model_triggers = state.get('vibe_triggers', []) # 모델이 찾은 것들
    
    found_results = [] # 최종 찾은 밈들 (중복 제거용 dict)
    
    # --- [Step 1] 족보(Dictionary) 전수 조사 (가장 빠르고 정확) ---
    # 문장에 족보 키워드가 포함되어 있는지 확인
    # (최적화: 긴 단어부터 찾아서 짧은 단어 중복 매칭 방지 가능하지만, 여기선 단순 포함 확인)
    if meme_dict:
        for keyword, data in meme_dict.items():
            if keyword in stt_text:
                log.info(f"🥇 [Step 1] 족보 발견: '{keyword}'")
                found_results.append({
                    "term": data['primary_keyword'],
                    "definition": data['definition'],
                    "source": "Dictionary (Exact)"
                })
    
    # --- [Step 2] 모델 추출 기반 VectorDB 검색 (족보에 없을 때) ---
    # 족보에서 못 찾았는데, 모델이 뭔가 찾았다고 주장하는 경우
    if not found_results and model_triggers:
        log.info(f"🥈 [Step 2] 모델 트리거로 DB 검색 시도: {model_triggers}")
        search_res = retriever.search(model_triggers, top_k=1)
        
        for res in search_res:
            # 유사도가 너무 낮으면 retriever 내부에서 걸러짐 (거리 < 1.1)
            log.info(f"   -> DB 매칭 성공: {res['primary_keyword']}")
            found_results.append({
                "term": res['primary_keyword'],
                "definition": res['definition'],
                "source": "Model + VectorDB"
            })

    # --- [Step 3] 문장 전체 VectorDB 검색 (아무것도 못 찾았을 때) ---
    if not found_results and len(stt_text.split()) < 15:
        log.info(f"🥉 [Step 3] 문장 전체로 DB 검색 시도 (Fallback)")
        search_res = retriever.search([stt_text], top_k=1)
        
        for res in search_res:
            # 문장 검색은 더 엄격해야 함 (예: 거리 < 0.8) - retriever 내부 로직 의존 or 여기서 필터링
            # 여기서는 retriever가 이미 1.1 미만만 주므로 일단 채택
            log.info(f"   -> 문장 매칭 성공: {res['primary_keyword']}")
            found_results.append({
                "term": res['primary_keyword'],
                "definition": res['definition'],
                "source": "Sentence VectorDB"
            })

    # --- [결과 정리 및 State 업데이트] ---
    
    # 중복 제거 (여러 단계에서 같은 밈을 찾았을 수 있음)
    unique_terms = {}
    for item in found_results:
        unique_terms[item['term']] = item
    
    final_terms_list = list(unique_terms.values())
    
    # 데이터 매핑
    found_terms_struct = [] # FoundTerm 형식
    context_lines = []
    explanation_lines = []
    
    for item in final_terms_list:
        term = item['term']
        definition = item['definition']
        
        found_terms_struct.append({"term": term, "meaning": definition})
        context_lines.append(f"'{term}' means '{definition}'")
        explanation_lines.append(f"{term}: {definition}")

    state['found_terms'] = found_terms_struct
    
    if context_lines:
        state['vibe_context'] = "Context Info:\n" + "\n".join(context_lines)
        state['final_explanation'] = " | ".join(explanation_lines)
        
        # ⭐️ [중요] Publish용 데이터 (LiveKit Agent가 사용할 필드에 맞게)
        # 기존 state.py에 manner_check_data가 없다면 found_terms를 쓰거나
        # 임시 필드를 써야 합니다. (아까 논의한 대로 found_terms 활용 추천)
        # state['manner_check_data'] = found_terms_struct 
    else:
        state['vibe_context'] = ""
        state['final_explanation'] = ""

    return state

# (emoji_agent 등은 그대로)
def emoji_agent(state: TranslateState) -> TranslateState:
    return state
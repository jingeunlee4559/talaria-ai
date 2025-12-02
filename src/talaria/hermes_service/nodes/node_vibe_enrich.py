import logging
from ..state import TranslateState, FoundTerm
from ..utils.model_loader import get_retriever, get_meme_dictionary

log = logging.getLogger(__name__)

# [설정] 유사도 임계값 (L2 Distance 기준, 낮을수록 엄격)
# 0.0: 완전 일치, ~0.3: 매우 유사, ~0.5: 유사함, 1.0~: 관련 없음
THRESHOLD_WORD_SEARCH = 0.45      # Step 2-2: 단어 기반 검색 허용 범위
THRESHOLD_SENTENCE_SEARCH = 0.35  # Step 2-3: 문장 기반 검색 (노이즈가 심하므로 더 엄격하게)

def context_enrich_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-2] (Hybrid RAG with Strict Filtering)
    1. 족보(JSON) 매칭 (Exact Match) - 신뢰도 100%
    2. 모델 추출 기반 VectorDB 검색 - 신뢰도 High (Threshold 적용)
    3. 문장 기반 VectorDB 검색 (Fallback) - 신뢰도 Low (Very Strict Threshold 적용)
    """
    retriever = get_retriever()
    meme_dict = get_meme_dictionary()
    
    stt_text = state.get('stt_text', "")
    model_triggers = state.get('vibe_triggers', []) # 모델(Step 1)이 찾은 밈 후보군
    
    found_results = [] # 최종 선별된 밈 리스트
    matched_keys = set() # 중복 방지용 키 집합
    
    # --- [Step 1] 족보(Dictionary) 전수 조사 (Exact Match) ---
    # 신뢰도가 가장 높으므로 무조건 포함
    if meme_dict:
        for keyword, data in meme_dict.items():
            # 1. 문장에 키워드가 포함되어 있는지 확인
            if keyword in stt_text:
                # 이미 찾은 키워드가 아니라면 추가
                if keyword not in matched_keys:
                    log.info(f"🥇 [Step 1] 족보 발견(Exact): '{keyword}'")
                    found_results.append({
                        "term": data['primary_keyword'],
                        "definition": data['definition'],
                        "source": "Dictionary (Exact)",
                        "score": 0.0 # 거리가 0 (완벽 일치)
                    })
                    matched_keys.add(keyword)
    
    # --- [Step 2] 모델 추출 기반 VectorDB 검색 ---
    # 모델이 밈이라고 '지목'한 단어가 있는데, 족보에는 없는 경우 DB에서 찾음
    if model_triggers:
        log.info(f"🥈 [Step 2] 모델 트리거로 DB 검색 시도: {model_triggers}")
        
        # 모델이 찾은 단어 각각에 대해 검색
        search_res = retriever.search(model_triggers, top_k=1)
        
        for res in search_res:
            # 거리(score)가 threshold보다 작은지 확인 (작을수록 유사)
            score = res.get('score', 10.0) # score가 없으면 아주 큰 값으로 가정
            keyword = res['primary_keyword']

            if score <= THRESHOLD_WORD_SEARCH:
                if keyword not in matched_keys:
                    log.info(f"   -> ✅ DB 매칭 성공 (Dist: {score:.3f}): {keyword}")
                    found_results.append({
                        "term": keyword,
                        "definition": res['definition'],
                        "source": "Model + VectorDB",
                        "score": score
                    })
                    matched_keys.add(keyword)
            else:
                log.info(f"   -> ❌ DB 매칭 실패 (Dist: {score:.3f} > {THRESHOLD_WORD_SEARCH}): {keyword} - 유사도 부족")

    # --- [Step 3] 문장 전체 VectorDB 검색 (Fallback) ---
    # 조건: 
    # 1. Step 1, 2에서 아무것도 못 찾았을 때만 시도 (False Positive 방지)
    # 2. 문장이 너무 길지 않을 때 (긴 문장은 전체 유사도 정확성이 떨어짐)
    # 3. 모델(Step 1)이 밈 감정을 'NEUTRAL'이 아닌 것으로 잡았을 때 (선택사항, 일단은 제외)
    
    if not found_results and len(stt_text.split()) < 15:
        log.info(f"🥉 [Step 3] 문장 전체로 DB 검색 시도 (Fallback)")
        search_res = retriever.search([stt_text], top_k=1)
        
        for res in search_res:
            score = res.get('score', 10.0)
            keyword = res['primary_keyword']
            
            # 문장 검색은 엉뚱한 밈을 가져올 확률이 매우 높으므로 더 엄격한 기준 적용
            if score <= THRESHOLD_SENTENCE_SEARCH:
                if keyword not in matched_keys:
                    log.info(f"   -> ✅ 문장 매칭 성공 (Dist: {score:.3f}): {keyword}")
                    found_results.append({
                        "term": keyword,
                        "definition": res['definition'],
                        "source": "Sentence VectorDB",
                        "score": score
                    })
                    matched_keys.add(keyword)
            else:
                log.info(f"   -> ❌ 문장 매칭 실패 (Dist: {score:.3f} > {THRESHOLD_SENTENCE_SEARCH}): {keyword} - 너무 다름")

    # --- [결과 정리 및 State 업데이트] ---
    
    # FoundTerm 구조체로 변환
    found_terms_struct = [] 
    context_lines = []
    explanation_lines = []
    
    for item in found_results:
        term = item['term']
        definition = item['definition']
        
        found_terms_struct.append({"term": term, "meaning": definition})
        context_lines.append(f"'{term}' means '{definition}'")
        explanation_lines.append(f"{term}: {definition}")

    # 최종적으로 찾은 게 없으면 빈 리스트가 들어감 -> manner_alert 안 뜸
    state['found_terms'] = found_terms_struct
    
    if context_lines:
        state['vibe_context'] = "Context Info:\n" + "\n".join(context_lines)
        state['final_explanation'] = " | ".join(explanation_lines)
    else:
        state['vibe_context'] = "" # LLM에게 줄 힌트 없음
        state['final_explanation'] = "" # 사용자에게 보여줄 설명 없음

    return state
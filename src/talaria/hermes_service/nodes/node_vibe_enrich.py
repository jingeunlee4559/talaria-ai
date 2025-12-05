# src/talaria/hermes_service/nodes/node_vibe_enrich.py
import logging
import re
from ..state import TranslateState, FoundTerm
from ..utils.model_loader import get_retriever, get_meme_dictionary

log = logging.getLogger(__name__)

THRESHOLD_WORD_SEARCH = 0.45

def context_enrich_agent(state: TranslateState) -> TranslateState:
    stt_lang = state.get('stt_lang', 'ko')
    if not stt_lang.startswith('ko'):
        state['found_terms'] = []
        state['vibe_context'] = ""
        state['final_explanation'] = None
        return state

    retriever = get_retriever()
    meme_dict = get_meme_dictionary()
    stt_text = state.get('stt_text', "")
    model_triggers = state.get('vibe_triggers', []) 
    
    all_results = [] 
    matched_keys = set() 
    
    # [Step 1] 족보 검색 (Exact Match with Boundary Check)
    if meme_dict:
        for keyword, data in meme_dict.items():
            # ⭐️ [수정] 검색 키워드(keyword)가 대표어(primary_keyword)와 다르면 건너뜀
            # 즉, related_keywords(동의어)는 검색 대상에서 제외됨
            if keyword != data.get('primary_keyword'):
                continue

            if len(keyword) >= 2: 
                # 정규식: 키워드 바로 앞에 '한글'이 오면 매칭 안 함 (Boundary Check)
                pattern = rf"(?<![가-힣]){re.escape(keyword)}"
                
                if re.search(pattern, stt_text) and keyword not in matched_keys:
                    all_results.append({
                        "term": data['primary_keyword'],
                        "definition": data['definition'],
                        "category": data.get('category', 'k-meme'),
                        "source": "Dictionary (Exact)",
                        "is_publishable": True 
                    })
                    matched_keys.add(keyword)
    
    # [Step 2] 모델 기반 DB 검색 (VectorDB)
    if model_triggers:
        search_res = retriever.search(model_triggers, top_k=1)
        for res in search_res:
            score = res.get('score', 10.0)
            keyword = res['primary_keyword'] # 모델 검색은 이미 primary_keyword 기준
            
            if score <= THRESHOLD_WORD_SEARCH and keyword not in matched_keys:
                pattern = rf"(?<![가-힣]){re.escape(keyword)}"
                
                if re.search(pattern, stt_text):
                    all_results.append({
                        "term": keyword,
                        "definition": res['definition'],
                        "category": res.get('category', 'k-meme'),
                        "source": "Model + VectorDB",
                        "is_publishable": True
                    })
                    matched_keys.add(keyword)
                else:
                    log.info(f"🚫 [Reject] 문맥 불일치(어미/포함): '{keyword}'")

    # [결과 정리]
    found_terms_struct = []
    context_lines = []
    explanation_lines = []
    
    for item in all_results:
        context_lines.append(f"'{item['term']}' means '{item['definition']}'")
        
        if item['is_publishable']:
            found_terms_struct.append({
                "term": item['term'], 
                "meaning": item['definition'],
                "category": item['category']
            })
            explanation_lines.append(f"{item['term']}: {item['definition']}")

    state['found_terms'] = found_terms_struct
    state['vibe_context'] = "Context Info:\n" + "\n".join(context_lines) if context_lines else ""
    state['final_explanation'] = " | ".join(explanation_lines) if explanation_lines else None

    return state
# src/talaria/hermes_service/nodes/node_vibe_enrich.py
import logging
import json
import os
from ..state import TranslateState
from typing import Dict # ⭐️ Dict 임포트

log = logging.getLogger(__name__)

# --- Rulebook 로드 (동일) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "..", "..", "assets")

def load_json_asset(filename: str) -> Dict:
    filepath = os.path.join(ASSETS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.error(f"❌ {filename} 로드 실패: {e}", exc_info=True)
        return {}

KCULTURE_RULEBOOK = load_json_asset("kculture_rulebook.json")
EMOJI_MAP = load_json_asset("emotion_emoji_map.json")
# --- 로드 완료 ---


# ⭐️ [수정] context_enrich_agent: 힌트 생성 로직 추가
def context_enrich_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-2] (규칙 기반)
    1. UI용 'final_explanation' (긴 설명) 생성
    2. LLM 2(번역)용 'vibe_context' (짧은 힌트) 생성
    """
    triggers = state.get("vibe_triggers", [])
    if not triggers:
        log.info("Vibe 힌트: 인식된 용어 없음.")
        state['vibe_context'] = "No specific K-Culture terms identified."
        return state
        
    explanations_for_ui = [] # (UI용)
    hints_for_llm = []     # (LLM 2용)
    
    for term in triggers:
        if term in KCULTURE_RULEBOOK:
            explanation = KCULTURE_RULEBOOK[term]
            # 1. UI용: "용어: 설명" (긴 형식)
            explanations_for_ui.append(f"'{term}': {explanation}")
            
            # 2. LLM 2용: "용어 means 설명" (짧은 힌트)
            # (설명이 너무 길면 LLM이 혼란스러워하므로, 'reliable'/'praise' 등 핵심 의미만 추출)
            # (임시) 여기서는 설명을 그대로 사용 (추후 'hint' 필드를 DB에 추가하면 더 좋음)
            hint = explanation.split(".")[0] # 첫 번째 문장만 사용
            hints_for_llm.append(f"'{term}' means '{hint}'")

    if explanations_for_ui:
        state['final_explanation'] = " | ".join(explanations_for_ui)
        state['vibe_context'] = "Context: " + ". ".join(hints_for_llm)
        log.info(f"Vibe 힌트(LLM 2용) 생성: {state['vibe_context']}")
    else:
        state['vibe_context'] = "No specific K-Culture terms identified."

    return state

# ⭐️ [수정 없음] emoji_agent는 수정할 필요 없음
def emoji_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-3] (규칙 기반)
    LLM 1이 분석한 'vibe_emotion_tag'를 'emotion_emoji_map.json'과 매핑합니다.
    """
    emotion = state.get("vibe_emotion_tag", "default")
    emoji = EMOJI_MAP.get(emotion, EMOJI_MAP.get("default", "💬"))
    
    state['final_emoji'] = emoji
    log.info(f"Vibe 이모지 매핑: {emotion} -> {emoji}")
    
    return state
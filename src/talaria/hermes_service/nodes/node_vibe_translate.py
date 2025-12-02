# src/talaria/hermes_service/nodes/node_vibe_translate.py
import logging
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..state import TranslateState

log = logging.getLogger(__name__)

try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.environ["OPENAI_API_KEY"],  # 권장: 환경변수 명시
    )
except:
    llm = None

EFFECT_TAG_MAP = {
    "PARTY": "🎉", "FIRE": "🔥", "HEART": "💖", "BOOM": "💥", 
    "COLD": "❄️", "MOON": "🌙", "MAGIC": "✨"
}

# [Prompt 1] 한국어용 (Context 사용)
KOR_PROMPT = """You are a Gen-Z translator. 
Translate **Korean** text into **authentic, vibe-heavy {target_lang} slang**.

### Critical Rules:
1. **Localize Metaphors:** DO NOT translate Korean idioms literally (e.g., 'Goguma', 'Cider'). Instead, find the **perfect slang equivalent** in {target_lang}.
   - (Bad): "I ate sweet potatoes."
   - (Good): "It was so suffocating / frustrating."
2. **Vibe First:** Focus on the emotion and impact, not the exact words.

### Context (Meme Meanings)
{vibe_context}

### Tone
{vibe_emotion}

### Effect Tags (Append ONE if fits):
<EFFECT:PARTY>, <EFFECT:FIRE>, <EFFECT:HEART>, <EFFECT:BOOM>, <EFFECT:COLD>, <EFFECT:MOON>, <EFFECT:MAGIC>

Input: "{corrected_stt}"
Translation:"""

# [Prompt 2] 외국어용 (Context 없음)
FOREIGN_PROMPT = """You are a Gen-Z translator. 
Translate **{source_lang}** text into **authentic, vibe-heavy {target_lang} slang**.

### Critical Rules:
1. **Localize Metaphors:** Use {target_lang} slang that carries the *same feeling*, not the literal meaning.
2. **Vibe First:** Make it sound like a native speaker of {target_lang}.

### Tone
{vibe_emotion}

### Effect Tags (Append ONE if fits):
<EFFECT:PARTY>, <EFFECT:FIRE>, <EFFECT:HEART>, <EFFECT:BOOM>, <EFFECT:COLD>, <EFFECT:MOON>, <EFFECT:MAGIC>

Input: "{corrected_stt}"
Translation:"""

chain_kor = ChatPromptTemplate.from_template(KOR_PROMPT) | llm | StrOutputParser()
chain_foreign = ChatPromptTemplate.from_template(FOREIGN_PROMPT) | llm | StrOutputParser()

EMOJI_THRESHOLD = 0.8

def _parse_effect(text: str, emotion: str, score: float):
    """
    텍스트 끝의 <EFFECT:XXX> 태그를 파싱하되,
    점수가 낮거나 감정이 NEUTRAL이면 태그를 제거하고 이모지를 붙이지 않습니다.
    """
    # 1. 태그 추출
    match = re.search(r"<EFFECT:([A-Z]+)>", text)
    clean_text = re.sub(r"\s*<EFFECT:[A-Z]+>", "", text).strip() # 일단 태그는 본문에서 지움
    
    # 2. ⭐️ [핵심] 필터링 로직
    # 감정이 NEUTRAL이거나, 점수가 기준치 미만이면 이모지 생성 안 함
    if emotion == "NEUTRAL" or score < EMOJI_THRESHOLD:
        if match:
            log.info(f"🔇 이펙트 무시됨 (Score {score:.2f} < {EMOJI_THRESHOLD} or Neutral)")
        return clean_text, ""

    # 3. 기준 통과 시 이모지 변환
    if match:
        tag_key = match.group(1)
        emoji = EFFECT_TAG_MAP.get(tag_key, "")
        if emoji:
            log.info(f"✨ 이펙트 적용! ({tag_key} -> {emoji}, Score: {score:.2f})")
            return clean_text, f" {emoji}"
            
    return clean_text, ""


# [Node 1] 한국어 번역
def vibe_translate_kor_agent(state: TranslateState) -> TranslateState:
    if not llm: return state
    
    # State에서 점수 가져오기
    emotion = state.get("vibe_emotion_tag", "NEUTRAL")
    score = state.get("vibe_emotion_score", 0.0) 
    
    try:
        res = chain_kor.invoke({
            # ... (기존 인자) ...
            "target_lang": state.get("target_lang", "English"),
            "vibe_context": state.get("vibe_context", ""),
            "vibe_emotion": emotion,
            "corrected_stt": state.get("corrected_stt", "")
        })
        
        raw_res = res.strip().strip('"')
        
        # ⭐️ 파싱 함수에 점수 전달
        clean, emoji = _parse_effect(raw_res, emotion, score)
        
        state['final_answer'] = f"{clean}{emoji}"
        if emoji: state['corrected_stt'] += emoji # 원문에도 이모지 부착
        
    except Exception as e:
        log.error(f"Trans-KO Error: {e}")
        state['final_answer'] = state.get("corrected_stt", "")
        
    return state

# [Node 2] 외국어 번역
def vibe_translate_foreign_agent(state: TranslateState) -> TranslateState:
    if not llm: return state
    
    emotion = state.get("vibe_emotion_tag", "NEUTRAL")
    score = state.get("vibe_emotion_score", 0.0)

    try:
        res = chain_foreign.invoke({
            # ... (기존 인자) ...
            "source_lang": state.get("stt_lang", "Foreign"),
            "target_lang": state.get("target_lang", "English"),
            "vibe_emotion": emotion,
            "corrected_stt": state.get("corrected_stt", "")
        })
        
        raw_res = res.strip().strip('"')
        
        # ⭐️ 파싱 함수에 점수 전달
        clean, emoji = _parse_effect(raw_res, emotion, score)
        
        state['final_answer'] = f"{clean}{emoji}"
        if emoji: state['corrected_stt'] += emoji
        
    except Exception as e:
        log.error(f"Trans-Foreign Error: {e}")
        state['final_answer'] = state.get("corrected_stt", "")
        
    return state
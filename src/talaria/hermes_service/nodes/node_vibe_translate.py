# src/talaria/hermes_service/nodes/node_vibe_translate.py
import logging
import random
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..state import TranslateState

log = logging.getLogger(__name__)

# (LLM 초기화 - 기존 동일)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_retries=2) 
    log.info("✅ Vibe용 LLM(Translate) (gpt-4o-mini) 로드 완료.")
except Exception as e:
    llm = None
    log.critical(f"❌ Vibe용 LLM(Translate) 로드 실패: {e}", exc_info=True)

# (프롬프트 - 기존 동일)
VIBE_TRANSLATE_PROMPT = """You are a witty, trendy Gen-Z translator.
Your goal is to translate Korean text into **authentic, vibe-heavy {target_lang} slang**.
Do NOT sound like a textbook or a robot. Sound like a cool friend texting in a group chat.

### 1. Cultural Context (Meme Hints)
Use the following meanings to capture the *intent* of the slang:
{vibe_context}

### 2. Emotional Tone (Match this Vibe)
The detected vibe is: **{vibe_emotion}**
Apply the specific style rules below:
- **SARCASM:** Dry humor, lowercase, '...' pauses. (e.g., "wow. just wow.")
- **ADMIRATION:** Hype it up! Caps, "GOAT", "Slay". (e.g., "BRO THIS IS IT")
- **SHOCK:** Disbelief. "?!", "No way". (e.g., "Wait what??")
- **NEUTRAL:** Chill, casual. (e.g., "just vibing")

### 3. Constraints
- **Preserve Proper Nouns:** Do NOT translate names.
- **No Explanations:** Just give the translated text.
- **Slang Usage:** Use internet abbreviations (idk, rn, tbh) where appropriate.

---
**Input Text:**
"{corrected_stt}"

**Translation ({target_lang}):**
"""

prompt = ChatPromptTemplate.from_template(VIBE_TRANSLATE_PROMPT)
chain = prompt | llm | StrOutputParser()

# (이모지 맵 - 기존 동일)
EMOJI_EFFECT_MAP = {
    "ADMIRATION": ["🔥", "✨", "💖", "🎉", "👑", "💎"],
    "SHOCK": ["💥", "⚡", "💫"],
    "SARCASM": ["❄️", "🌀", "🌙"],
    "NEUTRAL": [] 
}

# ⭐️ [수정] 텍스트를 반환하는 게 아니라, '선정된 이모지 문자열'만 반환하도록 변경
def select_emoji_suffix(emotion: str) -> str:
    """
    감정에 맞는 이모지를 확률적으로 선택하여 반환합니다. (없으면 빈 문자열)
    """
    candidates = EMOJI_EFFECT_MAP.get(emotion, [])
    if not candidates:
        return ""

    # 빈도 조절 (30% 확률)
    if random.random() > 0.3: 
        return ""

    # 이모지 하나 선택
    return f" {random.choice(candidates)}"  # 앞에 공백 포함해서 리턴


def vibe_translate_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-4] (LLM 2: 번역 + Python 후처리)
    """
    if not llm:
        log.error("Vibe LLM 로드 실패")
        state['final_answer'] = state['stt_text']
        return state
    
    corrected_stt = state.get("corrected_stt") or state.get("stt_text", "")
    vibe_context = state.get("vibe_context") or "No specific meme definitions found."
    target_lang = state.get("target_lang") or "English"
    vibe_emotion = state.get("vibe_emotion_tag") or "NEUTRAL"

    if not corrected_stt.strip():
        state['final_answer'] = ""
        return state

    try:
        # 1. LLM 번역
        response = chain.invoke({
            "corrected_stt": corrected_stt,
            "vibe_context": vibe_context,
            "vibe_emotion": vibe_emotion,
            "target_lang": target_lang
        })
        
        translated_text = response.strip().strip('"')
        
        # 2. ⭐️ [수정] 이모지 선정 (한 번만 수행하여 싱크 맞춤)
        emoji_suffix = select_emoji_suffix(vibe_emotion)
        
        # 3. ⭐️ [수정] 번역문과 원문 양쪽에 적용
        # (원문에 적용할 때는 state['corrected_stt']를 업데이트)
        
        state['final_answer'] = f"{translated_text}{emoji_suffix}"
        
        if emoji_suffix:
            # 원문에도 이모지가 붙도록 업데이트
            new_source_text = f"{corrected_stt}{emoji_suffix}"
            state['corrected_stt'] = new_source_text
            log.info(f"✨ 이모지 적용(Source & Target): {emoji_suffix.strip()}")
        else:
            log.info(f"Vibe 번역 완료 (No Emoji): '{translated_text}'")

    except Exception as e:
        log.error(f"Vibe LLM 수행 중 오류: {e}", exc_info=True)
        state['final_answer'] = corrected_stt
        
    return state
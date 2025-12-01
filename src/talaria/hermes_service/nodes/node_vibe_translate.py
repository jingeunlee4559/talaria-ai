# src/talaria/hermes_service/nodes/node_vibe_translate.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..state import TranslateState

log = logging.getLogger(__name__)

# (LLM 및 파서 초기화)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_retries=2) # 창의성을 위해 temperature를 살짝 높임 (0.1 -> 0.3)
    log.info("✅ Vibe용 LLM(Translate) (gpt-4o-mini) 로드 완료.")
except Exception as e:
    llm = None
    log.critical(f"❌ Vibe용 LLM(Translate) 로드 실패: {e}", exc_info=True)

# ⭐️ [수정] MZ 톤 & 감정 극대화 프롬프트
VIBE_TRANSLATE_PROMPT = """You are a witty, trendy Gen-Z translator.
Your goal is to translate Korean text into **authentic, vibe-heavy {target_lang} slang**.
Do NOT sound like a textbook or a robot. Sound like a cool friend texting in a group chat.

### 1. Cultural Context (Meme Hints)
Use the following meanings to capture the *intent* of the slang, not just the literal meaning:
{vibe_context}

### 2. Emotional Tone (⚠️ CRITICAL: Match this Vibe)
The detected vibe is: **{vibe_emotion}**
Apply the specific style rules below:

- **SARCASM:** Use dry humor, irony, or '...' pauses. Use lowercase to sound unbothered but biting. (e.g., "wow. just wow.", "suuure", "fr?", "big yikes")
- **ADMIRATION:** HYPE IT UP! Use caps for emphasis. Use slang like "GOAT", "Legend", "Slay", "Insane", "Lit". (e.g., "BRO THIS IS IT", "Literally the best")
- **SHOCK:** Express total disbelief. Use "?!", "No way", "Shook", "For real?". (e.g., "Wait what??", "Ain't no way")
- **NEUTRAL:** Keep it chill, casual, and concise. Texting style. (e.g., "just vibing", "kinda hungry", "wanna grab food?")

### 3. Constraints
- **Preserve Proper Nouns:** Do NOT translate names (people, places, games). If input is "Min-su", output "Min-su".
- **No Explanations:** Just give the translated text. Do not say "It means...".
- **Slang Usage:** Use internet abbreviations (idk, rn, tbh, imo) where appropriate for the {target_lang}.

---
**Input Text:**
"{corrected_stt}"

**Translation ({target_lang}):**
"""

prompt = ChatPromptTemplate.from_template(VIBE_TRANSLATE_PROMPT)
chain = prompt | llm | StrOutputParser()

def vibe_translate_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-4] (LLM 2: 번역)
    (1)보정된 STT, (2)DB에서 찾은 밈 뜻, (3)모델이 분석한 뉘앙스를 종합하여
    최종 Vibe 번역을 수행합니다.
    """
    if not llm:
        log.error("Vibe LLM(Translate)이 로드되지 않아 Vibe 모드를 건너뜁니다.")
        state['final_answer'] = f"[Vibe Mode Error] {state['stt_text']}"
        return state

    log.info(f"Vibe 번역 에이전트(LLM 2) 실행...")
    
    # (입력값 가져오기)
    corrected_stt = state.get("corrected_stt") or state.get("stt_text", "")
    vibe_context = state.get("vibe_context") or "No specific meme definitions found."
    target_lang = state.get("target_lang") or "English"
    
    # 자체 모델이 분석한 감정 태그 가져오기
    vibe_emotion = state.get("vibe_emotion_tag") or "NEUTRAL"

    if not corrected_stt.strip():
        log.warning("Vibe 번역: 입력 STT가 비어있어 건너뜁니다.")
        state['final_answer'] = ""
        return state

    try:
        # 감정(vibe_emotion) 변수 추가 전달
        response = chain.invoke({
            "corrected_stt": corrected_stt,
            "vibe_context": vibe_context,
            "vibe_emotion": vibe_emotion,
            "target_lang": target_lang
        })
        
        state['final_answer'] = response.strip().strip('"')
        log.info(f"Vibe 번역 완료: {state['final_answer']} (Tone: {vibe_emotion})")
        
    except Exception as e:
        log.error(f"Vibe LLM(Translate) 호출 중 오류: {e}", exc_info=True)
        state['final_answer'] = f"[Vibe Error] {corrected_stt}"
        
    return state
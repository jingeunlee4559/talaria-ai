# src/talaria/hermes_service/nodes/node_vibe_translate.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..state import TranslateState

log = logging.getLogger(__name__)

# (LLM 및 파서 초기화)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=2)
    log.info("✅ Vibe용 LLM(Translate) (gpt-4o-mini) 로드 완료.")
except Exception as e:
    llm = None
    log.critical(f"❌ Vibe용 LLM(Translate) 로드 실패: {e}", exc_info=True)

# ⭐️ [신규] "힌트 기반 번역" 전용 프롬프트
VIBE_TRANSLATE_PROMPT = """You are an expert translator specializing in K-Culture and Z-generation slang.
Your task is to translate the given Korean text into a natural, casual, Z-generation {target_lang} tone.

**Crucial Context:** You MUST use the following hints to understand the cultural nuances:
{vibe_context}
(e.g., If the context says '국밥' means 'reliable', you MUST translate it as 'reliable', NOT 'trash' or 'rice soup'.)

**Corrected STT (Input):**
"{corrected_stt}"

Respond ONLY with the final, vibe-translated {target_lang} text. Do not add explanations.
"""
prompt = ChatPromptTemplate.from_template(VIBE_TRANSLATE_PROMPT)
chain = prompt | llm | StrOutputParser()

def vibe_translate_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-4] (LLM 2: 번역)
    (1)보정된 STT와 (2)문화적 힌트를 기반으로 '최종 Vibe 번역'을 수행합니다.
    """
    if not llm:
        log.error("Vibe LLM(Translate)이 로드되지 않아 Vibe 모드를 건너뜁니다.")
        state['final_answer'] = f"[Vibe Mode Error] {state['stt_text']}"
        return state

    log.info(f"Vibe 번역 에이전트(LLM 2) 실행...")
    
    # (입력값)
    corrected_stt = state.get("corrected_stt") or state.get("stt_text", "")
    vibe_context = state.get("vibe_context") or "No context provided."
    target_lang = state.get("target_lang") or "English"

    if not corrected_stt.strip():
        log.warning("Vibe 번역: 입력 STT가 비어있어 건너뜁니다.")
        state['final_answer'] = ""
        return state

    try:
        response = chain.invoke({
            "corrected_stt": corrected_stt,
            "vibe_context": vibe_context,
            "target_lang": target_lang
        })
        
        state['final_answer'] = response.strip().strip('"')
        log.info(f"Vibe 번역(LLM 2) 완료: {state['final_answer']}")
        
    except Exception as e:
        log.error(f"Vibe LLM(Translate) 호출 중 오류: {e}", exc_info=True)
        state['final_answer'] = f"[Vibe Error] {corrected_stt}"
        
    return state
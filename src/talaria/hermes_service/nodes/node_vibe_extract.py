# src/talaria/hermes_service/nodes/node_vibe_extract.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from ..state import TranslateState

log = logging.getLogger(__name__)

try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=2)
    log.info("✅ Vibe용 LLM(Extract) (gpt-4o-mini) 로드 완료.")
except Exception as e:
    llm = None
    log.critical(f"❌ Vibe용 LLM(Extract) 로드 실패: {e}", exc_info=True)

VIBE_EXTRACT_PROMPT = """You are an expert in correcting STT (Speech-to-Text) errors based on context for multiple languages.
Your task is to correct a potentially erroneous STT sentence and classify its emotion, based on the provided language code.

Input Language: "{stt_lang}"
STT Input: "{stt_text}"
Context (Chat History):
{chat_history}

You MUST perform 2 tasks and respond ONLY with a JSON object:
1.  **Correct STT:** Correct STT typos based on the cultural context, internet slang, or common speech patterns for the given **Input Language**.
    * (Example for "ko-KR"): "베그 폼 미쳣다" -> "배그 폼 미쳤다", "킹밧네" -> "킹받네"
    * (Example for "en-US"): "That's bussin" -> "That's bussin'", "I'm finna go" -> "I'm gonna go", "He left me on read" -> "He left me on read"
2.  **Tag Emotion:** Classify the speaker's core emotion from this list: [joy, surprise, sadness, anger].
    If no strong emotion is detected, or the emotion is neutral, respond with "neutral".

Respond ONLY with the following JSON format:
{{
  "corrected_stt": "...",
  "emotion_tag": "..."
}}
"""
prompt = ChatPromptTemplate.from_template(VIBE_EXTRACT_PROMPT)
parser = JsonOutputParser()
chain = prompt | llm | parser

def vibe_extract_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-1] (LLM 1: STT 보정 + 감정)
    (1)STT 오류 보정, (2)감정분석을 수행합니다. (용어 추출 X)
    """
    if not llm:
        log.error("Vibe LLM(Extract)이 로드되지 않아 Vibe 모드를 건너뜁니다.")
        state['corrected_stt'] = state['stt_text'] # ⭐️ [수정] 원본 STT로 폴백
        state['vibe_emotion_tag'] = "neutral"
        return state

    log.info(f"Vibe 추출 에이전트(LLM 1) 실행: (STT: '{state['stt_text']}')")
    try:
        history_str = "\n".join(f"- {h}" for h in state.get("chat_history", []))
        
        response = chain.invoke({
            "stt_lang": state.get("stt_lang", "ko-KR"),
            "stt_text": state['stt_text'],
            "chat_history": history_str
        })
        
        # ⭐️ [수정] 'identified_terms' 대신 'corrected_stt'를 저장
        state['corrected_stt'] = response.get("corrected_stt", state['stt_text'])
        state['vibe_emotion_tag'] = response.get("emotion_tag", "neutral")
        
        log.info(f"Vibe 추출 완료: (Emotion: {state['vibe_emotion_tag']}, Corrected: {state['corrected_stt']})")
        
    except Exception as e:
        log.error(f"Vibe LLM(Extract) 호출 중 오류: {e}", exc_info=True)
        state['corrected_stt'] = state['stt_text'] 
        state['vibe_triggers'] = []
        state['vibe_emotion_tag'] = "neutral"
        
    return state
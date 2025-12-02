# src/talaria/hermes_service/nodes/node_vibe_extract.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser # JSON 파서 사용
from pydantic import BaseModel, Field
from ..state import TranslateState
from ..utils.model_loader import get_scanner

log = logging.getLogger(__name__)

# --- Foreign용 LLM 설정 ---
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
except:
    llm = None

# ⭐️ [수정] 외국어 감정 분석 스키마 (점수 포함)
class EmotionResult(BaseModel):
    emotion: str = Field(description="One of: ADMIRATION, SHOCK, SARCASM, NEUTRAL")
    score: int = Field(description="Intensity score (0-100). 0 for Neutral, 100 for Extreme.")

foreign_parser = JsonOutputParser(pydantic_object=EmotionResult)

EMOTION_PROMPT = """Analyze the emotion and intensity.
Return JSON:
{{
    "emotion": "ADMIRATION" | "SHOCK" | "SARCASM" | "NEUTRAL",
    "score": 0~100 (Integer)
}}

Text: "{text}"
"""
foreign_chain = ChatPromptTemplate.from_template(EMOTION_PROMPT) | llm | foreign_parser


# [Node 1] 한국어 전용 (Scanner)
def vibe_extract_kor_agent(state: TranslateState) -> TranslateState:
    stt_text = state['stt_text']
    scanner = get_scanner()
    
    if not scanner:
        state['vibe_triggers'] = []
        state['vibe_emotion_tag'] = "NEUTRAL"
        state['vibe_emotion_score'] = 0.0 # 기본값
        state['corrected_stt'] = stt_text
        return state

    try:
        # Scanner가 score(0.0~1.0)를 반환한다고 가정 (model_loader 확인됨)
        result = scanner.predict(current_text=stt_text, prev_text=state.get("chat_history", [])[-1] if state.get("chat_history") else "")
        
        state['vibe_triggers'] = result.get("meme_words", [])
        state['vibe_emotion_tag'] = result.get("nuance", "NEUTRAL")
        
        # ⭐️ [추가] 점수 저장 (이미 0.0~1.0 범위)
        state['vibe_emotion_score'] = result.get("score", 0.0) 
        
        state['corrected_stt'] = stt_text
        log.info(f"🇰🇷 [Scanner] 감정: {state['vibe_emotion_tag']} (Score: {state['vibe_emotion_score']:.2f})")
        
    except Exception as e:
        log.error(f"Scanner Error: {e}")
        state['vibe_emotion_score'] = 0.0
        
    return state


# [Node 2] 외국어 전용 (LLM)
def vibe_extract_foreign_agent(state: TranslateState) -> TranslateState:
    stt_text = state['stt_text']
    
    try:
        if llm:
            res = foreign_chain.invoke({"text": stt_text})
            emotion = res.get("emotion", "NEUTRAL").upper()
            score_val = res.get("score", 0)
            
            # 유효성 검사
            if emotion not in ["ADMIRATION", "SHOCK", "SARCASM", "NEUTRAL"]:
                emotion = "NEUTRAL"
                
            state['vibe_emotion_tag'] = emotion
            # ⭐️ [추가] 0~100 점수를 0.0~1.0으로 정규화
            state['vibe_emotion_score'] = float(score_val) / 100.0
            
        else:
            state['vibe_emotion_tag'] = "NEUTRAL"
            state['vibe_emotion_score'] = 0.0
        
        state['vibe_triggers'] = []
        state['corrected_stt'] = stt_text
        log.info(f"🌍 [Foreign] 감정: {state['vibe_emotion_tag']} (Score: {state['vibe_emotion_score']:.2f})")

    except Exception as e:
        log.error(f"Foreign Extract Error: {e}")
        state['vibe_emotion_tag'] = "NEUTRAL"
        state['vibe_emotion_score'] = 0.0
        state['corrected_stt'] = stt_text

    return state
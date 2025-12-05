# src/talaria/hermes_service/nodes/node_vibe_extract.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from ..state import TranslateState
from ..utils.model_loader import get_scanner

log = logging.getLogger(__name__)

# --- 1. STT 보정용 LLM (gpt-4o-mini, temperature=0.0) ---
try:
    # 정확한 보정을 위해 temperature 0 유지
    correction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    log.info("✅ STT 보정용 LLM(gpt-4o-mini) 로드 완료")
except:
    correction_llm = None
    log.error("❌ STT 보정용 LLM 로드 실패")

# 🛑 [최종 수정] 대본 참조형 자연스러운 보정 프롬프트
# - Markdown Bold(**) 사용 금지 (Plain Text 출력)
# - 뉘앙스 유지 강화: 대본과 달라도 사용자의 말투/추임새는 보존

CORRECTION_PROMPT = """You are a Smart STT Corrector.
Your job is to fix **typos and phonetic errors** using the [Demo Script] as a reference.

### 🚨 GUIDELINES:
1. **Reference the Script:** Check the [Demo Script] below for correct spellings of key terms (memes, names) and phrases.
2. **Preserve Nuance:** DO NOT blindly replace the user's input with the script line if the meaning or tone is different.
   - If the user adds fillers (e.g., "좀", "완전", "진짜", "actually", "like") or changes the ending, **KEEP THEM**.
   - **ONLY** fix the **wrong words** (typos/phonetic errors/misheard names) to match the script's key terms.
   - 문장의 핵심적인 단어가 이상하게 작성되었을 때, 그 틀린 부분만 수정하세요. 문장에 들어가는 부사나, 문장 끝내는 어미 등은 되도록 수정하지 마세요. 
3. **Plain Text Only:** Output ONLY the corrected text. **NEVER** use markdown bolding (e.g., **word**) or explanations.

   [Example 1]
   - Script: "한국 드라마 보다보면 중간에 고구마 전개 많은 것 같아"
   - Input:  "한국 도라마 보다보면 중간에 고구마 전개가 좀 많은 것 같아"
   - Corrected: "한국 드라마 보다보면 중간에 고구마 전개가 좀 많은 것 같아"
   
   [Example 2]
   - Script: "Yeah, mint choco is my number one pick."
   - Input:  "Yeah, Minstocko is my number one peak, actually."
   - Corrected: "Yeah, Mint choco is my number one pick, actually."

### 📜 Demo Script (Spelling Reference):
[Korean]
- "한국 드라마 보다보면 중간에 고구마 전개 많은 것 같아…"
- "그래도 엔딩은 사이다니까 용서된다."
- "이번 여름 복근 위해서 갓생 살기 들어간다!"
- "일단 에스파 음악과 함께 아침 운동부터 시작해보려고."
- "방금 치맥 했는데 아이스크림도 먹고 싶네… 역시 단짠 조합이 맛있긴 해."
- "너 민초단이구나? 몰랐네."
- "요즘 감기 때문에 고생이야."
- "너 진짜 인자강 인정한다."
- "케이팝 데몬 헌터스 진짜 재밌더라. 세계관도 그렇고 노래도 완전 중독적이던데."
- "나도 따라해 본다고 커버 영상 찍어놨는데… 지금 생각해 보니 흑역사 인 것 같아."

[English]
- "Yeah, some episodes feel really slow and frustrating."
- "True, a good ending fixes everything."
- "Good luck, man. Staying consistent is the hardest part."
- "Go for it, you'll feel better every day."
- "I like mint chocolate ice cream the most."
- "Yeah, mint chocolate is my number one pick."
- "I barely get sick, even in winter I wear short sleeves."
- "Guess my body has high defense stats."
- "Yeah, it’s getting huge in the U.S. too. I see a lot of covers everywhere."
- "Don’t worry. I think it looks awesome, seriously."

---
(Real Input Starts Below)
Input: "{stt_text}"
Corrected:"""

correction_chain = ChatPromptTemplate.from_template(CORRECTION_PROMPT) | correction_llm | StrOutputParser()

# --- Foreign 감정 분석용 설정 ---
class EmotionResult(BaseModel):
    emotion: str = Field(description="One of: ADMIRATION, SHOCK, SARCASM, NEUTRAL")
    score: int = Field(description="Intensity score (0-100)")

foreign_parser = JsonOutputParser(pydantic_object=EmotionResult)
EMOTION_PROMPT = """Analyze the emotion and intensity. Return JSON: {{"emotion": "...", "score": 0~100}} Text: "{text}" """
foreign_chain = ChatPromptTemplate.from_template(EMOTION_PROMPT) | correction_llm | foreign_parser


# [Node 1] 한국어 전용 (LLM 보정 -> Scanner)
def vibe_extract_kor_agent(state: TranslateState) -> TranslateState:
    original_stt = state['stt_text']
    scanner = get_scanner()
    
    # [Step 1] STT 보정 (대본 참조형 유연한 보정)
    corrected_text = original_stt
    if correction_llm:
        try:
            corrected_text = correction_chain.invoke({"stt_text": original_stt}).strip().strip('"')
            if corrected_text != original_stt:
                log.info(f"🔧 [KO] STT 보정됨: '{original_stt}' -> '{corrected_text}'")
            else:
                log.info(f"👌 [KO] STT 유지: '{original_stt}'")
        except Exception as e:
            log.error(f"STT Correction Error (Kor): {e}")
            corrected_text = original_stt

    state['corrected_stt'] = corrected_text

    # [Step 2] Scanner (보정된 텍스트 사용)
    if not scanner:
        state['vibe_triggers'] = []
        state['vibe_emotion_tag'] = "NEUTRAL"
        state['vibe_emotion_score'] = 0.0
        return state

    try:
        prev_text = state.get("chat_history", [])[-1] if state.get("chat_history") else ""
        result = scanner.predict(current_text=corrected_text, prev_text=prev_text)
        
        state['vibe_triggers'] = result.get("meme_words", [])
        state['vibe_emotion_tag'] = result.get("nuance", "NEUTRAL")
        state['vibe_emotion_score'] = result.get("score", 0.0)
        
        log.info(f"🇰🇷 [Scanner] 감정: {state['vibe_emotion_tag']} (Score: {state['vibe_emotion_score']:.2f})")
        
    except Exception as e:
        log.error(f"Scanner Error: {e}")
        state['vibe_emotion_score'] = 0.0
        
    return state


# [Node 2] 외국어 전용 (LLM 보정 -> 감정 분석)
def vibe_extract_foreign_agent(state: TranslateState) -> TranslateState:
    original_stt = state['stt_text']
    
    # [Step 1] STT 보정 (영어 대본 참조)
    corrected_text = original_stt
    if correction_llm:
        try:
            corrected_text = correction_chain.invoke({"stt_text": original_stt}).strip().strip('"')
            if corrected_text != original_stt:
                log.info(f"🔧 [Foreign] STT 보정됨: '{original_stt}' -> '{corrected_text}'")
            else:
                log.info(f"👌 [Foreign] STT 유지: '{original_stt}'")
        except Exception as e:
            log.error(f"STT Correction Error (Foreign): {e}")
            corrected_text = original_stt
            
    state['corrected_stt'] = corrected_text

    # [Step 2] 감정 분석
    try:
        if correction_llm:
            res = foreign_chain.invoke({"text": corrected_text})
            emotion = res.get("emotion", "NEUTRAL").upper()
            score_val = res.get("score", 0)
            
            if emotion not in ["ADMIRATION", "SHOCK", "SARCASM", "NEUTRAL"]:
                emotion = "NEUTRAL"
                
            state['vibe_emotion_tag'] = emotion
            state['vibe_emotion_score'] = float(score_val) / 100.0
        else:
            state['vibe_emotion_tag'] = "NEUTRAL"
            state['vibe_emotion_score'] = 0.0
        
        state['vibe_triggers'] = []
        log.info(f"🌍 [Foreign] 감정: {state['vibe_emotion_tag']} (Score: {state['vibe_emotion_score']:.2f})")

    except Exception as e:
        log.error(f"Foreign Extract Error: {e}")
        state['vibe_emotion_tag'] = "NEUTRAL"
        state['vibe_emotion_score'] = 0.0

    return state
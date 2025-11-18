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

# ⭐️ [수정] 프롬프트에 agent_code 분석 추가
VIBE_EXTRACT_PROMPT = """You are an expert in correcting STT (Speech-to-Text) errors, classifying emotions, and analyzing cultural content.

Input Language: "{stt_lang}"
STT Input: "{stt_text}"
Context (Chat History):
{chat_history}

You MUST perform 3 tasks and respond ONLY with a JSON object:

1. **Correct STT:** Fix typos based on cultural context, internet slang, or common speech patterns.
   * (Example for "ko-KR"): "베그 폼 미쳣다" -> "배그 폼 미쳤다", "킹밧네" -> "킹받네"
   * (Example for "en-US"): "That's bussin" -> "That's bussin'", "I'm finna go" -> "I'm gonna go"

2. **Tag Emotion:** Classify the speaker's core emotion:
   [joy_celebrate, joy_excitement, love_appreciation, surprise_amaze, cool_success]
   If neutral, respond with "neutral".

3. **Classify Content (agent_code):** Analyze the content type:
   - 'k-meme': Korean internet memes, slang, K-Culture (예: "이거 레알", "억텐", "~각", "웅앵웅")
   - 'global-meme': Global/overseas memes (예: "YOLO", "LOL", "cringe", "slay")
   - 'music': Music, K-POP, idols (예: "노래", "앨범", "컴백", "음방", "song", "album")
   - 'game': Gaming, e-sports (예: "겜", "롤", "게임", "game", "LOL", "Overwatch")
   - 'mukbang': Food, cooking (예: "먹방", "맛집", "레시피", "mukbang", "recipe")
   - 'travel': Travel, tourism (예: "여행", "관광지", "현지", "travel", "trip")
   - 'vlog': Daily life, vlog (예: "일상", "브이로그", "데일리", "daily", "vlog")
   - 'talk': Casual chat (예: "썰", "ㄹㅇ", "개인적으로", "honestly", "personally")
   - 'study': Study, learning (예: "공부", "시험", "과제", "study", "exam", "homework")
   
   If unclear, use "global-meme" as default.

Respond ONLY with this JSON format:
{{
  "corrected_stt": "...",
  "emotion_tag": "...",
  "agent_code": "..."
}}
"""

prompt = ChatPromptTemplate.from_template(VIBE_EXTRACT_PROMPT)
parser = JsonOutputParser()
chain = prompt | llm | parser

def vibe_extract_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-1] (LLM 1: STT 보정 + 감정 + 컨텐츠 분석)
    ⭐️ [수정] agent_code 분석 추가
    """
    if not llm:
        log.error("Vibe LLM(Extract)이 로드되지 않아 Vibe 모드를 건너뜁니다.")
        state['corrected_stt'] = state['stt_text']
        state['vibe_emotion_tag'] = "neutral"
        state['agent_code'] = "global-meme"  # ⭐️ 기본값
        return state

    log.info(f"Vibe 추출 에이전트(LLM 1) 실행: (STT: '{state['stt_text']}')")
    try:
        history_str = "\n".join(f"- {h}" for h in state.get("chat_history", []))
        
        response = chain.invoke({
            "stt_lang": state.get("stt_lang", "ko-KR"),
            "stt_text": state['stt_text'],
            "chat_history": history_str
        })
        
        # STT 보정
        state['corrected_stt'] = response.get("corrected_stt", state['stt_text'])
        
        # 감정 분석
        state['vibe_emotion_tag'] = response.get("emotion_tag", "neutral")
        
        # ⭐️ [신규] agent_code 분석
        state['agent_code'] = response.get("agent_code", "global-meme")
        
        log.info(f"✅ Vibe 추출 완료:")
        log.info(f"   - Corrected STT: {state['corrected_stt']}")
        log.info(f"   - Emotion: {state['vibe_emotion_tag']}")
        log.info(f"   - 🎭 Agent Code: {state['agent_code']}")
        
    except Exception as e:
        log.error(f"Vibe LLM(Extract) 호출 중 오류: {e}", exc_info=True)
        state['corrected_stt'] = state['stt_text'] 
        state['vibe_triggers'] = []
        state['vibe_emotion_tag'] = "neutral"
        state['agent_code'] = "global-meme"  # ⭐️ 기본값
        
    return state
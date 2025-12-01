import logging
from ..state import TranslateState
from ..utils.model_loader import get_scanner

log = logging.getLogger(__name__)

def vibe_extract_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-1] (자체 모델: MemeScanner)
    기존 LLM 기반 추출을 대체하여, 자체 학습 모델(KcELECTRA)로 밈과 뉘앙스를 추출합니다.
    결과는 기존 State 필드인 'vibe_triggers'와 'vibe_emotion_tag'에 매핑됩니다.
    """
    scanner = get_scanner()
    
    # 모델 로드 실패 시 안전 장치 (기본값 반환)
    if not scanner:
        log.error("❌ MemeScanner 로드 실패. Vibe 추출 건너뜁니다.")
        state['vibe_triggers'] = []
        state['vibe_emotion_tag'] = "NEUTRAL"
        # 모델은 STT 보정 기능이 없으므로 원본 유지
        state['corrected_stt'] = state['stt_text']
        return state

    stt_text = state['stt_text']
    
    # 이전 대화 맥락 활용 (chat_history의 마지막 발화)
    chat_history = state.get("chat_history", [])
    prev_text = chat_history[-1] if chat_history else ""

    log.info(f"🤖 Vibe 모델 분석 시작: Input='{stt_text}' (Context='{prev_text}')")

    try:
        # 1. 모델 추론 실행 (0.05초 내외 소요)
        result = scanner.predict(current_text=stt_text, prev_text=prev_text)
        
        # 2. 결과 파싱
        meme_words = result.get("meme_words", [])
        nuance = result.get("nuance", "NEUTRAL")
        score = result.get("score", 0.0)

        # 3. 기존 State 필드에 매핑 (스키마 변경 없음)
        state['vibe_triggers'] = meme_words      # List[str] -> 밈 단어 리스트
        state['vibe_emotion_tag'] = nuance       # str -> 감정 태그
        state['corrected_stt'] = stt_text        # 원본 유지

        log.info(f"✅ 모델 분석 완료: Triggers={meme_words}, Emotion={nuance} (Conf: {score:.1f}%)")
        
    except Exception as e:
        log.error(f"❌ 모델 추론 중 오류: {e}", exc_info=True)
        state['vibe_triggers'] = []
        state['vibe_emotion_tag'] = "NEUTRAL"
        state['corrected_stt'] = stt_text
        
    return state
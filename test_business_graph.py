import logging
import os
import sys
import time 
from typing import List, Optional, cast
from dotenv import load_dotenv
from langchain_core.documents import Document

# --- 1. 환경 설정 ---
load_dotenv()
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(_THIS_FILE)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- 2. 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# --- 3. 핵심 모듈 임포트 ---
try:
    from talaria.hermes_service.state import TranslateState, FoundTerm
    from talaria.hermes_service.graph import app as langgraph_app
except ImportError as e:
    log.error(f"❌ 임포트 실패. 'src' 디렉토리가 올바른지, {e}")
    log.error("이 스크립트는 프로젝트 루트('src' 폴더가 있는 곳)에서 실행되어야 합니다.")
    sys.exit(1)
except ValueError as ve:
    log.error(f"❌ 그래프 컴파일 오류: {ve}")
    sys.exit(1)

# --- 4. 테스트 실행 함수 ---
def run_business_test(
    stt_text: str, 
    papago_translation: str, 
    target_lang: str, # ⭐️ 테스트할 대상 언어 추가
    chat_history: List[str]
):
    """
    주어진 입력으로 Business 모드 LangGraph를 1회 실행하고 결과를 출력합니다.
    """
    log.info(f"--- 🚀 테스트 시작: [STT] {stt_text} ---")

    # ⭐️ [수정] graph_input_state에 'target_lang' 필드 추가
    graph_input_state = cast(TranslateState, {
        "stt_text": stt_text,
        "translated_text": papago_translation,
        "chat_history": chat_history,
        "meeting_id": 62, 
        "translation_mode": "business", 
        
        "target_lang": target_lang, # ⭐️ 빠뜨렸던 필드 추가 (KeyError 해결)

        # --- 모든 필드 초기화 ---
        "context": [],
        "docs_id": None, 
        "found_terms": [],
        "should_correct_business": False, 
        "vibe_emotion_tag": None,
        "vibe_triggers": None,
        "final_answer": "", 
        "final_explanation": None,
        "final_emoji": None,
    })

    try:
        # --- 5. LangGraph 실행 (시간 측정) ---
        log.info("그래프 실행(invoke) 시작...")
        start_time = time.perf_counter()
        
        final_state = langgraph_app.invoke(graph_input_state)
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        # --- 시간 측정 완료 ---

        
        # --- 6. 결과 출력 ---
        log.info(f"--- ✅ 테스트 완료: [STT] {stt_text} ---")
        log.info(f"  > ⏱️ Graph 실행 시간: {duration_ms:.2f} ms")
        log.info(f"  > 1차 번역 (Papago) : {papago_translation}")
        log.info(f"  > 최종 번역 (LLM/Fallback) : {final_state.get('final_answer')}")
        
        if final_state.get('final_answer') == papago_translation:
            log.info("  > [결과: ⚡️ Fast Path] 'standard_fallback_agent'가 실행되었습니다.")
        else:
            log.info("  > [결과: 🧠 Slow Path] 'correction_agent'가 실행되었습니다.")
        
    except Exception as e:
        log.error(f"--- ❌ 테스트 실패: [STT] {stt_text} ---")
        log.error(f"  > 그래프 실행 중 오류 발생: {e}", exc_info=True)


# --- 7. 메인 테스트 실행 ---
if __name__ == "__main__":
    
    log.info("="*80)
    log.info("LangGraph (Business Mode) 테스트 스크립트 시작")
    log.info("LLM 및 임베딩 모델 로드를 기다리는 중...")
    log.info("="*80 + "\n")

    # ---
    # 테스트 케이스 1: "인사말 필터" (Fast Path)
    # ---
    run_business_test(
        stt_text="안녕하세요.",
        papago_translation="Hello.",
        target_lang="en-US", # ⭐️ 대상 언어 전달
        chat_history=[]
    )

    print("\n" + "="*80 + "\n")

    # ---
    # 테스트 케이스 2: "7단어 규칙" (Slow Path / STT 오류 보정)
    # ---
    run_business_test(
        stt_text="A 테스트 잘 진행되고 있어요.",
        papago_translation="A test is progressing well.", 
        target_lang="en-US", # ⭐️ 대상 언어 전달
        chat_history=["Ab 테스트 잘 진행되고 있니...."] 
    )
    
    print("\n" + "="*80 + "\n")

    # ---
    # 테스트 케이스 3: "전문 용어" (Slow Path / Aho-Corasick)
    # ---
    run_business_test(
        stt_text="CR 이슈 관련해서 말씀드립니다.",
        papago_translation="I'm talking about the CR issue.",
        target_lang="en-US", # ⭐️ 대상 언어 전달
        chat_history=[]
    )
    
    print("\n" + "="*80 + "\n")
    log.info("모든 테스트 완료.")
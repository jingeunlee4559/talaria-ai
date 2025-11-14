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
    # ⭐️ 2-LLM Call (8-Node) 아키텍처로 수정된 graph.py 임포트
    from talaria.hermes_service.graph import app as langgraph_app
except ImportError as e:
    log.error(f"❌ 임포트 실패. 'src' 디렉토리가 올바른지, {e}")
    log.error("이 스크립트는 프로젝트 루트('src' 폴더가 있는 곳)에서 실행되어야 합니다.")
    sys.exit(1)
except Exception as e:
    log.error(f"❌ 그래프 컴파일 오류. graph.py에 문제가 있습니다: {e}", exc_info=True)
    sys.exit(1)

# --- 4. 테스트 실행 함수 ---
def run_vibe_test(
    stt_text: str, 
    chat_history: List[str]
):
    """
    주어진 입력으로 Vibe 모드 LangGraph를 1회 실행하고 결과를 출력합니다.
    """
    log.info(f"--- 🚀 Vibe 테스트 시작: [STT] {stt_text} ---")

    # ⭐️ [수정] 2-LLM Call 아키텍처에 맞게 State 초기화
    # (state.py에 추가된 'corrected_stt', 'vibe_context' 필드 초기화)
    graph_input_state = cast(TranslateState, {
        "stt_text": stt_text,
        "translated_text": "", # Vibe 모드는 1차 번역을 건너뜀
        "chat_history": chat_history,
        "meeting_id": 63, 
        
        "translation_mode": "vibe", # ⭐️ Vibe 모드 강제
        "target_lang": "en-US", 
        
        # --- 모든 필드 초기화 ---
        "context": [],
        "docs_id": None, 
        "found_terms": [],
        "should_correct_business": False, 
        "vibe_emotion_tag": None,
        "vibe_triggers": None,
        
        # ⭐️ 2-LLM Call 신규 필드 초기화
        "corrected_stt": None,
        "vibe_context": None,

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

        
        # ⭐️ --- 6. 결과 출력 (2-LLM Call 아키텍처에 맞게 수정) ---
        log.info(f"--- ✅ Vibe 테스트 완료: [STT] {stt_text} ---")
        log.info(f"  > ⏱️ Graph 실행 시간: {duration_ms:.2f} ms")
        
        log.info(f"  --- [LLM 1: Extract] ---")
        log.info(f"  > [B-1] 🏷️ 인식된 용어 (LLM 1) : {final_state.get('vibe_triggers')}")
        log.info(f"  > [B-1] 💖 감정 태그 (LLM 1) : {final_state.get('vibe_emotion_tag')}")
        log.info(f"  > [B-1] 📝 STT 보정본 (LLM 1) : {final_state.get('corrected_stt')}")
        
        log.info(f"  --- [Rules: Enrich] ---")
        log.info(f"  > [B-2] 📖 문화 설명 (Rule) : {final_state.get('final_explanation')}")
        log.info(f"  > [B-2] 💡 힌트 (LLM 2용) : {final_state.get('vibe_context')}")
        log.info(f"  > [B-3] 😊 이모지 (Rule) : {final_state.get('final_emoji')}")
        
        log.info(f"  --- [LLM 2: Translate] ---")
        log.info(f"  > [B-4] 🤖 최종 번역 (LLM 2) : {final_state.get('final_answer')}")
        
    except Exception as e:
        log.error(f"--- ❌ Vibe 테스트 실패: [STT] {stt_text} ---")
        log.error(f"  > 그래프 실행 중 오류 발생: {e}", exc_info=True)


# --- 7. 메인 테스트 실행 (⭐️ 테스트 케이스 변경) ---
if __name__ == "__main__":
    
    log.info("="*80)
    log.info("LangGraph (Vibe Mode - 2-LLM Arch) 테스트 스크립트 시작")
    log.info("LLM 로드를 기다리는 중...")
    log.info("="*80 + "\n")

    # ---
    # [신규] 테스트 케이스 1: STT 오류 + 신규 밈 ("킹밧네" -> "킹받네")
    # kculture_rulebook.json: "킹받네" (방금 추가)
    # emotion_emoji_map.json: "anger"
    # ---
    run_vibe_test(
        stt_text="아 진짜 킹밧네",
        chat_history=[]
    )

    print("\n" + "="*80 + "\n")

    # ---
    # [신규] 테스트 케이스 2: "국밥" 오역 수정 테스트 ("찐" + "국밥")
    # kculture_rulebook.json: "찐", "국밥" (방금 추가 및 기존)
    # LLM 2가 "reliable" 힌트를 받아 "trash"가 아닌지 확인
    # ---
    run_vibe_test(
        stt_text="이 캐릭터 찐 국밥이네", 
        chat_history=[]
    )
    
    print("\n" + "="*80 + "\n")
    
    # ---
    # 테스트 케이스 3: 기존 "베그" (STT 오류 + 밈)
    # kculture_rulebook.json: "배그", "폼 미쳤다"
    # ---
    run_vibe_test(
        stt_text="야 걔는 완전 베그 고인물이야.",
        chat_history=[]
    )
    
    print("\n" + "="*80 + "\n")
    log.info("모든 테스트 완료.")
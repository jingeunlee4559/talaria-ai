#LangGraph의 AgentState (세션 상태) 정의

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from typing import Dict, Any, Optional, List, Literal
from .schemas.summary import ActionItem

class FoundTerm(TypedDict):
    term: str       # 찾은 단어 (예: "CR")
    meaning: str    # DB에서 조회한 의미 (예: "변경 요청서")
    category: str

class TranslateState(TypedDict):
    stt_text : str # 사용자 발화 STT 내용
    stt_lang: str #  STT 원본 언어
    translated_text : str # 사용자 발화 파파고 단순 번역 내용
    target_lang: str # 번역 되어야 하는 언어
    context : List[Document] # RAG 검색 결과 청크
    meeting_id : int # 현재 진행중인 회의 ID
    docs_id : Optional[List[int]] # 회의에 매핑된 문서들의 ID
    chat_history : Optional[List[str]] # 이전 회의 대화 10개
    found_terms: List[FoundTerm] # 찾은 용어 리스트

    should_correct_business: bool # 비지니스 문맥 번역 조건 만족 여부

    translation_mode: Literal["business", "vibe"] # 비지니스 / 컬쳐 라우터 제어용
    vibe_emotion_tag: Optional[str]     # B-1. LLM이 분석한 감정
    vibe_emotion_score: Optional[float]
    vibe_triggers: Optional[List[str]]  # B-1. LLM이 인식한 밈/용어

    # 2-LLM Call을 위한 Vibe 모드 전용 필드
    corrected_stt: Optional[str]      # LLM 1이 보정한 STT (예: "베그 폼 미쳣다" -> "배그 폼 미쳤다")
    vibe_context: Optional[str]       # LLM 2에 주입할 힌트 (예: "'국밥': reliable")

    # --- [신규] 최종 통합 결과 필드 ---
    final_answer: str                 # ⭐️ 최종 번역문 (기존 answer 대체)
    final_explanation: Optional[str]  # B-2. 규칙 DB가 추가한 설명
    final_emoji: Optional[str]        # B-3. 규칙 DB가 매핑한 이모지

    is_safe: bool                 # 안전 여부 (True/False)
    safety_reason: Optional[str]  # 필터링 사유 (예: HATE_SPEECH, SLANG)
    original_text_before_guard: Optional[str] # 필터링 전 원본 (로그용)


# --- 회의 요약용 State 정의 ---
class SummaryState(TypedDict):
    # --- Input ---
    transcript: str
    current_date: str
    participants: List[str] # 발언 점유율 계산용

    # --- Track A: Summary & Keyword ---
    summary_intro: str      # 한 줄 소개
    keywords: List[str]     # 토픽
    decisions: List[str]    # 결정 사항
    track_a_feedback: str   # 평가 피드백
    track_a_retry: int

    # --- Track B: Action Items ---
    action_items: List[Dict] # Pydantic Model -> Dict
    track_b_feedback: str
    track_b_retry: int

    # --- Track C: Risk Analysis (2+2 Hybrid) ---
    # C-1: Content (LLM)
    content_score: float
    content_risks: List[str]
    content_improvements: List[str]
    
    # C-2: Process (Python Rule)
    process_score: float
    process_risks: List[str]
    
    # C-Merged
    coach_next_steps: List[str] # 전략적 제안

    # --- Final ---
    final_output: Dict[str, Any]
# src/talaria/hermes_service/schemas/utterance.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class UtteranceProcessIn(BaseModel):
    """livekit_agent -> hermes_service (입력)"""
    meeting_id: int = Field(..., description="현재 회의 ID (tl_meetings.meeting_id)")
    speaker_id: str = Field(..., description="발화자 LiveKit Identity")
    track_sid: Optional[str] = Field(None, description="오디오 트랙 SID")
    
    stt_text: str = Field(..., description="STT 원본 텍스트")
    lang: str = Field(..., description="STT 원본 언어 코드 (예: ko-KR)")
    confidence: Optional[float] = Field(None, description="STT 신뢰도 (0.0 ~ 1.0)")
    started_ms: Optional[int] = Field(None, description="발화 시작 시간 (ms)")
    ended_ms: Optional[int] = Field(None, description="발화 종료 시간 (ms)")
    
    # 1차 단순 번역본 (Business 모드의 Fast Path용)
    simple_translations: Dict[str, str] = Field(..., description="Papago 단순 번역 결과 (예: {'en-US': 'Hello'})")

    target_lang: str = Field("en-US", description="LLM 보정/번역을 수행할 대상 언어 (e.g., en-US)")

    # ⭐️ [신규] 모드 플래그
    translation_mode: str = Field("business", description="번역 모드 ('business' or 'vibe')")

class UtteranceProcessOut(BaseModel):
    """hermes_service -> livekit_agent (출력)"""
    stt_text: str = Field(..., description="입력받은 원본 STT")
    lang: str = Field(..., description="입력받은 원본 언어")
    
    # 그래프가 보정한 최종 번역본
    contextual_translations: Dict[str, str] = Field(..., description="최종 번역본 (Vibe톤, 이모지 포함 가능)")
    vibe_triggers: Optional[List[str]] = Field(None, description="Vibe 모드에서 감지된 키워드/밈 목록")
    final_explanation: Optional[str] = Field(None, description="Vibe 모드 문화/밈 설명 (DB 저장용)")
    final_emoji: Optional[str] = Field(None, description="Vibe 모드 감정 이모지 (Agent가 사용)")
    corrected_stt: Optional[str] = Field(None, description="[신규] Vibe 모드에서 보정된 STT")
    ai_tags: Dict = Field(..., description="DB에 저장될 모든 AI 분석 데이터")
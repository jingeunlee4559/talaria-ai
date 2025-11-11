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
    
    # agent.py가 생성한 1차 단순 번역본
    simple_translations: Dict[str, str] = Field(..., description="Papago 단순 번역 결과 (예: {'en-US': 'Hello'})")

class UtteranceProcessOut(BaseModel):
    """hermes_service -> livekit_agent (출력)"""
    stt_text: str = Field(..., description="입력받은 원본 STT")
    lang: str = Field(..., description="입력받은 원본 언어")
    
    # 그래프가 보정한 최종 번역본
    contextual_translations: Dict[str, str] = Field(..., description="문맥 보정이 적용된 최종 번역본")
    ai_tags: Dict = Field(..., description="DB에 저장된 ai_tags (manner 등 포함)")
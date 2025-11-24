from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class MeetingSummaryIn(BaseModel):
    meeting_id: str = Field(..., description="회의 고유 ID")
    current_date: str = Field(..., description="기준 날짜 (YYYY-MM-DD)")
    meeting_title: str = Field(..., description="회의 제목")
    participants: str = Field(..., description="참석자 목록 (콤마 구분)")
    transcript: str = Field(..., description="회의 스크립트 본문")

class MeetingSummaryOut(BaseModel):
    meta: Dict[str, Any]
    result: Dict[str, Any]
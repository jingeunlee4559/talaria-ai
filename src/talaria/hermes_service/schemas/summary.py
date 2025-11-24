# src/talaria/hermes_service/schemas/summary.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# --- 1. Action Item (Track B) ---
class ActionItem(BaseModel):
    owner: str = Field(..., description="담당자 이름 (예: 김철수, 미정)")
    task: str = Field(..., description="할 일 내용")
    deadline: str = Field(..., description="마감일 (예: '마감: 2025-10-05' 또는 '미정')")
    priority: str = Field("MEDIUM", description="우선순위 (HIGH, MEDIUM, LOW)")

# --- 2. AI Summary (Track A + B) ---
class AiSummaryObj(BaseModel):
    decisions: List[str] = Field(..., description="확정된 의사결정 리스트")
    actionItems: List[ActionItem] = Field(..., description="도출된 액션 아이템 리스트")

# --- 3. AI Coach (Track C) ---
class AiCoachObj(BaseModel):
    summary: str = Field(..., description="회의 전체를 관통하는 한 줄 요약")
    meetingScore: int = Field(..., description="4대 지표 종합 점수 (0~100)")
    riskLevel: str = Field(..., description="위험도 ('높음' | '보통' | '낮음')")
    risks: List[str] = Field(..., description="감지된 리스크 목록")
    improvements: List[str] = Field(..., description="개선 제안 목록")
    nextSteps: List[str] = Field(..., description="다음 액션 제안 (전략적 조언)")

# --- 4. 최종 API 입출력 ---
class MeetingSummaryIn(BaseModel):
    meeting_id: int
    current_date: str
    meeting_title: str
    participants: str
    transcript: str

class MeetingSummaryOut(BaseModel):
    meta: Dict[str, Any]
    # 프론트엔드가 'mockMinutes'의 요소로 바로 쓸 수 있는 구조
    result: Dict[str, Any]  # { aiSummary: ..., aiCoach: ..., topics: ... }
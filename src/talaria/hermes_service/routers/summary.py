from fastapi import APIRouter, HTTPException
from ..schemas.summary import MeetingSummaryIn, MeetingSummaryOut
from ..services import summary_service

router = APIRouter(prefix="/summary", tags=["summary"])

@router.post(
    "/process",
    summary="회의록 요약 및 액션 아이템 추출",
    response_model=MeetingSummaryOut
)
async def summarize_meeting(payload: MeetingSummaryIn):
    try:
        result = await summary_service.process_meeting_summary(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 처리 실패: {str(e)}")
# src/talaria/hermes_service/routers/utterance.py
from fastapi import APIRouter, Body, HTTPException
from ..schemas.utterance import UtteranceProcessIn, UtteranceProcessOut
from ..services import utterance_service

router = APIRouter(prefix="/utterances", tags=["utterances"])

@router.post(
    "/process",
    summary="STT 발화 문맥 보정 및 DB 저장",
    response_model=UtteranceProcessOut
)
def process_utterance(payload: UtteranceProcessIn):
    try:
        result = utterance_service.process_and_save_utterance(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"발화 처리 중 오류: {e}")
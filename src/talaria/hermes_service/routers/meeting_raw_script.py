from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter(
    prefix="/meeting",
    tags=["meeting-raw-script"],
)


class RawScriptRequest(BaseModel):
    meeting_id: int
    raw_script: str


@router.post("/raw-script")
async def handle_raw_script(req: RawScriptRequest) -> Dict[str, Any]:
    """
    백엔드(Java)에서 raw script text를 보내오면
    나중에 여기서 LLM 요약/파일 저장 등을 할 수 있음.
    지금은 일단 잘 들어오는지만 확인용으로 echo 응답.
    """
    # TODO: 여기서 파일로 저장하거나, LLM 호출 등 원하는 작업 수행
    #       (현재는 단순 echo)
    return {
        "status": "ok",
        "meeting_id": req.meeting_id,
        "received_length": len(req.raw_script),
    }

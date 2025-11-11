from fastapi import APIRouter, Body, HTTPException, Query
from starlette.responses import StreamingResponse
from typing import Optional

from ..schemas.docs_translation import (
    TranslateDocIn, 
    TranslateDocOut, 
    JobStatus
)
from ..services import docs_translation

router = APIRouter(prefix="/jobs", tags=["translation"])

# ---- 엔드포인트 (로직이 서비스 호출로 변경됨) ----
@router.post(
    "/translate-doc",
    summary="문서 번역 실행(제출→폴링→다운로드→저장)",
    response_model=TranslateDocOut,
)
def translate_doc(payload: TranslateDocIn = Body(
    ...,
    examples={
        "rel-path": {
            "summary": "상대경로 사용(권장)",
            "value": {
                "rel_path": "2025/11/10/sample_ko.hwp",
                "source": "ko",
                "target": "en"
            },
        },
        "abs-path": {
            "summary": "절대경로 사용",
            "value": {
                "abs_path": "C:/workspace/gitaction_assignment/talaria-ai/data/uploads/docs/2025/11/10/sample_ko.hwp",
                "source": "ko",
                "target": "en"
            },
        },
    }
)):

    try:
        return docs_translation.process_document_translation(payload)
    except HTTPException as e:
        # 서비스에서 발생한 HTTPException을 그대로 다시 발생시킴
        raise e
    except Exception as e:
        # 서비스에서 미처 처리하지 못한 일반 오류
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
    
@router.get(
    "/translate-doc/stream",
    summary="문서 번역 진행률 스트림(SSE/비동기)",
)
async def translate_doc_stream(
    rel_path: Optional[str] = Query(None, description="업로드 루트 기준 상대경로"),
    abs_path: Optional[str] = Query(None, description="절대경로"),
    source: str = Query(..., description="원본 언어 코드"),
    target: str = Query(..., description="타깃 언어 코드"),
    glossary_id: Optional[str] = Query(None, description="(선택) 용어집 ID"),
):
    """
    - SSE로 진행률(percent)과 상태를 연속 전송
    - 최종 완료 시 결과 파일 저장 후 complete 이벤트 전송
    """
    
    # --- (변경) 모든 로직을 service의 async generator로 위임 ---
    event_generator = docs_translation.stream_document_translation(
        rel_path=rel_path,
        abs_path=abs_path,
        source=source,
        target=target,
        glossary_id=glossary_id
    )

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream; charset=utf-8",
        "Connection": "keep-alive",
    }
    # --- (변경) 서비스가 반환한 생성기를 StreamingResponse로 감싸서 반환 ---
    return StreamingResponse(event_generator, headers=headers, media_type="text/event-stream")
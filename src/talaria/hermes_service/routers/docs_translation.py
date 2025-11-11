from fastapi import APIRouter, Body, HTTPException

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
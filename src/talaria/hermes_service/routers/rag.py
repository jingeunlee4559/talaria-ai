# src/talaria/hermes_service/routers/rag.py
from fastapi import APIRouter, Body, HTTPException, BackgroundTasks

# RAG용 스키마 및 서비스 import
from ..schemas.rag import ProcessDocIn, ProcessDocOut
from ..services import rag_service

# (참고) RAG API는 /rag 접두사를 사용하도록 분리
router = APIRouter(prefix="/rag", tags=["rag"])

@router.post(
    "/process-document",
    summary="문서 VectorDB화(RAG) 파이프라인 실행",
    description="지정된 문서를 백그라운드에서 파싱, 청킹, 임베딩하여 VectorDB에 저장합니다.",
    response_model=ProcessDocOut,
)
async def process_document(
    # (핵심) BackgroundTasks를 매개변수로 추가
    background_tasks: BackgroundTasks, 
    payload: ProcessDocIn = Body(
        ...,
        examples={
            "example": {
                "summary": "RAG 파이프라인 시작",
                "value": {
                    "document_id": "doc-uuid-12345",
                    "rel_path": "2025/11/11/some_report.pdf",
                    "language": "ko"
                },
            },
        }
    )
):
    """
    문서 RAG 파이프라인을 백그라운드로 실행합니다.
    - **document_id**: `tl_documents` 테이블의 `doc_id` (문자열)
    - **rel_path**: (권장) `data/uploads/docs` 기준 상대 경로
    - **language**: 문서의 원본 언어 (예: "ko", "en")
    """
    try:
        # RAG 서비스 호출 (BackgroundTasks 객체 전달)
        response = await rag_service.start_document_processing(payload, background_tasks)
        
        if response.status == "FAILED":
            # 경로 검증 등에서 동기적으로 실패한 경우
            raise HTTPException(status_code=400, detail=response.message)
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
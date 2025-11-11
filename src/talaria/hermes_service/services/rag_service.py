# src/talaria/hermes_service/services/rag_service.py
from fastapi import HTTPException, BackgroundTasks
from pathlib import Path
import logging

# RAG 파이프라인 매니저 import
from ..rag_processor.manager import process_and_load_document
# 공용 스키마 import
from ..schemas.rag import ProcessDocIn, ProcessDocOut, RAGJobStatus
# 공용 유틸리티 import
from ..utils.file_utils import (
    _get_root,
    _build_abs_path,
    _ensure_under_root
)

log = logging.getLogger(__name__)

async def start_document_processing(
    payload: ProcessDocIn, 
    background_tasks: BackgroundTasks
) -> ProcessDocOut:
    """
    RAG 파이프라인을 백그라운드 작업으로 시작시킵니다.
    """
    try:
        root = _get_root()
        # (참고) ProcessDocIn이 TranslateDocIn과 필드가 호환되므로 file_utils 사용 가능
        p = _build_abs_path(payload, root) 
        _ensure_under_root(p, root)
    except HTTPException as e:
        # 경로 검증 실패 시
        return ProcessDocOut(
            status=RAGJobStatus.FAILED,
            document_id=payload.document_id,
            message=f"파일 경로 유효성 검사 실패: {e.detail}"
        )

    # (핵심) process_and_load_document 함수를 백그라운드에서 실행
    background_tasks.add_task(
        process_and_load_document,
        file_path=str(p),
        document_id=payload.document_id,
        language=payload.language
    )
    
    log.info(f"RAG 파이프라인 백그라운드 작업 시작 (Doc ID: {payload.document_id})")

    # API는 즉시 응답 반환
    return ProcessDocOut(
        status=RAGJobStatus.PROCESSING,
        document_id=payload.document_id,
        message="문서 벡터화(RAG) 파이프라인이 백그라운드에서 시작되었습니다."
    )
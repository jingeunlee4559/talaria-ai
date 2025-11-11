# src/talaria/hermes_service/schemas/rag.py
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field

class RAGJobStatus(str, Enum):
    """RAG 처리 상태"""
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED" # (참고: 백그라운드 작업이라 이 응답은 거의 없음)

class ProcessDocIn(BaseModel):
    """
    RAG 파이프라인 입력 스키마.
    (TranslateDocIn과 거의 동일하지만, doc_id와 language가 필수)
    """
    document_id: str = Field(..., description="문서를 식별하는 고유 ID (tl_documents의 doc_id)")
    abs_path: Optional[str] = Field(None, description="원본 문서 절대경로")
    rel_path: Optional[str] = Field(None, description="업로드 루트 기준 상대경로")
    language: str = Field(..., description="문서의 원본 언어 (예: ko, en)")

class ProcessDocOut(BaseModel):
    """RAG 파이프라인 즉시 응답 스키마"""
    status: RAGJobStatus = Field(..., description="작업 접수 상태")
    document_id: str = Field(..., description="처리를 시작한 문서 ID")
    message: str = Field(..., description="응답 메시지")
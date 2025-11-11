from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field

# ---- 상태 스키마 ----
class JobStatus(str, Enum):
    COMPLETED = "COMPLETED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    FAIL = "FAIL"
    PROCESSING = "PROCESSING"
    PROGRESS = "PROGRESS"
    PENDING = "PENDING"

# ---- 입력 모델 ----
class TranslateDocIn(BaseModel):
    abs_path: Optional[str] = Field(
        None, description="원본 문서 절대경로 (DOC_UPLOAD_ROOT 하위)"
    )
    rel_path: Optional[str] = Field(
        None, description="업로드 루트 기준 상대경로 (예: 2025/11/10/file.hwp)"
    )
    source: str = Field(..., description="원본 언어 코드 (예: ko, en, ja, zh-CN)")
    target: str = Field(..., description="타깃 언어 코드 (예: en, ko, ja, zh-CN)")
    glossary_id: Optional[str] = Field(None, description="(선택) Papago 용어집 ID")

# ---- 출력 모델 ----
class TranslateDocOut(BaseModel):
    status: JobStatus = Field(..., description="Papago 상태")
    requestId: Optional[str] = Field(None, description="Papago 요청 ID")
    output_path: Optional[str] = Field(None, description="번역 결과 파일 절대경로")
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
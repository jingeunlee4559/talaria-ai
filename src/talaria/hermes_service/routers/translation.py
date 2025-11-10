# src/talaria/hermes_service/routers/translation.py
import os
import os.path as osp
from pathlib import Path
from typing import Optional
from enum import Enum

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/jobs", tags=["translation"])


# ---- 상태 스키마 (Pydantic v1/2 호환) ----
class JobStatus(str, Enum):
    COMPLETED = "COMPLETED"
    COMPLETE = "COMPLETE"  # ✅ Papago가 반환하는 값
    FAILED = "FAILED"
    FAIL = "FAIL"  # ✅ 혹시 모르니 추가
    PROCESSING = "PROCESSING"
    PROGRESS = "PROGRESS"  # ✅ 로그에서 본 값
    PENDING = "PENDING"

# ---- 입력: abs_path 또는 rel_path 중 하나 필수 ----
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

class TranslateDocOut(BaseModel):
    status: JobStatus = Field(..., description="Papago 상태")
    requestId: Optional[str] = Field(None, description="Papago 요청 ID")
    output_path: Optional[str] = Field(None, description="번역 결과 파일 절대경로")
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None

# ---- 유틸: 루트/경로 빌드 & 안전검사 ----
def _get_root() -> Path:
    root = os.getenv("DOC_UPLOAD_ROOT", "talaria-ai/data/uploads/docs")
    return Path(root).resolve()

def _normalize_seps(s: str) -> str:
    return s.replace("\\", "/")

def _build_abs_path(payload: TranslateDocIn, root: Path) -> Path:
    if payload.abs_path:
        return Path(_normalize_seps(payload.abs_path)).resolve()
    if payload.rel_path:
        return (root / _normalize_seps(payload.rel_path)).resolve()
    raise HTTPException(status_code=400, detail="abs_path 또는 rel_path 중 하나는 필수입니다.")

def _ensure_under_root(p: Path, root: Path):
    # 드라이브/대소문자/슬래시 차이 보정한 공통경로 비교
    try:
        p_norm    = osp.normcase(str(p))
        root_norm = osp.normcase(str(root))
        common    = osp.normcase(osp.commonpath([p_norm, root_norm]))
    except Exception:
        # 드라이브 불일치 등
        raise HTTPException(status_code=400, detail="업로드 루트 밖의 경로입니다.")
    if common != root_norm:
        raise HTTPException(status_code=400, detail="업로드 루트 밖의 경로입니다.")

# ---- 엔드포인트 ----
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
    root = _get_root()
    p = _build_abs_path(payload, root)
    _ensure_under_root(p, root)

    print("DOC_UPLOAD_ROOT =", root)
    print("REQUEST_PATH   =", p)

    # 날짜 폴더 규칙 확인: [..., YYYY, MM, DD, file]
    parts = p.parts[-4:-1]
    if len(parts) != 3 or not all(x.isdigit() for x in parts):
        raise HTTPException(status_code=400, detail="날짜 폴더(YYYY/MM/DD) 규칙 위반")

    # Papago 클라이언트 (실제 위치에 맞춰 import)
    try:
        from talaria.common.integrations.papago_doc import PapagoDocClient
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Papago 클라이언트를 불러올 수 없습니다: {e}")

    client = PapagoDocClient()
    try:
        req_id = client.translate(p, payload.source, payload.target, payload.glossary_id)
        meta = client.wait_until_done(
            req_id,
            poll_interval=int(os.getenv("TRANSLATION_POLL_INTERVAL", "2")),
            timeout=int(os.getenv("TRANSLATION_POLL_TIMEOUT", "600")),
        )
        content = client.download(req_id)
    except Exception as e:
        # papago_doc.py에서 상세 본문을 포함하도록 raise하는 것을 권장
        raise HTTPException(status_code=502, detail=f"Papago 처리 실패: {e}")

    out = p.with_name(f"{p.stem}.translated.{payload.target}{p.suffix}")
    out.write_bytes(content)

    # ✅ Papago 상태를 표준 JobStatus Enum으로 매핑
    papago_status = meta.get("status", "COMPLETE")
    status_mapping = {
        "COMPLETE": JobStatus.COMPLETED,
        "COMPLETED": JobStatus.COMPLETED,
        "FAIL": JobStatus.FAILED,
        "FAILED": JobStatus.FAILED,
        "PROGRESS": JobStatus.PROCESSING,
        "PROCESSING": JobStatus.PROCESSING,
        "PENDING": JobStatus.PENDING,
    }
    final_status = status_mapping.get(papago_status, JobStatus.COMPLETED)

    return TranslateDocOut(
        status=final_status,  # ✅ 매핑된 Enum 값 사용
        requestId=meta.get("requestId", req_id),
        output_path=str(out),
        source_lang=payload.source,
        target_lang=payload.target,
    )

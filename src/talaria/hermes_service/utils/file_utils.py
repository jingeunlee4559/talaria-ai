# src/talaria/hermes_service/utils/file_utils.py
import os
import os.path as osp
from pathlib import Path
from fastapi import HTTPException
from ..schemas.docs_translation import TranslateDocIn # docs_translation의 스키마를 그대로 재사용

# (참고) RAG 스키마에서 rel_path, abs_path를 사용한다면 해당 스키마로 변경해도 됩니다.
# from ..schemas.rag import ProcessDocIn 

def _get_root() -> Path:
    """DOC_UPLOAD_ROOT 환경 변수를 기준으로 절대 경로를 반환합니다."""
    root = os.getenv("DOC_UPLOAD_ROOT", "data/uploads/docs")
    return Path(root).resolve()

def _normalize_seps(s: str) -> str:
    """경로 구분자를 / (Forward slash)로 통일합니다."""
    return s.replace("\\", "/")

def _build_abs_path(payload: TranslateDocIn, root: Path) -> Path: # Type Hint는 스키마에 맞게 조정
    """payload의 abs_path 또는 rel_path를 기반으로 절대 경로를 빌드합니다."""
    if payload.abs_path:
        return Path(_normalize_seps(payload.abs_path)).resolve()
    if payload.rel_path:
        return (root / _normalize_seps(payload.rel_path)).resolve()
    raise HTTPException(status_code=400, detail="abs_path 또는 rel_path 중 하나는 필수입니다.")

def _ensure_under_root(p: Path, root: Path):
    """해당 경로가 DOC_UPLOAD_ROOT 하위에 있는지 보안 검사를 수행합니다."""
    try:
        p_norm    = osp.normcase(str(p))
        root_norm = osp.normcase(str(root))
        common    = osp.normcase(osp.commonpath([p_norm, root_norm]))
    except Exception:
        # 드라이브 불일치 등
        raise HTTPException(status_code=400, detail="업로드 루트 밖의 경로입니다.")
    if common != root_norm:
        raise HTTPException(status_code=400, detail="업로드 루트 밖의 경로입니다.")
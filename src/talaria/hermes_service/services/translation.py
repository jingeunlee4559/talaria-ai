from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import os

from services.papago_doc import PapagoDocClient

router = APIRouter(prefix="/jobs", tags=["translation"])

class TranslateDocIn(BaseModel):
    # 예: "talaria-ai/data/uploads/docs/2025/11/10/Report_ko.docx"
    abs_path: str = Field(..., description="원본 파일 절대경로")
    source: str = Field(..., description="원본 언어 코드 (예: ko, en, ja, zh-CN)")
    target: str = Field(..., description="타깃 언어 코드 (예: en, ko, ja, zh-CN)")
    glossary_id: str | None = Field(None, description="Papago 용어집 ID (선택)")

@router.post("/translate-doc")
def translate_doc(payload: TranslateDocIn):
    p = Path(payload.abs_path).resolve()
    root = Path(os.getenv("DOC_UPLOAD_ROOT", "talaria-ai/data/uploads/docs")).resolve()
    if not p.exists() or not str(p).startswith(str(root)):
        raise HTTPException(status_code=400, detail="경로가 존재하지 않거나 업로드 루트 밖입니다.")

    client = PapagoDocClient()
    try:
        req_id = client.translate(p, payload.source, payload.target, payload.glossary_id)
        meta = client.wait_until_done(req_id,
                                      poll_interval=int(os.getenv("TRANSLATION_POLL_INTERVAL", "2")),
                                      timeout=int(os.getenv("TRANSLATION_POLL_TIMEOUT", "600")))
        content = client.download(req_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Papago 처리 실패: {e}")

    # 저장: 동일 날짜 폴더에 "원본이름.translated.{target}.{ext}"
    date_parts = p.parts[-4:-1]  # ["2025","11","10"]
    if len(date_parts) != 3:
        raise HTTPException(status_code=400, detail="날짜 폴더(YYYY/MM/DD) 규칙 위반")
    stem = p.stem  # Report_ko
    ext = p.suffix  # .docx
    out = p.with_name(f"{stem}.translated.{payload.target}{ext}")
    out.write_bytes(content)

    return {
        "requestId": meta.get("requestId"),
        "status": meta.get("status"),
        "output_path": str(out),
        "source_lang": payload.source,
        "target_lang": payload.target
    }

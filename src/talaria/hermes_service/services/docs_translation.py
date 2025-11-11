import os
import os.path as osp
from pathlib import Path
from typing import Optional, AsyncGenerator, Dict, Any
from fastapi import HTTPException
import asyncio
import json
from ...common.integrations.papago_doc import PapagoDocClient
from ...core.database import get_db_cursor

from ..schemas.docs_translation import (
    TranslateDocIn, 
    TranslateDocOut, 
    JobStatus
)

from ..utils.file_utils import (
    _get_root,
    _build_abs_path,
    _ensure_under_root
)


def _get_original_doc_id(original_rel_path: str) -> Optional[int]:
    """
    원본 파일의 상대 경로를 사용해 tl_documents 테이블에서 doc_id를 조회합니다.
    """
    try:
        with get_db_cursor(dictionary=True) as cursor:
            # Java 백엔드에서 저장한 경로 형식(Forward slash)으로 비교
            sql = "SELECT doc_id FROM tl_documents WHERE filepath = %s LIMIT 1"
            cursor.execute(sql, (original_rel_path,))
            result = cursor.fetchone()
            
            if result:
                return result['doc_id']
            else:
                return None
    except Exception as e:
        print(f"❌ DB 조회 오류 (_get_original_doc_id): {e}")
        # DB 조회에 실패해도 일단 번역은 성공한 것으로 처리하기 위해 None 반환
        return None

# --- (신규 추가) DB 저장을 위한 헬퍼 함수 ---

def _save_translation_record(
    original_doc_id: Optional[int],
    target_language: str,
    translated_rel_path: str,
    request_id: str,
    size_bytes: int
):
    """
    번역 완료된 정보를 tl_trans_docs 테이블에 저장합니다.
    """
    # original_doc_id가 없으면(DB에서 못 찾았으면) 0 또는 NULL로 저장 (FK 제약조건에 따라)
    # 여기서는 NULL을 허용한다고 가정하고, 없으면 0으로 저장
    doc_id_to_save = original_doc_id if original_doc_id is not None else 0

    try:
        with get_db_cursor(commit=True) as cursor:
            # translated_at, created_at, updated_at는 DB의 CURRENT_TIMESTAMP 기본값을 사용
            sql = """
                INSERT INTO tl_trans_docs (
                    original_doc_id, target_language, translated_rel_path,
                    request_id, provider, size_bytes, translated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            params = (
                doc_id_to_save,
                target_language,
                translated_rel_path,
                request_id,
                "papago_doc", # provider (테이블 기본값 사용도 가능)
                size_bytes
            )
            cursor.execute(sql, params)
            print(f"✅ 번역 기록 DB 저장 완료. (Original Doc ID: {doc_id_to_save})")

    except Exception as e:
        # (중요) DB 저장에 실패하더라도, 이미 파일 번역은 성공했으므로
        # API 자체를 실패시키지 않고 로그만 남깁니다.
        print(f"❌ DB 저장 오류 (_save_translation_record): {e}")

# --- 기존 routers/translation.py 에 있던 핵심 로직 (함수로 분리) ---

def process_document_translation(payload: TranslateDocIn) -> TranslateDocOut:
    """
    문서 번역의 핵심 비즈니스 로직을 수행합니다.
    """
    root = _get_root()
    p = _build_abs_path(payload, root)
    _ensure_under_root(p, root)

    print("DOC_UPLOAD_ROOT =", root)
    print("REQUEST_PATH   =", p)

    # 날짜 폴더 규칙 확인: [..., YYYY, MM, DD, file]
    parts = p.parts[-4:-1]
    if len(parts) != 3 or not all(x.isdigit() for x in parts):
        raise HTTPException(status_code=400, detail="날짜 폴더(YYYY/MM/DD) 규칙 위반")

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

    # 1. 번역 파일 저장
    out = p.with_name(f"{p.stem}.translated.{payload.target}{p.suffix}")
    out.write_bytes(content)
    content_size_bytes = len(content)
    
    # --- (신규) DB 저장을 위한 데이터 준비 ---
    try:
        # 2. 원본 파일의 상대 경로 계산
        # (예: "2025/11/11/1762819217073_2025_09_26회의록.pdf")
        original_rel_path = str(p.relative_to(root)).replace("\\", "/")
        
        # 3. 원본 doc_id 조회
        original_doc_id = _get_original_doc_id(original_rel_path)
        if original_doc_id is None:
             # 여기서 오류를 발생시키지 않고, 경고만 남깁니다.
            print(f"⚠️ 경고: 원본 문서를 DB(tl_documents)에서 찾을 수 없습니다. (경로: {original_rel_path})")

        # 4. 번역된 파일의 상대 경로 계산
        translated_rel_path = str(out.relative_to(root)).replace("\\", "/")
        
        # 5. Papago 요청 ID
        request_id = meta.get("requestId", req_id)
        
        # 6. DB 저장 함수 호출
        _save_translation_record(
            original_doc_id=original_doc_id,
            target_language=payload.target,
            translated_rel_path=translated_rel_path,
            request_id=request_id,
            size_bytes=content_size_bytes
        )
    
    except Exception as e:
        # DB 작업 중 오류가 발생해도 번역 파일은 생성되었으므로 API를 중단시키지 않음
        print(f"❌ 번역 후 DB 기록 작업 중 예기치 않은 오류 발생: {e}")

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

# --- (신규) SSE 포맷 헬퍼 ---
def _sse_event(payload: dict) -> bytes:
    """SSE 포맷: data: <json>\n\n"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# --- (신규) 비동기식(Async) 스트리밍 로직 ---
async def stream_document_translation(
    rel_path: Optional[str],
    abs_path: Optional[str],
    source: str,
    target: str,
    glossary_id: Optional[str]
) -> AsyncGenerator[bytes, Any]:
    """
    문서 번역을 비동기 생성기(async generator)로 스트리밍합니다.
    (기존 router의 event_gen 로직)
    """
    
    # --- 1. 경로 유효성 검사 (로직 시작점) ---
    try:
        root = _get_root()
        # Pydantic 모델을 임시로 사용해 유효성 검사 로직 재사용
        tmp_payload = TranslateDocIn(abs_path=abs_path, rel_path=rel_path, source=source, target=target, glossary_id=glossary_id)
        p = _build_abs_path(tmp_payload, root)
        _ensure_under_root(p, root)
    except HTTPException as e:
        yield _sse_event({"type": "failed", "status": "FAILED", "error": f"validation_error: {e.detail}"})
        return
    except Exception as e:
        yield _sse_event({"type": "failed", "status": "FAILED", "error": f"validation_error: {str(e)}"})
        return

    # --- 2. Papago 클라이언트 및 번역 제출 ---
    client = PapagoDocClient()

    try:
        req_id = client.translate(p, source, target, glossary_id)
    except Exception as e:
        yield _sse_event({"type": "failed", "status": "FAILED", "error": f"submit_error: {e}"})
        return

    # --- 3. 시작 이벤트 전송 ---
    yield _sse_event({"type": "started", "requestId": req_id, "status": "PENDING", "percent": 0})
    
    # --- 4. 상태 폴링 및 완료/실패 처리 ---
    try:
        while True:
            # (중요) PapagoDocClient에 get_status가 있어야 함
            meta = client.get_status(req_id) 
            data = (meta or {}).get("data", {})
            status_raw = str(data.get("status", "")).upper()
            percent = int(data.get("progressPercent", 0))

            # 진행 이벤트
            if status_raw in ("PROGRESS", "PROCESSING"):
                yield _sse_event({"type": "progress", "status": "PROCESSING", "percent": max(0, min(99, percent))})
                await asyncio.sleep(1.2) # (중요) DB 부하를 줄이기 위해 sleep은 service 쪽에
                continue

            # 완료
            if status_raw in ("COMPLETE", "COMPLETED"):
                content = client.download(req_id)
                out = p.with_name(f"{p.stem}.translated.{target}{p.suffix}")
                out.write_bytes(content)
                content_size_bytes = len(content)

                # --- (신규) 완료 시 DB 저장 ---
                try:
                    original_rel_path = str(p.relative_to(_get_root())).replace("\\", "/")
                    original_doc_id = _get_original_doc_id(original_rel_path)
                    if original_doc_id is None:
                        print(f"⚠️ [SSE] 경고: 원본 문서를 DB에서 찾을 수 없습니다. (경로: {original_rel_path})")

                    translated_rel_path = str(out.relative_to(_get_root())).replace("\\", "/")
                    request_id = meta.get("requestId", req_id)
                    
                    _save_translation_record(
                        original_doc_id=original_doc_id,
                        target_language=target,
                        translated_rel_path=translated_rel_path,
                        request_id=request_id,
                        size_bytes=content_size_bytes
                    )
                except Exception as e:
                    print(f"❌ [SSE] 번역 후 DB 기록 작업 중 예기치 않은 오류 발생: {e}")
                # --- DB 저장 끝 ---

                yield _sse_event({
                    "type": "complete", "status": "COMPLETED", "percent": 100,
                    "requestId": req_id, "output_path": str(out),
                    "source_lang": source, "target_lang": target,
                })
                break # 루프 종료

            # 실패
            if status_raw in ("FAIL", "FAILED"):
                yield _sse_event({"type": "failed", "status": "FAILED", "percent": percent, "requestId": req_id})
                break # 루프 종료

            # 알 수 없는 상태 (일단 진행 중으로 처리)
            yield _sse_event({"type": "progress", "status": status_raw or "PROCESSING", "percent": percent})
            await asyncio.sleep(1.2)

    except Exception as e:
        yield _sse_event({"type": "failed", "status": "FAILED", "error": f"poll_error: {e}"})
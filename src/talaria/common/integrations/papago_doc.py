# src/talaria/common/integrations/papago_doc.py
import os, mimetypes, time, requests
from pathlib import Path

class PapagoDocClient:
    def __init__(self):
        key_id = os.getenv("PAPAGO_CLIENT_ID") or os.getenv("NCP_PAPAGO_KEY_ID")
        key    = os.getenv("PAPAGO_CLIENT_SECRET") or os.getenv("NCP_PAPAGO_KEY")
        if not key_id or not key:
            raise RuntimeError(
                "Papago 자격증명 누락: PAPAGO_CLIENT_ID/SECRET 또는 NCP_PAPAGO_KEY_ID/KEY"
            )

        self.h = {
            "X-NCP-APIGW-API-KEY-ID": key_id,
            "X-NCP-APIGW-API-KEY": key,
        }
        # 신 도메인 기본값(구 도메인 사용 금지)
        self.base = (os.getenv("PAPAGO_BASE") or "https://papago.apigw.ntruss.com").rstrip("/")

    def translate(self, file_path: Path, source: str, target: str, glossary_id: str | None = None) -> str:
        url = f"{self.base}/doc-trans/v1/translate"
        mime, _ = mimetypes.guess_type(str(file_path))
        with open(file_path, "rb") as fh:
            files = {"file": (file_path.name, fh, mime or "application/octet-stream")}
            data = {"source": source, "target": target}
            if glossary_id:
                data["glossaryId"] = glossary_id
            r = requests.post(url, headers=self.h, files=files, data=data, timeout=60)

        ct = r.headers.get("content-type", "")
        if r.status_code >= 400:
            raise RuntimeError(f"[translate] {r.status_code} {r.text}")

        try:
            js = r.json()
        except Exception:
            raise RuntimeError(f"[translate] non-JSON response: {r.status_code} {ct} {r.text[:400]}")

        # ✅ data 객체 안에서 requestId 찾기
        rid = js.get("data", {}).get("requestId")
        if not rid:
            raise RuntimeError(f"[translate] unexpected response: {r.status_code} {js}")
        return rid
    
    
    def get_status(self, request_id: str) -> dict:
            """
            문서 번역 상태를 1회 조회합니다. (SSE 폴링용)
            wait_until_done과 동일한 엔드포인트를 호출합니다.
            """
            if not request_id:
                raise ValueError("request_id가 필요합니다.")

            # ✅ 'self.base' (self.base_url 아님)
            # ✅ '.../status' (wait_until_done과 동일한 엔드포인트)
            url = f"{self.base}/doc-trans/v1/status"
            
            try:
                # ✅ 'self.h' (self.headers 아님)
                # ✅ 'params'로 requestId 전달 (wait_until_done과 동일)
                response = requests.get(
                    url, 
                    headers=self.h, 
                    params={"requestId": request_id}, 
                    timeout=10
                )
                
                response.raise_for_status() 
                js = response.json()
                
                # ✅ FastAPI 라우터가 일관되게 js.get("data", ...)를 사용할 수 있도록
                #    최상위 json을 그대로 반환합니다.
                return js

            except requests.exceptions.HTTPError as e:
                try:
                    error_data = e.response.json()
                except Exception:
                    error_data = e.response.text
                raise Exception(f"Papago 상태 조회 실패 (HTTP {e.response.status_code}): {error_data}") from e
            
            except Exception as e:
                raise Exception(f"Papago 상태 조회 중 네트워크 오류: {e}") from e
    
    def wait_until_done(self, request_id: str, poll_interval=2, timeout=600):
        url = f"{self.base}/doc-trans/v1/status"
        t0 = time.time()
        while True:
            r = requests.get(url, headers=self.h, params={"requestId": request_id}, timeout=30)
            if r.status_code >= 400:
                raise RuntimeError(f"[status] {r.status_code} {r.text}")
            js = r.json()
            
            print(f"[PAPAGO STATUS] {js}")
            
            # ✅ data 객체 안에서 status 가져오기
            data = js.get("data", {})
            st = data.get("status") or js.get("status")
            
            # ✅ COMPLETE (D 없음!)
            if st == "COMPLETE":  # COMPLETED 아님!
                return data if data else js
            if st == "FAIL":  # FAILED도 아닐 수 있음
                raise RuntimeError(f"[status] FAIL {js}")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"[status] Timeout waiting {request_id}")
            time.sleep(poll_interval)

    def download(self, request_id: str) -> bytes:
        url = f"{self.base}/doc-trans/v1/download"
        r = requests.get(url, headers=self.h, params={"requestId": request_id}, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"[download] {r.status_code} {r.text}")
        return r.content

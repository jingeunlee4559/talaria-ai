# src/talaria/livekit_agent/integrations.py
import os
import logging
import httpx
import asyncio
from typing import Optional, List, Dict

# (utils.py에서 import 필요)
from .utils import locale_to_papago_lang

log = logging.getLogger(__name__)

# Papago API 설정
PAPAGO_CLIENT_ID = os.getenv("PAPAGO_CLIENT_ID", "")
PAPAGO_CLIENT_SECRET = os.getenv("PAPAGO_CLIENT_SECRET", "")
PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

# hermes_service 주소 (포트 8000번)
HERMES_SERVICE_URL = os.getenv("HERMES_SERVICE_URL", "http://localhost:8000")

# 비동기 HTTP 클라이언트 (Agent 수명 주기 동안 공유)
http_client = httpx.AsyncClient(timeout=10.0)


async def translate_text_papago(text: str, source_lang="ko", target_lang="en") -> Optional[str]:
    """Papago API를 사용해 텍스트를 비동기적으로 번역합니다."""
    
    if not text or not text.strip():
        return None
    
    if not PAPAGO_CLIENT_ID or not PAPAGO_CLIENT_SECRET:
        return None
    
    headers = {
        "x-ncp-apigw-api-key-id": PAPAGO_CLIENT_ID,
        "x-ncp-apigw-api-key": PAPAGO_CLIENT_SECRET,
    }
    data = {
        "source": source_lang,
        "target": target_lang,
        "text": text,
        "honorific": True,
    }
    
    try:
        # (주의) Agent의 메인 http_client 대신 별도의 클라이언트를 사용합니다.
        # Agent 종료 시 http_client.aclose()가 호출되므로, 
        # Papago 호출이 많은 경우 별도 클라이언트가 안정적일 수 있습니다.
        # (또는 메인 http_client를 사용해도 무방합니다)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PAPAGO_URL, headers=headers, data=data)
            
            if response.status_code != 200:
                log.warning(f"❌ Papago API error ({source_lang}->{target_lang}): {response.status_code}")
                return None
            
            result_json = response.json()
            translated_text = result_json.get("message", {}).get("result", {}).get("translatedText")
            
            return translated_text
            
    except httpx.TimeoutException:
        log.warning(f"❌ Papago timeout ({source_lang}->{target_lang})")
        return None
    except Exception as e:
        log.error(f"❌ Papago error ({source_lang}->{target_lang}): {e}")
        return None


async def translate_to_multiple_languages(
    text: str, 
    source_locale: str,  # BCP47 형식 (예: "ja-JP")
    target_locales: List[str]  # BCP47 형식 리스트
) -> Dict[str, str]:
    """
    여러 언어로 동시 번역
    """
    if not text or not text.strip():
        return {}
    
    translations = {}
    
    source_code = locale_to_papago_lang(source_locale) # (utils.py 사용)
    
    # 번역 태스크 생성
    tasks = []
    task_locales = []
    
    for target_locale in target_locales:
        if target_locale == source_locale:
            continue
        
        target_code = locale_to_papago_lang(target_locale) # (utils.py 사용)
        
        if target_code == source_code:
            continue
        
        tasks.append(translate_text_papago(text, source_code, target_code))
        task_locales.append(target_locale)
    
    if not tasks:
        return {}
    
    # 병렬 실행
    log.info(f"🌍 번역 시작: {source_locale} ({source_code}) -> {task_locales}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 수집
    for target_locale, result in zip(task_locales, results):
        if isinstance(result, str) and result:
            translations[target_locale] = result
            log.info(f"✅ 번역 성공: {target_locale}: {result[:30]}...")
        elif isinstance(result, Exception):
            log.warning(f"❌ 번역 실패: {target_locale}: {result}")
    
    return translations
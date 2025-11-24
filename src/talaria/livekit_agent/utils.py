# src/talaria/livekit_agent/utils.py
import os
import logging
from livekit.plugins import silero, clova as clova_stt
from livekit.agents.stt import StreamAdapter

log = logging.getLogger(__name__)

# 🔥 Locale <-> 언어 코드 매핑 (Clova 형식에 정확히 맞춤!)
def locale_to_clova_lang(locale: str) -> str:
    """
    BCP47 locale을 Clova STT 언어 코드로 변환
    
    Clova 지원 형식:
    - ko-KR (하이픈 포함)
    - ja (일본어만 하이픈 없음)
    - en-US (하이픈 포함)
    - enko (한영 혼용)
    - zh-cn (소문자!)
    - zh-tw (소문자!)
    """
    # 수정 사항
    if locale in ["ko-KR", "en-US", "ko", "en"]:
        return "enko"

    mapping = {
        "ko-KR": "ko-KR",      # ✅ 하이픈 포함
        "en-US": "en-US",      # ✅ 하이픈 포함
        "ja-JP": "ja",         # ✅ 일본어만 하이픈 없음
        "zh-CN": "zh-cn",      # ✅ 소문자
        "zh-TW": "zh-tw",      # ✅ 소문자
    }
    
    result = mapping.get(locale, "enko")  # 기본값: 한영 혼용
    log.info(f"🔤 Locale mapping: {locale} -> {result}")
    return result


def locale_to_papago_lang(locale: str) -> str:
    """BCP47 locale을 Papago 언어 코드로 변환"""
    mapping = {
        "ko-KR": "ko",
        "en-US": "en",
        "ja-JP": "ja",
        "zh-CN": "zh-CN",
        "zh-TW": "zh-TW",
        "vi-VN": "vi",
        "th-TH": "th",
        "id-ID": "id",
        "fr-FR": "fr",
        "es-ES": "es",
        "ru-RU": "ru",
        "de-DE": "de",
        "it-IT": "it",
    }
    return mapping.get(locale, locale.split("-")[0] if "-" in locale else locale)


def to_bcp47(lang: str) -> str:
    """언어 코드를 BCP47 형식으로 변환"""
    mapping = {
        "ko": "ko-KR",
        "ko-KR": "ko-KR",
        "en": "en-US",
        "en-US": "en-US",
        "ja": "ja-JP",
        "zh-CN": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-TW": "zh-TW",
        "zh-tw": "zh-TW",
        "vi": "vi-VN",
        "th": "th-TH",
        "id": "id-ID",
        "fr": "fr-FR",
        "es": "es-ES",
        "ru": "ru-RU",
        "de": "de-DE",
        "it": "it-IT",
    }
    return mapping.get(lang, f"{lang}-{lang.upper()}" if "-" not in lang else lang)


# (temp.py의 entrypoint 함수에서 이동)
def create_stt_for_locale(locale: str):
    """참가자의 locale에 맞는 STT 모델 생성"""
    clova_lang = locale_to_clova_lang(locale)
    
    try:
        log.info(f"🔧 Creating Clova STT: locale={locale}, clova_lang={clova_lang}")
        base_stt = clova_stt.STT(language=clova_lang)
        stt_model = StreamAdapter(
            stt=base_stt,
            vad=silero.VAD.load(),
        )
        log.info(f"✅ Created STT for locale={locale}, clova_lang={clova_lang}")
        return stt_model
    except Exception as e:
        log.error(f"❌ Failed to create STT for {locale}: {e}")
        log.exception("Full traceback:")
        
        # 폴백: 기본 언어 사용
        default_lang = os.getenv("CLOVA_LANG", "enko")
        log.warning(f"⚠️ Falling back to default: {default_lang}")
        base_stt = clova_stt.STT(language=default_lang)
        stt_model = StreamAdapter(
            stt=base_stt,
            vad=silero.VAD.load(),
        )
        return stt_model
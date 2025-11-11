import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR  = os.path.dirname(_THIS_FILE)
_SRC_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "../../"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import json
import asyncio
import logging
import time
from typing import Optional, Any, List, Dict, Set

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, WorkerOptions, cli
from livekit.plugins import silero
from livekit.agents.stt import StreamAdapter
from livekit.plugins import clova as clova_stt
from talaria.livekit_agent.manner_checker import MannerChecker

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Papago API 설정
PAPAGO_CLIENT_ID = os.getenv("PAPAGO_CLIENT_ID", "")
PAPAGO_CLIENT_SECRET = os.getenv("PAPAGO_CLIENT_SECRET", "")
PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

print("=" * 80)
print("🚀 AGENT STARTING WITH MULTI-LANGUAGE SUPPORT")
print(f"📝 PAPAGO_CLIENT_ID: {'✅ SET' if PAPAGO_CLIENT_ID else '❌ NOT SET'}")
print(f"📝 PAPAGO_CLIENT_SECRET: {'✅ SET' if PAPAGO_CLIENT_SECRET else '❌ NOT SET'}")
print(f"📝 PAPAGO_ENABLED: {os.getenv('PAPAGO_ENABLED', 'NOT SET')}")
print("=" * 80)


class STTAgent(Agent):
    def __init__(self, room):
        super().__init__(instructions="Multi-user Multi-language STT agent with Clova + MannerChecker + Papago")
        self.room = room


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
    mapping = {
        "ko-KR": "ko-KR",      # ✅ 하이픈 포함
        "en-US": "en-US",      # ✅ 하이픈 포함
        "ja-JP": "ja",         # ✅ 일본어만 하이픈 없음
        "zh-CN": "zh-cn",      # ✅ 소문자
        "zh-TW": "zh-tw",      # ✅ 소문자
    }
    
    result = mapping.get(locale, "enko")  # 기본값: 한영 혼용
    logging.info(f"🔤 Locale mapping: {locale} -> {result}")
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PAPAGO_URL, headers=headers, data=data)
            
            if response.status_code != 200:
                logging.warning(f"❌ Papago API error ({source_lang}->{target_lang}): {response.status_code}")
                return None
            
            result_json = response.json()
            translated_text = result_json.get("message", {}).get("result", {}).get("translatedText")
            
            return translated_text
            
    except httpx.TimeoutException:
        logging.warning(f"❌ Papago timeout ({source_lang}->{target_lang})")
        return None
    except Exception as e:
        logging.error(f"❌ Papago error ({source_lang}->{target_lang}): {e}")
        return None


async def translate_to_multiple_languages(
    text: str, 
    source_locale: str,  # BCP47 형식 (예: "ja-JP")
    target_locales: List[str]  # BCP47 형식 리스트
) -> Dict[str, str]:
    """
    여러 언어로 동시 번역
    
    Args:
        text: 번역할 원본 텍스트
        source_locale: 원본 locale (BCP47)
        target_locales: 타겟 locale 리스트 (BCP47)
    
    Returns:
        {locale: translated_text} 형태의 딕셔너리
    """
    if not text or not text.strip():
        return {}
    
    translations = {}
    
    source_code = locale_to_papago_lang(source_locale)
    
    # 번역 태스크 생성
    tasks = []
    task_locales = []
    
    for target_locale in target_locales:
        # 원본 언어와 동일하면 스킵
        if target_locale == source_locale:
            continue
        
        target_code = locale_to_papago_lang(target_locale)
        
        # 같은 언어 코드면 스킵 (ko-KR와 ko는 같은 것으로 취급)
        if target_code == source_code:
            continue
        
        tasks.append(translate_text_papago(text, source_code, target_code))
        task_locales.append(target_locale)
    
    if not tasks:
        return {}
    
    # 병렬 실행
    logging.info(f"🌍 번역 시작: {source_locale} ({source_code}) -> {task_locales}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 수집
    for target_locale, result in zip(task_locales, results):
        if isinstance(result, str) and result:
            translations[target_locale] = result
            logging.info(f"✅ 번역 성공: {target_locale}: {result[:30]}...")
        elif isinstance(result, Exception):
            logging.warning(f"❌ 번역 실패: {target_locale}: {result}")
    
    return translations


class ParticipantSTTHandler:
    """Handle realtime STT for a participant's audio track."""

    def __init__(
        self,
        participant,
        track,
        participant_locale: str,  # 🔥 참가자의 locale
        stt_model,  # 🔥 참가자별 STT 모델
        room,
        publish_callback,
        *,
        manner_checker: Optional[MannerChecker] = None,
        publish_manner_callback=None,
        translation_enabled: bool = False,
        supported_locales: Set[str] = None,  # 🔥 회의의 모든 locale
    ):
        self.participant = participant
        self.track = track
        self.participant_locale = participant_locale
        self.stt = stt_model
        self.room = room
        self.publish_callback = publish_callback

        # ✅ Manner
        self.mchecker = manner_checker
        self.publish_manner_callback = publish_manner_callback

        # ✅ Translation
        self.translation_enabled = translation_enabled
        self.supported_locales = supported_locales or set()
        
        clova_lang = locale_to_clova_lang(participant_locale)
        logging.info(f"🎯 ParticipantSTTHandler: locale={participant_locale}, clova_lang={clova_lang}, translation={translation_enabled}")

        self.stream = None
        self.task = None
        self._is_running = True

    async def start(self):
        audio_stream = rtc.AudioStream(self.track)
        pid = getattr(self.participant, "identity", "unknown")
        try:
            while self._is_running:
                logging.info(f"[{pid}] 🔄 Starting new STT stream...")
                self.stream = self.stt.stream()
                self.task = asyncio.create_task(self._process_stream())

                try:
                    async for event in audio_stream:
                        if not self._is_running:
                            break
                        if self.stream:
                            self.stream.push_frame(event.frame)
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "is closed" in msg:
                        logging.warning(f"[{pid}] ⚠️ Stream closed (likely silence). Restarting...")
                    else:
                        logging.error(f"[{pid}] Audio push runtime error: {msg}", exc_info=True)
                        raise

                await self._cleanup_stream_task()
                if not self._is_running:
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logging.info(f"[{pid}] STT handler task cancelled")
        except Exception as e:
            logging.error(f"[{pid}] STT handler failed catastrophically: {e}", exc_info=True)
        finally:
            logging.info(f"[{pid}] 🛑 Cleaning up STT handler...")
            self._is_running = False
            await self._cleanup_stream_task()
            await audio_stream.aclose()

    async def _process_stream(self):
        pid = getattr(self.participant, "identity", "unknown")
        try:
            async for event in self.stream:
                if not self._is_running:
                    break
                await self._handle_transcription(event)
        except StopAsyncIteration:
            logging.info(f"[{pid}] STT stream ended normally")
        except Exception as e:
            error_msg = str(e)
            try:
                if "Unexpected response" in error_msg and "'text': ''" in error_msg:
                    logging.debug(f"[{pid}] 💤 Clova: empty response (silence)")
                elif "Confidence" in error_msg and "bellow threshold" in error_msg:
                    logging.debug(f"[{pid}] ⚠️ low-confidence result ignored: {error_msg}")
                else:
                    logging.error(f"[{pid}] ❌ STT stream error: {error_msg}")
            except Exception as log_error:
                logging.error(f"[{pid}] 🚨 error while logging: {log_error}")

    def _get_current_locales(self) -> Set[str]:
        """(실시간) 현재 회의의 모든 참가자 locale (지원 언어만) 반환"""
        locales = set()
        
        # 로컬 참가자 (에이전트 자신 - 보통 metadata가 없음)
        if self.room.local_participant and self.room.local_participant.metadata:
            try:
                meta = json.loads(self.room.local_participant.metadata)
                if "appLocale" in meta:
                    locales.add(meta["appLocale"])
            except:
                pass
        
        # 원격 참가자들 (실제 사용자)
        for participant in self.room.remote_participants.values():
            if participant.metadata:
                try:
                    meta = json.loads(participant.metadata)
                    if "appLocale" in meta:
                        locales.add(meta["appLocale"])
                except:
                    pass
        
        # 지원하는 locale만 필터링
        return locales.intersection(self.supported_locales)

    async def _handle_transcription(self, event):
        pid = getattr(self.participant, "identity", "unknown")
        try:
            event_type = getattr(event, 'type', '').lower()
            is_final = 'final' in event_type and 'interim' not in event_type

            text = ""
            if hasattr(event, 'alternatives') and event.alternatives:
                text = event.alternatives[0].text
            elif hasattr(event, 'text'):
                text = event.text

            if not text or not text.strip():
                logging.debug(f"[{pid}] Empty text skipped")
                return

            username = self._get_username()
            
            # 🔥 다국어 번역: 자신의 locale을 제외한 모든 locale로 번역
            translations = {}
            
            if is_final and self.translation_enabled:
               # 🔥 (핵심 수정) 번역 시점의 최신 locale 목록을 가져옴
                current_locales = self._get_current_locales()
                
                # 번역 대상: 자신을 제외한 모든 참가자의 locale
                target_locales = [loc for loc in current_locales if loc != self.participant_locale]
                
                if target_locales:
                    try:
                        translations = await translate_to_multiple_languages(
                            text, 
                            self.participant_locale,  # 원본 언어
                            target_locales  # 타겟 언어들
                        )
                    except Exception as trans_err:
                        logging.error(f"❌ 번역 중 예외: {trans_err}", exc_info=True)

            # 1) publish STT
            await self.publish_callback(
                text=text,
                final=is_final,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                locale=self.participant_locale,  # 🔥 원본 locale
                translations=translations,
            )

            # 2) Manner check only for FINAL
            if is_final and self.mchecker and self.publish_manner_callback:
                try:
                    clova_lang = locale_to_clova_lang(self.participant_locale)
                    alerts = await self.mchecker.process_text(
                        text=text,
                        room_name=self.room.name,
                        speaker_identity=self.participant.identity,
                        speaker_username=username,
                        language=clova_lang,
                    )
                    for payload in alerts:
                        await self.publish_manner_callback(payload)
                except Exception:
                    logging.exception(f"[{pid}] manner checking failed")
        except Exception as e:
            logging.error(f"[{pid}] 🧩 transcription handling error: {e}", exc_info=True)

    def _get_username(self) -> str:
        try:
            metadata = json.loads(self.participant.metadata or '{}')
            if "username" in metadata:
                return metadata["username"]
        except Exception:
            pass
        return self.participant.name or self.participant.identity

    async def _cleanup_stream_task(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        if self.stream:
            try:
                await self.stream.aclose()
            except Exception:
                pass
            self.stream = None

    async def stop(self):
        pid = getattr(self.participant, "identity", "unknown")
        logging.info(f"[{pid}] Received stop signal. Cleaning up...")
        self._is_running = False
        await self._cleanup_stream_task()
        logging.info(f"[{pid}] STT handler stopped cleanly")


async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)

    # 🔥 지원 언어 목록
    supported_locales_str = os.getenv("SUPPORTED_LOCALES", "ko-KR,en-US,ja-JP,zh-CN,zh-TW")
    supported_locales = set(loc.strip() for loc in supported_locales_str.split(",") if loc.strip())
    
    logging.info(f"🌍 지원 언어: {supported_locales}")

    # MannerChecker init
    base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")
    api_token = os.getenv("BACKEND_API_TOKEN", None)
    refresh_sec = int(os.getenv("MANNER_RULES_REFRESH_SEC", "30"))
    agent_codes = [x.strip() for x in os.getenv("MANNER_AGENT_CODES", "direct,address,tone,negative").split(",") if x.strip()]
    app_locale = os.getenv("APP_LOCALE", "ko")
    target_locale = os.getenv("TARGET_LOCALE", "") or None
    culture_pair = None
    if app_locale and target_locale:
        culture_pair = f"{app_locale.split('-')[0]}->{target_locale.split('-')[0]}"

    mchecker = MannerChecker(
        base_url=base_url,
        api_token=api_token,
        refresh_sec=refresh_sec,
        agent_codes=agent_codes,
        app_locale_prefix=app_locale,
        target_locale_prefix=target_locale,
        culture_pair=culture_pair,
    )
    await mchecker.start()

    # Translation settings
    translation_enabled = os.getenv("PAPAGO_ENABLED", "false").lower() == "true"
    
    print("=" * 80)
    print("🌐 Multi-language Translation settings:")
    print(f"   Enabled: {translation_enabled}")
    print(f"   Supported Locales: {supported_locales}")
    print(f"   Papago Client ID: {'✅ Set' if PAPAGO_CLIENT_ID else '❌ Not set'}")
    print(f"   Papago Client Secret: {'✅ Set' if PAPAGO_CLIENT_SECRET else '❌ Not set'}")
    print("=" * 80)

    handlers: Dict[str, ParticipantSTTHandler] = {}

    # 🔥 회의 참가자들의 locale 추적
    def get_all_participant_locales() -> Set[str]:
        """현재 회의의 모든 참가자 locale 반환"""
        locales = set()
        
        # 로컬 참가자
        if ctx.room.local_participant and ctx.room.local_participant.metadata:
            try:
                meta = json.loads(ctx.room.local_participant.metadata)
                if "appLocale" in meta:
                    locales.add(meta["appLocale"])
            except:
                pass
        
        # 원격 참가자들
        for participant in ctx.room.remote_participants.values():
            if participant.metadata:
                try:
                    meta = json.loads(participant.metadata)
                    if "appLocale" in meta:
                        locales.add(meta["appLocale"])
                except:
                    pass
        
        # 지원하는 locale만 필터링
        return locales.intersection(supported_locales)

    # DataChannel publishers
    async def _publish_stt(
        text: str, 
        final: bool, 
        identity: str, 
        psid: str, 
        username: str,
        locale: str,  # 🔥 원본 locale
        translations: Dict[str, str] = None
    ):
        payload = {
            "type": "stt",
            "final": final,
            "text": text or "",
            "identity": identity,
            "username": username,
            "participant_sid": psid,
            "ts": time.time(),
            "locale": locale,  # 🔥 원본 locale
            "translations": translations or {},
        }
        
        log_type = "FINAL" if final else "PARTIAL"
        logging.info(f"📤 STT [{log_type}] {username} ({locale}): {text[:50]}...")
        if translations:
            logging.info(f"🌍 TRANSLATIONS: {translations}")
        
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic="stt",
            )
        except Exception as e:
            logging.exception(f"❌ publish_data failed: {e}")

    async def _publish_manner(payload: dict):
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic="manner",
            )
            logging.info(f"✅ Published MANNER: rule={payload.get('rule_id')} match={payload.get('match')}")
        except Exception as e:
            logging.exception(f"❌ publish manner failed: {e}")

    # 🔥 참가자별 STT 모델 생성
    def create_stt_for_locale(locale: str):
        """참가자의 locale에 맞는 STT 모델 생성"""
        clova_lang = locale_to_clova_lang(locale)
        
        try:
            logging.info(f"🔧 Creating Clova STT: locale={locale}, clova_lang={clova_lang}")
            base_stt = clova_stt.STT(language=clova_lang)
            stt_model = StreamAdapter(
                stt=base_stt,
                vad=silero.VAD.load(),
            )
            logging.info(f"✅ Created STT for locale={locale}, clova_lang={clova_lang}")
            return stt_model
        except Exception as e:
            logging.error(f"❌ Failed to create STT for {locale}: {e}")
            logging.exception("Full traceback:")
            
            # 폴백: 기본 언어 사용
            default_lang = os.getenv("CLOVA_LANG", "enko")
            logging.warning(f"⚠️ Falling back to default: {default_lang}")
            base_stt = clova_stt.STT(language=default_lang)
            stt_model = StreamAdapter(
                stt=base_stt,
                vad=silero.VAD.load(),
            )
            return stt_model

    # Room event handlers
    def _on_track_published(publication, participant):
        asyncio.create_task(_handle_track_published(publication, participant))

    async def _handle_track_published(publication, participant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return
        handler_key = f"{participant.sid}_{publication.sid}"
        if handler_key in handlers:
            logging.info(f"[{participant.identity}] Audio track already being processed")
            return
        
        logging.info(f"🎤 [{participant.identity}] Audio track published: {publication.sid}")
        
        track = publication.track
        if not track:
            logging.info(f"[{participant.identity}] Waiting for track to be available...")
            await asyncio.sleep(0.5)
            track = publication.track
        if not track:
            logging.warning(f"[{participant.identity}] Track still not available, skipping")
            return
        
        # 🔥 참가자의 locale 가져오기
        participant_locale = "ko-KR"  # 기본값
        try:
            if participant.metadata:
                meta = json.loads(participant.metadata)
                if "appLocale" in meta:
                    participant_locale = meta["appLocale"]
                    logging.info(f"✅ Detected participant locale: {participant_locale}")
        except Exception as e:
            logging.warning(f"⚠️ Could not parse metadata, using default locale: {e}")
        
        # 🔥 참가자의 locale에 맞는 STT 생성
        stt_model = create_stt_for_locale(participant_locale)
        
        # 🔥 현재 회의의 모든 locale
        all_locales = get_all_participant_locales()
        logging.info(f"🌍 Current meeting locales: {all_locales}")
        
        handler = ParticipantSTTHandler(
            participant=participant,
            track=track,
            participant_locale=participant_locale,  # 🔥 참가자 locale
            stt_model=stt_model,  # 🔥 참가자별 STT
            room=ctx.room,
            publish_callback=_publish_stt,
            manner_checker=mchecker,
            publish_manner_callback=_publish_manner,
            translation_enabled=translation_enabled,
            supported_locales=supported_locales,
        )
        handlers[handler_key] = handler
        asyncio.create_task(handler.start())

    def _on_track_unpublished(publication, participant):
        asyncio.create_task(_handle_track_unpublished(publication, participant))

    async def _handle_track_unpublished(publication, participant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return
        handler_key = f"{participant.sid}_{publication.sid}"
        handler = handlers.pop(handler_key, None)
        if handler:
            logging.info(f"🔇 [{participant.identity}] Audio track unpublished: {publication.sid}")
            await handler.stop()

    # Register listeners
    ctx.room.on("track_published", _on_track_published)
    ctx.room.on("track_unpublished", _on_track_unpublished)

    logging.info("🎙️ Multi-language STT Agent starting...")

    # Connect room
    try:
        await ctx.connect()
        logging.info("✅ Connected to room")
    except Exception as e:
        logging.exception(f"❌ Failed to connect to room: {e}")
        await mchecker.close()
        return

    # Sweep pre-existing tracks
    await asyncio.sleep(1)
    logging.info("🔍 Processing existing tracks...")
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.track:
                logging.info(f"🔌 Found existing audio track from {participant.identity}")
                await _handle_track_published(publication, participant)

    logging.info("🎙️ Multi-language STT Agent ready (Clova + MannerChecker + Papago)")

    # Keep alive
    try:
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.warning("[JobContext] cancelled by dispatcher")
    except Exception as e:
        logging.exception(f"[entrypoint] error: {e}")
    finally:
        for handler in list(handlers.values()):
            await handler.stop()
        await mchecker.close()
        logging.info("🛑 STT Agent stopped")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
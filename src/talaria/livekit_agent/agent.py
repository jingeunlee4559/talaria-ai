# 문맥기반 번역까지 구현.

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
import datetime
from typing import Optional, Any, List, Dict, Set
# import httpx # (integrations.py로 이동)

from dotenv import load_dotenv
load_dotenv()

from livekit import agents, rtc
from livekit.agents import Agent, WorkerOptions, cli
from talaria.livekit_agent.manner_checker import MannerChecker

# (신규) 분리된 모듈에서 import
from .handler import ParticipantSTTHandler
from .utils import create_stt_for_locale, locale_to_clova_lang
from .integrations import http_client, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- (integrations.py로 이동) ---
# Papago API 설정
# hermes_service 주소
# 비동기 HTTP 클라이언트

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
        # ts = datetime.datetime.now().timestamp()
        ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
        payload = {
            "type": "stt",
            "final": final,
            "text": text or "",
            "identity": identity,
            "username": username,
            "participant_sid": psid,
            "ts": ts,
            "locale": locale,  # 🔥 원본 locale
            "translations": translations or {},
        }
        
        log_type = "FINAL" if final else "PARTIAL"
        logging.info(f"📤 STT [{log_type}], TIME[{ts}] {username} ({locale}): {text[:50]}...")
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
        
        # 🔥 참가자의 locale에 맞는 STT 생성 (utils.py 사용)
        stt_model = create_stt_for_locale(participant_locale)
        
        # 🔥 현재 회의의 모든 locale
        all_locales = get_all_participant_locales()
        logging.info(f"🌍 Current meeting locales: {all_locales}")
        
        # (신규) handler.py의 ParticipantSTTHandler 사용
        handler = ParticipantSTTHandler(
            participant=participant,
            track=track,
            participant_locale=participant_locale,
            stt_model=stt_model,
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
        await http_client.aclose() # (신규) http_client 종료
        logging.info("🛑 STT Agent stopped")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
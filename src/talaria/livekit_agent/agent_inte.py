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
from typing import Optional, Any, List, Dict

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
print("🚀 AGENT STARTING WITH TRANSLATION")
print(f"📝 PAPAGO_CLIENT_ID: {'✅ SET' if PAPAGO_CLIENT_ID else '❌ NOT SET'}")
print(f"📝 PAPAGO_CLIENT_SECRET: {'✅ SET' if PAPAGO_CLIENT_SECRET else '❌ NOT SET'}")
print(f"📝 PAPAGO_ENABLED: {os.getenv('PAPAGO_ENABLED', 'NOT SET')}")
print("=" * 80)


class STTAgent(Agent):
    def __init__(self, room):
        super().__init__(instructions="Multi-user STT agent with Clova + MannerChecker + Papago")
        self.room = room


async def translate_text_papago(text: str, source_lang="ko", target_lang="en") -> Optional[str]:
    """Papago API를 사용해 텍스트를 비동기적으로 번역합니다."""
    print(f"🔥 translate_text_papago 호출됨! text={text[:30]}, source={source_lang}, target={target_lang}")
    
    if not text:
        print("⚠️ Papago: 빈 텍스트")
        return None
    
    if not PAPAGO_CLIENT_ID:
        print("❌ Papago: CLIENT_ID가 설정되지 않음")
        return None
    
    if not PAPAGO_CLIENT_SECRET:
        print("❌ Papago: CLIENT_SECRET이 설정되지 않음")
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
        print(f"🌐 Papago API 호출 중...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PAPAGO_URL, headers=headers, data=data)
            
            print(f"📡 Papago 응답: status={response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Papago API 오류: {response.text[:200]}")
                return None
            
            result_json = response.json()
            translated_text = result_json.get("message", {}).get("result", {}).get("translatedText")
            
            if translated_text:
                print(f"✅ Papago 번역 성공: {translated_text[:50]}")
            else:
                print(f"⚠️ Papago 응답에 번역 결과 없음: {result_json}")
            
            return translated_text
            
    except httpx.TimeoutException:
        print("❌ Papago 번역 타임아웃 (10초)")
        return None
    except Exception as e:
        print(f"❌ Papago 번역 실패: {e}")
        logging.error(f"Papago error details:", exc_info=True)
        return None


class ParticipantSTTHandler:
    """Handle realtime STT for a participant's audio track."""

    def __init__(
        self,
        participant,
        track,
        stt_model,
        room,
        publish_callback,
        *,
        manner_checker: Optional[MannerChecker] = None,
        publish_manner_callback=None,
        stt_language: str = "enko",
        translation_enabled: bool = False,
        translation_source: str = "ko",
        translation_target: str = "en",
    ):
        self.participant = participant
        self.track = track
        self.stt = stt_model
        self.room = room
        self.publish_callback = publish_callback

        # ✅ Manner
        self.mchecker = manner_checker
        self.publish_manner_callback = publish_manner_callback
        self.stt_language = stt_language

        # ✅ Translation
        self.translation_enabled = translation_enabled
        self.translation_source = translation_source
        self.translation_target = translation_target
        
        print(f"🎯 ParticipantSTTHandler 생성: translation_enabled={translation_enabled}")

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
            
            print(f"=" * 80)
            print(f"🎤 STT 받음: username={username}, is_final={is_final}, text={text[:50]}")
            print(f"🔍 translation_enabled={self.translation_enabled}")

            # ✅ Translation (FINAL only)
            translated_text: Optional[str] = None
            
            if is_final:
                print(f"✅ FINAL 텍스트 감지!")
                if self.translation_enabled:
                    print(f"✅ 번역 활성화됨! 번역 시작...")
                    try:
                        translated_text = await translate_text_papago(
                            text, 
                            source_lang=self.translation_source, 
                            target_lang=self.translation_target
                        )
                        print(f"🎉 번역 완료: {translated_text}")
                    except Exception as trans_err:
                        print(f"❌ 번역 중 예외: {trans_err}")
                        logging.error(f"Translation error:", exc_info=True)
                else:
                    print(f"❌ 번역 비활성화됨 (translation_enabled={self.translation_enabled})")
            else:
                print(f"⏭️ PARTIAL 텍스트라서 번역 스킵")
            
            print(f"=" * 80)

            # 1) publish STT
            await self.publish_callback(
                text=text,
                final=is_final,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                translated=translated_text,
            )

            # 2) Manner check only for FINAL
            if is_final and self.mchecker and self.publish_manner_callback:
                try:
                    alerts = await self.mchecker.process_text(
                        text=text,
                        room_name=self.room.name,
                        speaker_identity=self.participant.identity,
                        speaker_username=username,
                        language=self.stt_language,
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

    # STT init (Clova + StreamAdapter + Silero VAD)
    try:
        base_stt = clova_stt.STT(
            language=os.getenv("CLOVA_LANG", "enko"),
        )
        stt_model = StreamAdapter(
            stt=base_stt,
            vad=silero.VAD.load(),
        )
        logging.info("✅ STT provider = CLOVA (with StreamAdapter)")
        logging.info(f"   Language: {os.getenv('CLOVA_LANG', 'enko')}")
    except Exception as e:
        logging.exception(f"❌ Failed to initialize Clova STT: {e}")
        return

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
    translation_source = os.getenv("PAPAGO_SOURCE_LANG", "ko")
    translation_target = os.getenv("PAPAGO_TARGET_LANG", "en")
    
    print("=" * 80)
    print("🌐 Translation settings:")
    print(f"   Enabled: {translation_enabled}")
    print(f"   Source: {translation_source} -> Target: {translation_target}")
    print(f"   Papago Client ID: {'✅ Set' if PAPAGO_CLIENT_ID else '❌ Not set'}")
    print(f"   Papago Client Secret: {'✅ Set' if PAPAGO_CLIENT_SECRET else '❌ Not set'}")
    print("=" * 80)

    handlers: Dict[str, ParticipantSTTHandler] = {}

    # DataChannel publishers
    async def _publish_stt(text: str, final: bool, identity: str, psid: str, username: str, translated: Optional[str] = None):
        payload = {
            "type": "stt",
            "final": final,
            "text": text or "",
            "identity": identity,
            "username": username,
            "participant_sid": psid,
            "ts": time.time(),
            "translated": translated,
        }
        log_type = "FINAL" if final else "PARTIAL"
        logging.info(f"📤 STT [{log_type}] {username}: {text[:50]}...")
        if translated:
            logging.info(f"🌍 TRANSLATED: {translated[:50]}...")
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
        handler = ParticipantSTTHandler(
            participant=participant,
            track=track,
            stt_model=stt_model,
            room=ctx.room,
            publish_callback=_publish_stt,
            manner_checker=mchecker,
            publish_manner_callback=_publish_manner,
            stt_language=os.getenv("CLOVA_LANG", "enko"),
            translation_enabled=translation_enabled,
            translation_source=translation_source,
            translation_target=translation_target,
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

    logging.info("🎙️ Multi-user STT Agent starting...")

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

    logging.info("🎙️ Multi-user STT Agent ready (Clova + MannerChecker + Papago)")

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
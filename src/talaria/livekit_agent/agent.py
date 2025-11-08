
import os
import sys

_THIS_FILE = os.path.abspath(__file__)                              # .../src/talaria/livekit_agent/agent.py
_THIS_DIR  = os.path.dirname(_THIS_FILE)                            # .../src/talaria/livekit_agent
_SRC_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "../../"))     # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import json
import asyncio
import logging
import time
from typing import Optional, Any, List, Dict

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, WorkerOptions, cli
from livekit.plugins import silero
from livekit.agents.stt import StreamAdapter
from livekit.plugins import clova as clova_stt
from talaria.livekit_agent.manner_checker import MannerChecker

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class STTAgent(Agent):
    def __init__(self, room):
        super().__init__(instructions="Multi-user STT agent with Clova + MannerChecker")
        self.room = room


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
        stt_language: str = "ko-KR",
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

        self.stream = None
        self.task = None
        self._is_running = True

    async def start(self):
        audio_stream = rtc.AudioStream(self.track)
        pid = getattr(self.participant, "identity", "unknown")
        try:
            while self._is_running:
                logging.info(f"[{pid}] 🔄 Starting new STT stream...")
                # 1) open a new STT stream
                self.stream = self.stt.stream()
                self.task = asyncio.create_task(self._process_stream())

                try:
                    # 2) forward audio frames
                    async for event in audio_stream:
                        if not self._is_running:
                            break
                        if self.stream:
                            self.stream.push_frame(event.frame)
                    # track unpublished => exit loop
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

            # 1) publish STT
            await self.publish_callback(
                text=text,
                final=is_final,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username
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


# ---------------------------------------------------------------------------
# Job entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)

    # STT init (Clova + StreamAdapter + Silero VAD)
    try:
        base_stt = clova_stt.STT(
            language=os.getenv("CLOVA_LANG", "ko-KR"),
        )
        stt_model = StreamAdapter(
            stt=base_stt,
            vad=silero.VAD.load(
                # Optional tuning:
                # min_speech_duration=0.5,
                # min_silence_duration=0.8,
            ),
        )
        logging.info("✅ STT provider = CLOVA (with StreamAdapter)")
        logging.info(f"   Language: {os.getenv('CLOVA_LANG', 'ko-KR')}")
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

    handlers: Dict[str, ParticipantSTTHandler] = {}

    # DataChannel publishers
    async def _publish_stt(text: str, final: bool, identity: str, psid: str, username: str):
        payload = {
            "type": "stt",
            "final": final,
            "text": text or "",
            "identity": identity,
            "username": username,
            "participant_sid": psid,
            "ts": time.time(),
        }
        log_type = "FINAL" if final else "PARTIAL"
        logging.info(f"📤 STT [{log_type}] {username}: {text[:50]}...")
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

    # Room event handlers (sync -> create_task)
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
            stt_language=os.getenv("CLOVA_LANG", "ko-KR"),
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
                logging.info(f"📌 Found existing audio track from {participant.identity}")
                await _handle_track_published(publication, participant)

    logging.info("🎙️ Multi-user STT Agent ready (Clova + MannerChecker)")

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

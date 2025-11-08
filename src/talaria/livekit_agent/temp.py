import os, json, sys, asyncio, logging, time
from typing import Optional, Dict, Set
import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, WorkerOptions, cli
from livekit.plugins import silero
from livekit.agents.stt import StreamAdapter
from livekit.plugins import clova as clova_stt

load_dotenv()

PAPAGO_CLIENT_ID = os.getenv("PAPAGO_CLIENT_ID")
PAPAGO_CLIENT_SECRET = os.getenv("PAPAGO_CLIENT_SECRET")
PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🌍 지원 언어 매핑 (Papago 언어 코드)
LOCALE_TO_PAPAGO = {
    "ko": "ko",  # 한국어
    "en": "en",  # 영어
    "ja": "ja",  # 일본어
    "zh": "zh-CN",  # 중국어 간체
    "zh-TW": "zh-TW",  # 중국어 번체
    "es": "es",  # 스페인어
    "fr": "fr",  # 프랑스어
    "de": "de",  # 독일어
    "ru": "ru",  # 러시아어
    "pt": "pt",  # 포르투갈어
    "it": "it",  # 이탈리아어
    "vi": "vi",  # 베트남어
    "th": "th",  # 태국어
    "id": "id",  # 인도네시아어
    "hi": "hi",  # 힌디어
}

class STTAgent(Agent):
    def __init__(self, room):
        super().__init__(instructions="Multi-user multilingual STT agent")
        self.room = room


class ParticipantSTTHandler:
    """참가자별 실시간 음성(STT) 처리 핸들러"""

    def __init__(self, participant, track, stt_model, room, publish_callback):
        self.participant = participant
        self.track = track
        self.stt = stt_model
        self.room = room
        self.publish_callback = publish_callback
        self.stream = None
        self.task = None
        self._is_running = True

    async def start(self):
        audio_stream = rtc.AudioStream(self.track)

        try:
            while self._is_running:
                pid = getattr(self.participant, "identity", "unknown")
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
                        logging.warning(f"[{pid}] ⚠️ Stream closed. Restarting...")
                    else:
                        logging.error(f"[{pid}] Audio push error: {msg}", exc_info=True)
                        raise

                await self._cleanup_stream_task()
                if not self._is_running:
                    break
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logging.info(f"[{self.participant.identity}] Task cancelled")
        except Exception as e:
            logging.error(f"[{self.participant.identity}] Handler failed: {e}", exc_info=True)
        finally:
            logging.info(f"[{self.participant.identity}] 🛑 Cleaning up...")
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
            if "Unexpected response" in error_msg and "'text': ''" in error_msg:
                logging.debug(f"[{pid}] 💤 Empty response (silence)")
            elif "Confidence" in error_msg and "bellow threshold" in error_msg:
                logging.debug(f"[{pid}] ⚠️ Low confidence ignored")
            else:
                logging.error(f"[{pid}] ❌ Stream error: {error_msg}", exc_info=False)

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
            speaker_locale = self._get_locale()

            await self.publish_callback(
                text=text,
                final=is_final,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                speaker_locale=speaker_locale
            )

        except Exception as e:
            logging.error(f"[{pid}] 🧩 Transcription error: {e}", exc_info=True)

    def _get_username(self) -> str:
        try:
            metadata = json.loads(self.participant.metadata or '{}')
            if "username" in metadata:
                return metadata["username"]
        except Exception:
            pass
        return self.participant.name or self.participant.identity

    def _get_locale(self) -> str:
        """참가자의 locale 추출"""
        try:
            metadata = json.loads(self.participant.metadata or '{}')
            return metadata.get("locale", "ko")
        except Exception:
            return "ko"

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
        logging.info(f"[{pid}] Stopping...")
        self._is_running = False
        await self._cleanup_stream_task()


async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)
    
    try:
        base_stt = clova_stt.STT(language=os.getenv("CLOVA_LANG", "enko"))
        stt_model = StreamAdapter(stt=base_stt, vad=silero.VAD.load())
        logging.info("✅ STT provider = CLOVA (multilingual)")
    except Exception as e:
        logging.exception(f"❌ Failed to initialize STT: {e}")
        return
    
    handlers: Dict[str, ParticipantSTTHandler] = {}

    # 🌍 참가자 locale 추적
    participant_locales: Dict[str, str] = {}

    def _get_participant_locale(participant) -> str:
        """참가자의 locale 가져오기"""
        try:
            metadata = json.loads(participant.metadata or '{}')
            return metadata.get("locale", "ko")
        except Exception:
            return "ko"

    def _update_participant_locales():
        """모든 참가자의 locale 업데이트"""
        participant_locales.clear()
        
        # Local participant
        if ctx.room.local_participant:
            locale = _get_participant_locale(ctx.room.local_participant)
            participant_locales[ctx.room.local_participant.identity] = locale
        
        # Remote participants
        for p in ctx.room.remote_participants.values():
            locale = _get_participant_locale(p)
            participant_locales[p.identity] = locale
        
        logging.info(f"🌍 Participant locales: {participant_locales}")

    async def translate_to_multiple_languages(
        text: str, 
        source_lang: str, 
        target_locales: Set[str]
    ) -> Dict[str, str]:
        """여러 언어로 동시 번역"""
        translations = {}
        
        # 원문 언어는 번역하지 않음
        target_locales = target_locales - {source_lang}
        
        if not target_locales or not PAPAGO_CLIENT_ID or not PAPAGO_CLIENT_SECRET:
            return translations
        
        source_papago = LOCALE_TO_PAPAGO.get(source_lang, "ko")
        
        headers = {
            "x-ncp-apigw-api-key-id": PAPAGO_CLIENT_ID,
            "x-ncp-apigw-api-key": PAPAGO_CLIENT_SECRET,
        }
        
        async with httpx.AsyncClient() as client:
            tasks = []
            for target_locale in target_locales:
                target_papago = LOCALE_TO_PAPAGO.get(target_locale)
                if not target_papago:
                    continue
                
                # Papago는 같은 언어 번역 불가
                if source_papago == target_papago:
                    continue
                
                data = {
                    "source": source_papago,
                    "target": target_papago,
                    "text": text,
                    "honorific": True,
                }
                
                tasks.append((target_locale, client.post(PAPAGO_URL, headers=headers, data=data)))
            
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for (target_locale, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logging.error(f"❌ Translation to {target_locale} failed: {result}")
                    continue
                
                try:
                    result.raise_for_status()
                    result_json = result.json()
                    translated_text = result_json.get("message", {}).get("result", {}).get("translatedText")
                    if translated_text:
                        translations[target_locale] = translated_text
                        logging.info(f"✅ Translated to {target_locale}: {translated_text[:30]}...")
                except Exception as e:
                    logging.error(f"❌ Parse error for {target_locale}: {e}")
        
        return translations

    async def _publish_stt(
        text: str, 
        final: bool, 
        identity: str, 
        psid: str, 
        username: str,
        speaker_locale: str
    ):
        """STT 결과를 DataChannel로 발행 (다국어 번역 포함)"""
        
        log_type = "FINAL" if final else "PARTIAL"
        logging.info(f"📤 STT [{log_type}] {username} ({speaker_locale}): {text[:50]}...")
        
        translations_by_locale = {}
        
        try:
            if final and text:
                # 참가자 locale 정보 업데이트
                _update_participant_locales()
                
                # 필요한 타겟 언어들 추출
                target_locales = set(participant_locales.values())
                
                logging.info(f"🔄 Translating to: {target_locales}")
                
                # 다국어 번역 실행
                translations_by_locale = await translate_to_multiple_languages(
                    text=text,
                    source_lang=speaker_locale,
                    target_locales=target_locales
                )
                
                logging.info(f"✅ Translations complete: {list(translations_by_locale.keys())}")

            payload = {
                "type": "stt",
                "final": final,
                "text": text or "",
                "identity": identity,
                "username": username,
                "participant_sid": psid,
                "speaker_locale": speaker_locale,
                "translations": translations_by_locale,  # 🌍 locale별 번역
                "ts": time.time(),
            }

            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic="stt",
            )

        except Exception as e:
            logging.exception(f"❌ publish_data failed: {e}")

    def _on_track_published(publication, participant):
        asyncio.create_task(_handle_track_published(publication, participant))
    
    async def _handle_track_published(publication, participant):
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return
        
        handler_key = f"{participant.sid}_{publication.sid}"
        if handler_key in handlers:
            return
        
        logging.info(f"🎤 [{participant.identity}] Audio track published")
        
        track = publication.track
        if not track:
            await asyncio.sleep(0.5)
            track = publication.track
        
        if not track:
            logging.warning(f"[{participant.identity}] Track not available")
            return
        
        # Locale 업데이트
        _update_participant_locales()
        
        handler = ParticipantSTTHandler(
            participant=participant,
            track=track,
            stt_model=stt_model,
            room=ctx.room,
            publish_callback=_publish_stt
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
            logging.info(f"🔇 [{participant.identity}] Audio track unpublished")
            await handler.stop()
        
        # Locale 업데이트
        _update_participant_locales()
    
    # 참가자 입장/퇴장 이벤트 추가
    def _on_participant_connected(participant):
        logging.info(f"👤 Participant connected: {participant.identity}")
        _update_participant_locales()
    
    def _on_participant_disconnected(participant):
        logging.info(f"👋 Participant disconnected: {participant.identity}")
        _update_participant_locales()
    
    ctx.room.on("track_published", _on_track_published)
    ctx.room.on("track_unpublished", _on_track_unpublished)
    ctx.room.on("participant_connected", _on_participant_connected)
    ctx.room.on("participant_disconnected", _on_participant_disconnected)
    
    logging.info("🎙️ Multilingual STT Agent starting...")
    
    try:
        await ctx.connect()
        logging.info("✅ Connected to room")
    except Exception as e:
        logging.exception(f"❌ Failed to connect: {e}")
        return
    
    await asyncio.sleep(1)
    
    # 초기 locale 수집
    _update_participant_locales()
    
    logging.info("🔍 Processing existing tracks...")
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.track:
                await _handle_track_published(publication, participant)
    
    logging.info("🎙️ Multilingual STT Agent ready")
    
    try:
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.warning("Cancelled by dispatcher")
    except Exception as e:
        logging.exception(f"Error: {e}")
    finally:
        for handler in list(handlers.values()):
            await handler.stop()
        logging.info("🛑 STT Agent stopped")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
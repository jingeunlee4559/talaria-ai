import os, json, sys, asyncio, logging, time
from typing import Optional, Tuple, Any, List, Dict
import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, WorkerOptions, cli
from livekit.plugins import silero
from livekit.agents.stt import StreamAdapter

# Clova STT import
from livekit.plugins import clova as clova_stt

load_dotenv()

PAPAGO_CLIENT_ID = os.getenv("PAPAGO_CLIENT_ID", "YOUR_CLIENT_ID")
PAPAGO_CLIENT_SECRET = os.getenv("PAPAGO_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
PAPAGO_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
src_path = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class STTAgent(Agent):
    def __init__(self, room):
        super().__init__(instructions="Multi-user STT agent with Clova")
        self.room = room


# ------------------------------
# 🆕 각 참가자별 STT 스트림 관리 클래스
# ------------------------------


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

    # --------------------------------------------------
    # 🎤 STT 시작 (자동 재시작 로직 포함)
    # --------------------------------------------------
    async def start(self):
        audio_stream = rtc.AudioStream(self.track)

        try:
            while self._is_running:
                pid = getattr(self.participant, "identity", "unknown")
                logging.info(f"[{pid}] 🔄 Starting new STT stream...")

                # 1️⃣ 새 STT 스트림 생성
                self.stream = self.stt.stream()
                self.task = asyncio.create_task(self._process_stream())

                try:
                    # 2️⃣ 오디오 프레임 전달
                    async for event in audio_stream:
                        if not self._is_running:
                            break
                        if self.stream:
                            self.stream.push_frame(event.frame)

                    # 트랙이 unpublish된 경우 종료
                    break

                except RuntimeError as e:
                    msg = str(e)
                    if "is closed" in msg:
                        logging.warning(f"[{pid}] ⚠️ Stream closed (likely silence). Restarting...")
                    else:
                        logging.error(f"[{pid}] Audio push runtime error: {msg}", exc_info=True)
                        raise

                # 3️⃣ 스트림 및 태스크 정리
                await self._cleanup_stream_task()

                # stop() 호출로 중단되었는지 확인
                if not self._is_running:
                    break

                await asyncio.sleep(0.1)  # 재시작 대기

        except asyncio.CancelledError:
            logging.info(f"[{self.participant.identity}] STT handler task cancelled")

        except Exception as e:
            logging.error(
                f"[{self.participant.identity}] STT handler failed catastrophically: {e}",
                exc_info=True
            )

        finally:
            logging.info(f"[{self.participant.identity}] 🛑 Cleaning up STT handler...")
            self._is_running = False
            await self._cleanup_stream_task()
            await audio_stream.aclose()

    # --------------------------------------------------
    # 🧩 STT 결과 처리 루프
    # --------------------------------------------------
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
                # 1️⃣ Clova: 빈 응답 (무음)
                if "Unexpected response" in error_msg and "'text': ''" in error_msg:
                    logging.debug(f"[{pid}] 💤 Clova: 빈 응답 감지 (무음 처리됨)")

                # 2️⃣ StreamAdapter: 신뢰도 임계값 미만
                elif "Confidence" in error_msg and "bellow threshold" in error_msg:
                    logging.debug(f"[{pid}] ⚠️ 낮은 신뢰도 결과 무시: {error_msg}")

                # 3️⃣ 기타 오류
                else:
                    logging.error(f"[{pid}] ❌ STT 스트림 처리 오류: {error_msg}", exc_info=False)

            except Exception as log_error:
                logging.error(f"[{pid}] 🚨 예외 로깅 중 오류: {log_error}")


    # --------------------------------------------------
    # 🗣️ 전사 결과 처리 및 전송
    # --------------------------------------------------
    async def _handle_transcription(self, event):
        pid = getattr(self.participant, "identity", "unknown")
        try:
            # 이벤트 타입 식별
            event_type = getattr(event, 'type', '').lower()
            is_final = 'final' in event_type and 'interim' not in event_type

            # 텍스트 추출
            text = ""
            if hasattr(event, 'alternatives') and event.alternatives:
                text = event.alternatives[0].text
            elif hasattr(event, 'text'):
                text = event.text

            # 빈 텍스트는 무시
            if not text or not text.strip():
                logging.debug(f"[{pid}] Empty text skipped")
                return

            username = self._get_username()

            # DataChannel 전송
            await self.publish_callback(
                text=text,
                final=is_final,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username
            )

        except Exception as e:
            logging.error(f"[{pid}] 🧩 Transcription handling error: {e}", exc_info=True)

    # --------------------------------------------------
    # 👤 username 추출
    # --------------------------------------------------
    def _get_username(self) -> str:
        try:
            metadata = json.loads(self.participant.metadata or '{}')
            if "username" in metadata:
                return metadata["username"]
        except Exception:
            pass
        return self.participant.name or self.participant.identity

    # --------------------------------------------------
    # 🧹 스트림/태스크 정리
    # --------------------------------------------------
    async def _cleanup_stream_task(self):
        """스트림 및 태스크 안전 종료"""
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

    # --------------------------------------------------
    # 🛑 STT 정지
    # --------------------------------------------------
    async def stop(self):
        pid = getattr(self.participant, "identity", "unknown")
        logging.info(f"[{pid}] Received stop signal. Cleaning up...")
        self._is_running = False
        await self._cleanup_stream_task()
        logging.info(f"[{pid}] STT handler stopped cleanly")


async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)
    
    # ✅ Clova STT 초기화 (StreamAdapter 사용)
    try:
        base_stt = clova_stt.STT(
            language=os.getenv("CLOVA_LANG", "enko"),
        )
        
        # StreamAdapter로 래핑 (더 긴 VAD 설정)
        stt_model = StreamAdapter(
            stt=base_stt,
            # vad=silero.VAD.load(
            #     min_speech_duration=0.5,  # 최소 음성 길이 (초)
            #     min_silence_duration=0.8,  # 최소 침묵 길이 (초)
            # ),
            
            vad=silero.VAD.load(

            ),
        )
        
        
        
        logging.info("✅ STT provider = CLOVA (with StreamAdapter)")
        logging.info(f"   Language: {os.getenv('CLOVA_LANG', 'ko-KR')}")
    except Exception as e:
        logging.exception(f"❌ Failed to initialize Clova STT: {e}")
        return
    
    # 참가자별 STT 핸들러 관리
    handlers: Dict[str, ParticipantSTTHandler] = {}

    # --------------------------------------------------
    # 🗣️ 파파고 STT 텍스트 번역
    # --------------------------------------------------
    async def translate_text_papago(text: str, source_lang="ko", target_lang="en") -> Optional[str]:
        """Papago API를 사용해 텍스트를 비동기적으로 번역합니다."""
        if not text or not PAPAGO_CLIENT_ID or not PAPAGO_CLIENT_SECRET:
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
            async with httpx.AsyncClient() as client:
                response = await client.post(PAPAGO_URL, headers=headers, data=data)
                response.raise_for_status()  # 오류가 있으면 예외 발생
                
                result_json = response.json()
                translated_text = result_json.get("message", {}).get("result", {}).get("translatedText")
                return translated_text
                
        except Exception as e:
            logging.error(f"❌ Papago 번역 실패: {e}", exc_info=True)
            return None
    
    # ------------------------------
    # DataChannel 퍼블리시
    # ------------------------------
    async def _publish_stt(text: str, final: bool, identity: str, psid: str, username: str):
        """STT 결과를 DataChannel로 발행"""

        translated_text: Optional[str] = None
        
        log_type = "FINAL" if final else "PARTIAL"
        logging.info(f"📤 STT [{log_type}] {username}: {text[:50]}...")
        
        try:
            if final and text:
                logging.info(f"🔄 [{username}] Papago 번역 시작...")
                
                # (위에서 정의한 Papago 함수 호출)
                translated_text = await translate_text_papago(text, source_lang="ko", target_lang="en") 
                
                if translated_text:                
                    logging.info(f"📤 TRANS [FINAL] {username}: {translated_text[:50]}...")

            payload = {
                "type": "stt",
                "final": final,
                "text": text or "",
                "identity": identity,
                "username": username,
                "participant_sid": psid,
                "ts": time.time(),
                "translated": translated_text,
            }

            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic="stt",
            )

        except Exception as e:
            logging.exception(f"❌ publish_data failed: {e}")
    
    # ------------------------------
    # 트랙 발행 이벤트
    # ------------------------------
    def _on_track_published(publication, participant):
        """트랙이 발행되면 호출 (동기 콜백)"""
        asyncio.create_task(_handle_track_published(publication, participant))
    
    async def _handle_track_published(publication, participant):
        """오디오 트랙 발행 처리 (비동기 로직)"""
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
            publish_callback=_publish_stt
        )
        handlers[handler_key] = handler
        asyncio.create_task(handler.start())
    
    # ------------------------------
    # 트랙 제거 이벤트
    # ------------------------------
    def _on_track_unpublished(publication, participant):
        """트랙이 제거되면 호출 (동기 콜백)"""
        asyncio.create_task(_handle_track_unpublished(publication, participant))
    
    async def _handle_track_unpublished(publication, participant):
        """트랙 제거 처리 (비동기 로직)"""
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return
        
        handler_key = f"{participant.sid}_{publication.sid}"
        handler = handlers.pop(handler_key, None)
        
        if handler:
            logging.info(f"🔇 [{participant.identity}] Audio track unpublished: {publication.sid}")
            await handler.stop()
    
    # ------------------------------
    # 이벤트 리스너 등록
    # ------------------------------
    ctx.room.on("track_published", _on_track_published)
    ctx.room.on("track_unpublished", _on_track_unpublished)
    
    logging.info("🎙️ Multi-user STT Agent starting...")
    
    # ------------------------------
    # 룸 연결 대기
    # ------------------------------
    try:
        await ctx.connect()
        logging.info("✅ Connected to room")
    except Exception as e:
        logging.exception(f"❌ Failed to connect to room: {e}")
        return
    
    # ------------------------------
    # 이미 발행된 트랙 처리
    # ------------------------------
    await asyncio.sleep(1)
    
    logging.info("🔍 Processing existing tracks...")
    
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.track:
                logging.info(f"📌 Found existing audio track from {participant.identity}")
                await _handle_track_published(publication, participant)
    
    logging.info("🎙️ Multi-user STT Agent ready (Clova)")
    
    # ------------------------------
    # Agent 유지
    # ------------------------------
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
        logging.info("🛑 STT Agent stopped")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
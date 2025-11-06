import os, json, sys, asyncio, logging, time
from typing import Optional, Tuple, Any, List, Dict

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, WorkerOptions, cli
from livekit.plugins import silero
from livekit.agents.stt import StreamAdapter

# Clova STT import
from livekit.plugins import clova as clova_stt

load_dotenv()

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
        """오디오 트랙으로 STT 시작"""
        try:
            logging.info(f"[{self.participant.identity}] Starting Clova STT with track: {self.track.sid}")
            
            # STT 스트림 생성
            self.stream = self.stt.stream()
            self.task = asyncio.create_task(self._process_stream())
            
            # 오디오 데이터를 STT로 전송
            async for event in rtc.AudioStream(self.track):
                if not self._is_running:
                    break
                if self.stream:
                    try:
                        self.stream.push_frame(event.frame)
                    except RuntimeError as e:
                        if "is closed" in str(e):
                            logging.warning(f"[{self.participant.identity}] Stream closed, stopping")
                            break
                        raise
                    
        except asyncio.CancelledError:
            logging.info(f"[{self.participant.identity}] STT handler cancelled")
        except Exception as e:
            logging.error(f"[{self.participant.identity}] STT handler error: {e}")
        finally:
            self._is_running = False
    
    async def _process_stream(self):
        """STT 결과 처리"""
        try:
            async for event in self.stream:
                if not self._is_running:
                    break
                await self._handle_transcription(event)
        except StopAsyncIteration:
            logging.info(f"[{self.participant.identity}] STT stream ended")
        except Exception as e:
            # 빈 응답 에러는 무시하고 계속 진행
            error_msg = str(e)
            if "Unexpected response" in error_msg and "'text': ''" in error_msg:
                logging.debug(f"[{self.participant.identity}] Empty response from Clova (silence detected)")
            else:
                logging.error(f"[{self.participant.identity}] Stream processing error: {e}")
    
    async def _handle_transcription(self, event):
        """전사 결과를 DataChannel로 발행"""
        try:
            # 이벤트 타입 확인
            event_type = getattr(event, 'type', None)
            
            # Clova STT는 SPEECH_FINAL, SPEECH_INTERIM으로 구분
            is_final = True
            if event_type:
                event_type_str = str(event_type).lower()
                if 'interim' in event_type_str or 'partial' in event_type_str:
                    is_final = False
                elif 'final' in event_type_str:
                    is_final = True
            
            # 텍스트 추출
            text = ""
            if hasattr(event, 'alternatives') and event.alternatives:
                text = event.alternatives[0].text
            elif hasattr(event, 'text'):
                text = event.text
            
            # 빈 텍스트는 무시
            if not text or not text.strip():
                logging.debug(f"[{self.participant.identity}] Empty text, skipping")
                return
            
            # username 추출
            username = self._get_username()
            
            await self.publish_callback(
                text=text,
                final=is_final,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username
            )
            
        except Exception as e:
            logging.error(f"[{self.participant.identity}] Transcription handling error: {e}")
    
    def _get_username(self) -> str:
        """participant의 username 추출"""
        try:
            metadata = json.loads(self.participant.metadata or '{}')
            if "username" in metadata:
                return metadata["username"]
        except:
            pass
        return self.participant.name or self.participant.identity
    
    async def stop(self):
        """STT 스트림 정리"""
        self._is_running = False
        
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        if self.stream:
            try:
                await self.stream.aclose()
            except:
                pass
        
        logging.info(f"[{self.participant.identity}] STT handler stopped")


async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)
    
    # ✅ Clova STT 초기화 (StreamAdapter 사용)
    try:
        base_stt = clova_stt.STT(
            language=os.getenv("CLOVA_LANG", "ko-KR"),
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
    
    # ------------------------------
    # DataChannel 퍼블리시
    # ------------------------------
    async def _publish_stt(text: str, final: bool, identity: str, psid: str, username: str):
        """STT 결과를 DataChannel로 발행"""
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
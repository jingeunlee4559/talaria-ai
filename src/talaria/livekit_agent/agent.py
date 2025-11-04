import os, json, sys, asyncio, logging, time
from typing import Optional, Tuple, Any, List

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import silero
from livekit.plugins import openai  # ← 현재 OpenAI STT 사용

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
src_path = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class STTAgent(Agent):
    def __init__(self, room):
        super().__init__(instructions="STT-function agent")
        self.room = room


# ------------------------------
# 유틸: 디바운스 (active speaker 스위칭 과도 시 진동 방지)
# ------------------------------
class Debouncer:
    def __init__(self, delay_sec: float = 0.3):
        self._delay = delay_sec
        self._task: Optional[asyncio.Task] = None

    def run(self, coro_func, *args, **kwargs):
        if self._task and not self._task.done():
            self._task.cancel()
        async def _job():
            try:
                await asyncio.sleep(self._delay)
                await coro_func(*args, **kwargs)
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.exception("[Debouncer] job error")
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(_job())


async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)

    # --- AgentSession 구성 ---
    session = AgentSession(
        stt=openai.STT(
            model="gpt-4o-transcribe",
            language="ko",        # OpenAI STT에 맞춘 언어 코드
        ),
        vad=silero.VAD.load(),   # 필요 시 민감도 조정: threshold는 STT 쪽, 여기선 기본
    )

    # 현재 STT 입력을 누구로 볼지 추적 (액티브 스피커 기반)
    current_speaker_identity: Optional[str] = None
    current_speaker_sid: Optional[str] = None
    switch_debouncer = Debouncer(delay_sec=0.35)  # 빠른 스위칭 완화

    # room_io가 있는 버전이면 사용 (없으면 스위칭만 추적)
    room_io = getattr(session, "room_io", None)
    has_set_participant = hasattr(room_io, "set_participant") if room_io else False

    async def _switch_stt_input_to(identity: Optional[str], sid: Optional[str]):
        nonlocal current_speaker_identity, current_speaker_sid
        if identity:
            current_speaker_identity = identity
        if sid:
            current_speaker_sid = sid

        if has_set_participant and identity:
            try:
                # 🔁 STT 입력 참가자 전환 (지원 버전에서만)
                room_io.set_participant(identity)
                logging.info(f"[RoomIO] switched STT input to participant identity={identity} sid={sid}")
            except Exception:
                logging.exception("[RoomIO] set_participant failed")

    # ------------------------------
    # 🆕 유틸: participant로부터 실제 표시용 이름 가져오기
    # ------------------------------
    def _get_display_name(participant) -> str:
        """
        participant의 metadata에서 username을 추출하거나,
        없으면 name 또는 identity를 반환
        """
        try:
            # 1) metadata에 username이 있는지 확인
            metadata_str = getattr(participant, "metadata", "")
            if metadata_str:
                metadata = json.loads(metadata_str)
                if "username" in metadata:
                    return metadata["username"]
            
            # 2) name 속성 확인
            name = getattr(participant, "name", "")
            if name:
                return name
            
            # 3) fallback: identity
            return getattr(participant, "identity", "Unknown")
        except Exception:
            return getattr(participant, "identity", "Unknown")

    # ------------------------------
    # 액티브 스피커 변경 이벤트: 현재 화자를 추적하고 STT 입력 전환
    # ------------------------------
    @ctx.room.on("active_speakers_changed")
    def _on_active_speakers_changed(*args: Any):
        """
        SDK/버전별 시그니처 차이를 흡수하기 위해 가변 인자 처리:
        보통 첫 인자에 speakers(list[Participant])가 옵니다.
        """
        try:
            speakers: Optional[List[Any]] = None
            # args 예: (speakers,) 또는 (room, speakers) 등일 수 있으니 유연 처리
            for a in args:
                if isinstance(a, list):
                    speakers = a
                    break
            if not speakers:
                return
            # 가장 우선 스피커를 택함
            top = speakers[0]
            pid = getattr(top, "identity", None)
            psid = getattr(top, "sid", None) or getattr(top, "participant_sid", None)
            if pid or psid:
                switch_debouncer.run(_switch_stt_input_to, pid, psid)
        except Exception:
            logging.exception("active_speakers_changed handler error")

    # ------------------------------
    # DataChannel 퍼블리시 (v1 시그니처)
    # ------------------------------
    async def _publish_stt(text: str, final: bool, identity: Optional[str], psid: Optional[str], username: Optional[str]):
        """
        🆕 username 파라미터 추가: 프론트엔드에서 표시할 실제 이름
        """
        payload = {
            "type": "stt",
            "final": final,
            "text": text or "",
            "identity": identity or current_speaker_identity or "",  # 원본 identity는 유지
            "username": username or identity or current_speaker_identity or "",  # 🆕 표시용 이름
            "participant_sid": psid or current_speaker_sid,
            "ts": time.time(),
        }
        logging.info(f"📤 STT publish [{'FINAL' if final else 'PARTIAL'}]: username={username}, text={text[:30]}...")
        try:
            # ✅ Python RTC v1: payload(bytes), reliable=True, topic="stt"
            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic="stt",
            )
            logging.info(f"✅ Published STT: {text[:50]}...")
        except Exception as e:
            logging.exception(f"❌ publish_data 실패: {e}")

    # ------------------------------
    # 전사 이벤트: 실제 오디오 트랙의 참가자를 정확하게 추출
    # ------------------------------
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        # 디버깅용 원본 출력
        logging.debug(f"[ev] {ev}")
        try:
            text = getattr(ev, "transcript", "")
            is_final = bool(getattr(ev, "is_final", False))
            
            # 🔍 실제 말한 사람 찾기 (우선순위 순서)
            identity = None
            psid = None
            username = None
            participant_obj = None
            
            # 1) 이벤트에서 직접 participant 정보 확인
            p = getattr(ev, "participant", None)
            if p is not None:
                identity = getattr(p, "identity", None)
                psid = getattr(p, "sid", None) or getattr(p, "participant_sid", None)
                username = _get_display_name(p)
                participant_obj = p
                logging.info(f"🎤 Speaker from ev.participant: {username} (identity={identity})")
            
            # 2) track_sid가 있으면 room에서 해당 트랙의 참가자 찾기
            if not identity:
                track_sid = getattr(ev, "track_sid", None)
                if track_sid:
                    # local participant 확인
                    for track_pub in ctx.room.local_participant.track_publications.values():
                        if track_pub.sid == track_sid:
                            identity = ctx.room.local_participant.identity
                            psid = ctx.room.local_participant.sid
                            username = _get_display_name(ctx.room.local_participant)
                            participant_obj = ctx.room.local_participant
                            logging.info(f"🎤 Found LOCAL speaker by track_sid: {username}")
                            break
                    
                    # remote participants 확인
                    if not identity:
                        for participant in ctx.room.remote_participants.values():
                            for track_pub in participant.track_publications.values():
                                if track_pub.sid == track_sid:
                                    identity = participant.identity
                                    psid = participant.sid
                                    username = _get_display_name(participant)
                                    participant_obj = participant
                                    logging.info(f"🎤 Found REMOTE speaker by track_sid: {username}")
                                    break
                            if identity:
                                break
            
            # 3) fallback: 우리가 추적한 active speaker 사용
            if not identity:
                identity = current_speaker_identity
                psid = current_speaker_sid
                # active speaker의 username 찾기
                if identity:
                    if ctx.room.local_participant.identity == identity:
                        username = _get_display_name(ctx.room.local_participant)
                    else:
                        for p in ctx.room.remote_participants.values():
                            if p.identity == identity:
                                username = _get_display_name(p)
                                break
                logging.info(f"🎤 Using active speaker fallback: {username}")

            loop = asyncio.get_running_loop()
            loop.create_task(_publish_stt(text, is_final, identity, psid, username))

        except Exception:
            logging.exception("transcribed 이벤트 파싱 실패")
            return

    # ------------------------------
    # 종료/에러 로깅
    # ------------------------------
    @session.on("end")
    def on_end(ev=None):
        logging.info(f"[AgentSession END] ev={ev}")

    @session.on("error")
    def on_error(err):
        logging.exception(f"[AgentSession ERROR] {err}")

    # ------------------------------
    # 세션 시작
    # ------------------------------
    try:
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(
                noise_cancellation=None,
                close_on_disconnect=False,  # 잠깐 이탈해도 세션 유지
            ),
        )
    except asyncio.CancelledError:
        logging.warning("[JobContext] cancelled by dispatcher")
        raise
    except Exception as e:
        logging.exception(f"[entrypoint] session.start 예외: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
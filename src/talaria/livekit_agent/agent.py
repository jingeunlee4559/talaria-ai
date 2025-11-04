import os, json, sys, asyncio, logging, time
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import silero, clova

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


async def entrypoint(ctx: agents.JobContext):
    agent = STTAgent(ctx.room)

    session = AgentSession(
        stt=clova.STT(language="ko-KR", threshold=0.3),  # 필요시 0.2로 완화
        vad=silero.VAD.load(),
    )

    async def _publish_stt(text: str, final: bool, speaker_id: str | None):
        payload = {
            "type": "stt",
            "final": final,
            "text": text or "",
            "identity": speaker_id or "user",
            "track_sid": None,
            "ts": time.time(),
        }
        logging.info(f"📤 STT publish [{'FINAL' if final else 'PARTIAL'}]: {payload}")
        try:
            # ✅ v1 시그니처: payload(bytes), reliable=True, topic="stt"
            await ctx.room.local_participant.publish_data(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic="stt",
            )
            logging.info(f"✅ Published STT: {text[:50]}...")
        except Exception as e:
            logging.exception(f"❌ publish_data 실패: {e}")

    # ✅ 올바른 이벤트: user_input_transcribed (is_final로 최종 여부 구분)
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        try:
            text = getattr(ev, "transcript", "")
            is_final = bool(getattr(ev, "is_final", False))
            speaker_id = getattr(ev, "speaker_id", None)
        except Exception:
            logging.exception("transcribed 이벤트 파싱 실패")
            return

        loop = asyncio.get_running_loop()
        loop.create_task(_publish_stt(text, is_final, speaker_id))

    # ✅ 종료/에러 로깅 핸들러
    @session.on("end")
    def on_end(ev=None):
        logging.info(f"[AgentSession END] ev={ev}")

    @session.on("error")
    def on_error(err):
        logging.exception(f"[AgentSession ERROR] {err}")

    try:
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(
                noise_cancellation=None,
                close_on_disconnect=False,  # 참가자 이탈시 세션 유지(원인 파악/복구에 유리)
            ),
        )
    except asyncio.CancelledError:
        # 디스패처/워커가 잡을 내릴 때 여기로 들어옵니다.
        logging.warning("[JobContext] cancelled by dispatcher")
        raise
    except Exception as e:
        logging.exception(f"[entrypoint] session.start 예외: {e}")
        raise



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

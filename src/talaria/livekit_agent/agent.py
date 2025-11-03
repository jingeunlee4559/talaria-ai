import os, json
import sys
import asyncio
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import clova
from livekit.plugins import openai

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

src_path = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

class STTAgent(Agent):
    def __init__(self):
        super().__init__(instructions="STT-only agent")

async def entrypoint(ctx: agents.JobContext):

    session = AgentSession(
        stt=openai.STT(
            model="gpt-4o-transcribe", 
            language="ko",
            temperature=0.2
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=STTAgent(),
        room_input_options=RoomInputOptions(noise_cancellation=None),
    )

    # --- 비동기(await)로 실행할 publish 작업을 별도 함수로 분리 ---
    async def _publish_transcript(payload: dict, reliable: bool):
        """
        비동기 publish 로직을 처리하는 헬퍼 함수
        """
        await session.room.local_participant.publish_data(
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            reliable=reliable,
            topic="stt",
        )

    @session.on("transcript")
    def on_transcript(ev):
        is_final = bool(getattr(ev, "is_final", False))
        text = getattr(ev, "text", "") or ""

        payload = {
            "type": "stt",
            "final": is_final,
            "text": getattr(ev, "text", "") or "",
            "identity": getattr(ev, "participant_identity", None),
            "track_sid": getattr(ev, "track_sid", None),
            "ts": getattr(ev, "timestamp", None),
        }
        
        identity_str = payload["identity"] or "AGENT"
        log_prefix = "[FINAL]" if is_final else "[PARTIAL]"
        logging.info(f"{log_prefix} STT ({identity_str}): {text}")
        asyncio.create_task(_publish_transcript(payload, is_final))

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
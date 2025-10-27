# talaria-ai/agent_stt.py
import os, json
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import clova

load_dotenv()

class STTOnlyAgent(Agent):
    def __init__(self):
        super().__init__(instructions="STT-only agent")

async def entrypoint(ctx: agents.JobContext):
    #  CLOVA STT로 세션 구성
    session = AgentSession(
        stt=clova.STT(
            language="ko-KR", # ko-KR / en-US / ja / zh-cn / zh-tw / enko 등
            # word_boost=["Talaria", "LiveKit"],  # 선택: 용어 부스트
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=STTOnlyAgent(),
        room_input_options=RoomInputOptions(noise_cancellation=None),
    )

    @session.on("transcript")
    async def on_transcript(ev):
        is_final = bool(getattr(ev, "is_final", False))
        payload = {
            "type": "stt",
            "final": is_final,
            "text": getattr(ev, "text", "") or "",
            "identity": getattr(ev, "participant_identity", None),
            "track_sid": getattr(ev, "track_sid", None),
            "ts": getattr(ev, "timestamp", None),
        }
        # partial → unreliable / final → reliable 로 전송
        await session.room.local_participant.publish_data(
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            reliable=is_final,
            topic="stt",
        )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

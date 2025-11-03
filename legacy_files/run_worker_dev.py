import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    # .env 로드 (현재 디렉토리)
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

    # 필수 키 확인
    required = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing required .env keys: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # livekit agents CLI(dev) 호출
    cmd = [
        sys.executable, "-m", "livekit.agents.cli", "dev",
        "--url", os.getenv("LIVEKIT_URL"),
        "--api-key", os.getenv("LIVEKIT_API_KEY"),
        "--api-secret", os.getenv("LIVEKIT_API_SECRET"),
        "agent_stt:entrypoint",
    ]
    print("[INFO] Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()

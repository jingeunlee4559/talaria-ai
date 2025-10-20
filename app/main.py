from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import uuid

app = FastAPI(
    title="Talaria AI Service",
    description="STT / Translation / TTS mock service for backend integration",
    version="0.1.0"
)

# ========================
#  Pydantic Models
# ========================

class STTRequest(BaseModel):
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    language: str = Field(default="auto", description="Source language (auto detect)")

class STTResponse(BaseModel):
    transcript: str
    confidence: float = 0.92
    is_final: bool = True


class TranslateRequest(BaseModel):
    source_text: str
    source_lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    mode: str = Field(default="stream", description="stream | context")

class TranslateResponse(BaseModel):
    translated_text: str
    provider: str = "mock-papago"
    mode: str = "stream"


class TTSRequest(BaseModel):
    text: str
    voice: str = "kr-standard-1"
    speed: float = 0.0
    volume: float = -8.0

class TTSResponse(BaseModel):
    audio_url: str
    duration_ms: int = 1200
    provider: str = "mock-clova"


# ========================
# 📡 API Endpoints
# ========================

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "talaria-ai"}


# ---- STT (Speech to Text)
@app.post("/stt", response_model=STTResponse)
def stt(req: STTRequest):
    # TODO: 실제 CLOVA Speech-to-Text 연동 (지금은 Mock)
    if req.audio_url:
        transcript = f"Mock transcript for {req.audio_url}"
    else:
        transcript = "This is a mock transcript."
    return STTResponse(transcript=transcript)


# ---- Translation (Papago NMT)
@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    # TODO: 실제 Papago API 연동 (지금은 Mock)
    fake_translation = req.source_text[::-1]  # 단순히 문자열 뒤집기
    return TranslateResponse(translated_text=fake_translation, mode=req.mode)


# ---- TTS (Text to Speech)
@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    # TODO: CLOVA TTS 연동 (지금은 Mock)
    fake_url = f"https://example.com/audio/{uuid.uuid4()}.mp3"
    return TTSResponse(audio_url=fake_url)

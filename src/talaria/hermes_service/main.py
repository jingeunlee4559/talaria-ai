# src/talaria/hermes_service/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from talaria.hermes_service.routers import meeting_raw_script
from talaria.hermes_service.routers import summary as summary_router
# ⭐️ [추가] 모델 로더 임포트
from talaria.hermes_service.utils.model_loader import loader as model_loader
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

log = logging.getLogger(__name__)

# ⭐️ [추가] Lifespan (수명 주기) 정의: 서버 시작 시 모델을 미리 로드합니다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 [Startup] Talaria 서버 시작! ML 모델 및 리소스(DB, 족보) 미리 로딩 중...")
    try:
        # 여기서 initialize()를 호출하면 서버가 켜질 때 약간의 시간이 걸리지만,
        # 대신 사용자는 첫 회의부터 즉각적인 반응 속도를 경험할 수 있습니다.
        model_loader.initialize()
        log.info("✅ [Startup] 모든 AI 모델 및 데이터 로딩 완료! 서비스 준비 끝.")
    except Exception as e:
        log.critical(f"❌ [Startup] 모델 로딩 중 치명적 오류 발생: {e}", exc_info=True)
        # 모델 로드 실패하더라도 서버는 켜지게 할지, 아니면 멈출지 결정 (여기선 로그만 남기고 진행)
    
    yield  # 애플리케이션 실행 구간
    
    log.info("👋 [Shutdown] Talaria 서버 종료 중...")

tags_metadata = [
    {"name": "translation", "description": "Papago 문서 번역(doc-trans) 라우트."},
    {"name": "rag", "description": "RAG(VectorDB) 처리 라우트."},
    {"name": "utterances", "description": "실시간 발화 처리 및 저장"},
    {"name": "health", "description": "헬스체크 라우트."},
]

app = FastAPI(
    title="Talaria-AI Doc Translation API",
    version="1.0.0",
    description="Talaria 문서 번역 API",
    openapi_tags=tags_metadata,
    lifespan=lifespan # ⭐️ [추가] lifespan 연결
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 문서 번역 라우터
try:
    from talaria.hermes_service.routers.docs_translation import router as translation_router
    app.include_router(translation_router, tags=["translation"])
except Exception as e:
    log.warning("translation router not loaded: %s", e)

# RAG 라우터
try:
    from talaria.hermes_service.routers.rag import router as rag_router
    app.include_router(rag_router, tags=["rag"])
except Exception as e:
    log.warning("rag router not loaded: %s", e)

# Utterance 라우터
try:
    from talaria.hermes_service.routers.utterance import router as utterance_router
    app.include_router(utterance_router, tags=["utterances"])
except Exception as e:
    log.warning("utterance router not loaded: %s", e)

try:
    app.include_router(summary_router.router)
except Exception as e:
    log.warning("summary router not loaded: %s", e)

# Meeting Raw Script 라우터
app.include_router(meeting_raw_script.router)

@app.get("/health", tags=["health"], summary="Liveness/Readiness probe")
def health(): return {"status": "ok"}

@app.get("/ping", tags=["health"])
def ping(): return {"pong": True}
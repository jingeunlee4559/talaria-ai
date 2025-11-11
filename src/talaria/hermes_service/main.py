from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

tags_metadata = [
    {"name": "translation", "description": "Papago 문서 번역(doc-trans) 라우트."},
    {"name": "rag", "description": "RAG(VectorDB) 처리 라우트."},
    {"name": "health", "description": "헬스체크 라우트."},
]

app = FastAPI(
    title="Talaria-AI Doc Translation API",
    version="1.0.0",
    description="Talaria 문서 번역 API",
    openapi_tags=tags_metadata,
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
    logging.getLogger(__name__).warning("translation router not loaded: %s", e)

# RAG 라우터
try:
    from talaria.hermes_service.routers.rag import router as rag_router
    app.include_router(rag_router, tags=["rag"])
except Exception as e:
    logging.getLogger(__name__).warning("rag router not loaded: %s", e)

@app.get("/health", tags=["health"], summary="Liveness/Readiness probe")
def health(): return {"status": "ok"}

@app.get("/ping", tags=["health"])
def ping(): return {"pong": True}

# %%
# src/talaria/hermes_service/rag_processor/manager.py

import logging
from typing import List, Optional

from .parser import parse_document
from .chunker import chunk_text
from .embedder import get_embedder
from .vector_store import get_vector_store

log = logging.getLogger(__name__)

async def process_and_load_document(
    file_path: str, 
    document_id: str, 
    language: str
) -> bool:
    """
    (⭐️ RAG 파이프라인 총괄 함수)
    업로드된 문서를 파싱, 청킹, 임베딩하여 VectorDB에 저장합니다.
    FastAPI의 BackgroundTask로 실행되는 것을 상정합니다.

    Args:
        file_path (str): 서버에 임시 저장된 파일 경로
        document_id (str): 이 문서의 고유 ID (예: "doc_uuid_abc")
        language (str): 문서의 원본 언어 (예: "ko", "en")

    Returns:
        bool: 파이프라인 전체 성공 여부
    """
    
    log.info(f"✅ RAG 파이프라인 시작. (Document ID: {document_id})")
    
    try:
        # --- 1단계: 파싱 (텍스트 추출) ---
        log.info(f"[{document_id}] 1. 파싱 시작...")
        parsed_text = parse_document(file_path)
        
        if not parsed_text:
            log.error(f"[{document_id}] 1. 파싱 실패: 텍스트를 추출할 수 없습니다.")
            return False

        # --- 2단계: 청킹 (텍스트 분할) ---
        log.info(f"[{document_id}] 2. 청킹 시작...")
        chunked_texts: List[str] = chunk_text(
            text=parsed_text,
            chunk_size=500,     # (우리가 논의한 최적값)
            chunk_overlap=100
        )
        
        if not chunked_texts:
            log.warning(f"[{document_id}] 2. 청킹 완료: 생성된 청크가 없습니다. (문서가 비어있을 수 있음)")
            return True # 실패는 아니므로 True 반환

        log.info(f"[{document_id}] 2. 청킹 완료: {len(chunked_texts)}개 청크 생성.")

        # --- 3단계: 임베딩 (벡터 변환) ---
        log.info(f"[{document_id}] 3. 임베딩 시작...")
        embedder = get_embedder() # (싱글톤) 메모리에 로드된 모델 가져오기
        
        vectors: Optional[List[List[float]]] = embedder.embed_chunks(chunked_texts)

        if vectors is None or len(vectors) != len(chunked_texts):
            log.error(f"[{document_id}] 3. 임베딩 실패: 벡터가 생성되지 않았습니다.")
            return False

        log.info(f"[{document_id}] 3. 임베딩 완료: {len(vectors)}개 벡터 생성.")

        # --- 4단계: VectorDB 저장 ---
        log.info(f"[{document_id}] 4. VectorDB 저장 시작...")
        vector_store = get_vector_store() # (싱글톤) VectorDB 연결 가져오기

        success = vector_store.save_chunks(
            document_id=document_id,
            language=language,
            chunks=chunked_texts,
            vectors=vectors
        )

        if not success:
            log.error(f"[{document_id}] 4. VectorDB 저장 실패.")
            return False

        log.info(f"✅ [{document_id}] RAG 파이프라인 성공적으로 완료.")
        return True

    except Exception as e:
        log.error(f"[{document_id}] RAG 파이프라인 처리 중 치명적 오류 발생: {e}", exc_info=True)
        return False



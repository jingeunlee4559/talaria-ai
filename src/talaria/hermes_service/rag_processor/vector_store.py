# %%
# src/talaria/hermes_service/rag_processor/vector_store.py

import logging
from typing import List, Optional, Dict, Any
import chromadb
import uuid
import os  

log = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))


# --- VectorDB 설정 ---
# 프로젝트 루트 기준으로 절대 경로 생성
CHROMA_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "persistent_data/chroma_db_store")

# 우리가 설계한 '문서 청크' 컬렉션 이름
COLLECTION_NAME = "global_chunks"

class VectorStore:
    """
    ChromaDB를 관리하는 클래스 (싱글톤 패턴).
    텍스트 청크, 벡터, 메타데이터를 저장하고 검색합니다.
    """
    _instance = None  # 싱글톤 인스턴스
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIRECTORY):
        if self._initialized:
            return
            
        try:
            # 서버 재시작 시에도 데이터를 보존하는 'PersistentClient' 사용
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # 'global_chunks' 컬렉션을 가져오거나, 없으면 생성
            #    (참고: embedder.py의 모델 차원(768)과 맞춰야 하지만,
            #     ChromaDB는 첫 데이터 추가 시 자동으로 차원을 설정해줍니다.)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
            
            log.info(f"✅ VectorDB 클라이언트 연결 성공. (저장 경로: {persist_directory})")
            log.info(f"✅ '{COLLECTION_NAME}' 컬렉션 로드 완료.")
            
        except Exception as e:
            log.error(f"❌ ChromaDB 클라이언트 초기화 실패: {e}", exc_info=True)
            self.client = None
            self.collection = None
            
        self._initialized = True

    def save_chunks(
        self,
        document_id: str,
        language: str,
        chunks: List[str],
        vectors: List[List[float]]
    ) -> bool:
        """
        문서에서 추출된 청크와 벡터를 메타데이터와 함께 DB에 저장합니다.

        Args:
            document_id (str): 원본 문서의 고유 ID (예: "doc_uuid_abc")
            language (str): 문서의 언어 (예: "ko", "en")
            chunks (List[str]): 텍스트 청크 리스트
            vectors (List[List[float]]): 임베딩 벡터 리스트

        Returns:
            bool: 저장 성공 여부
        """
        if not self.collection:
            log.error("VectorDB 컬렉션이 로드되지 않아 저장을 중단합니다.")
            return False
            
        if not (len(chunks) == len(vectors) and len(chunks) > 0):
            log.error("청크, 벡터의 개수가 일치하지 않거나 비어있습니다. 저장을 중단합니다.")
            return False

        try:
            log.info(f"[{document_id}] VectorDB에 {len(chunks)}개 청크 저장 시작...")

            # --- ChromaDB에 맞는 3가지 리스트 준비 ---
            
            # 3a. ID 리스트 (각 청크의 고유 ID)
            #     (나중에 'doc_uuid_abc_0', 'doc_uuid_abc_1' ... 처럼 생성)
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            # 3b. 메타데이터 리스트 (우리가 검색할 핵심 정보)
            #     (텍스트 자체도 메타데이터에 저장해야 나중에 쉽게 꺼내볼 수 있음)
            metadatas = [
                {
                    "document_id": document_id,
                    "language": language,
                    "text_content": chunk, # (⭐️) 원본 텍스트 저장
                    "chunk_index": i
                }
                for i, chunk in enumerate(chunks)
            ]

            # 4. (⭐️) DB에 일괄 추가 (add)
            self.collection.add(
                embeddings=vectors,
                metadatas=metadatas,
                ids=ids
            )
            
            log.info(f"✅ [{document_id}] {len(chunks)}개 청크 저장 완료.")
            return True

        except Exception as e:
            log.error(f"[{document_id}] VectorDB 저장 중 오류 발생: {e}", exc_info=True)
            return False

    def search_chunks(
        self,
        query_vector: List[float],
        document_ids: List[str],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        (⭐️ 나중에 사용할 검색 함수)
        특정 문서들 내에서 쿼리 벡터와 가장 유사한 청크를 검색합니다.

        Args:
            query_vector (List[float]): 사용자의 STT 발화를 임베딩한 벡터
            document_ids (List[str]): 검색을 허용할 문서 ID 리스트 (현재 회의에 등록된)
            n_results (int): 상위 몇 개를 가져올지

        Returns:
            List[Dict[str, Any]]: 검색된 청크 정보 (메타데이터, 유사도 점수)
        """
        if not self.collection:
            log.error("VectorDB 컬렉션이 로드되지 않아 검색을 중단합니다.")
            return []
            
        try:
            # (⭐️ 핵심) 우리가 설계한 메타데이터 필터링
            where_filter = {
                "document_id": {"$in": document_ids}
            }

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where_filter
                # (참고) 텍스트 자체를 검색어로 넣으려면 query_texts=["검색어"] 사용
            )
            
            # 검색 결과를 [ { "text": ..., "distance": ... }, ... ] 형태로 가공
            parsed_results = []
            if results and results.get("ids")[0]:
                for i in range(len(results["ids"][0])):
                    parsed_results.append({
                        "text": results["metadatas"][0][i]["text_content"],
                        "language": results["metadatas"][0][i]["language"],
                        "document_id": results["metadatas"][0][i]["document_id"],
                        "distance": results["distances"][0][i] # (유사도 점수)
                    })
            return parsed_results

        except Exception as e:
            log.error(f"VectorDB 검색 중 오류 발생: {e}", exc_info=True)
            return []

def get_vector_store() -> VectorStore:
    """
    어디서든 이 함수를 호출하면,
    이미 연결된 VectorStore 인스턴스를 반환합니다.
    """
    return VectorStore()



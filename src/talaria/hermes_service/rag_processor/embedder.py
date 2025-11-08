# %%
# src/talaria/hermes_service/rag_processor/embedder.py

import logging
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# --- 임베딩 모델 설정 ---
# 1. 다국어 지원 및 성능이 우수한 모델을 선택합니다.
#    'paraphrase-multilingual-mpnet-base-v2'는 50개 이상의 언어를 지원하며,
#    의미 검색(Semantic Search) 성능이 매우 뛰어납니다.
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'


class Embedder:
    """
    텍스트 청크를 임베딩 벡터로 변환하는 클래스.
    (⭐️ 싱글톤 패턴: 모델을 메모리에 한 번만 로드합니다)
    """
    _instance = None  # 싱글톤 인스턴스
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = MODEL_NAME):
        if self._initialized:
            return
            
        log.info(f"임베딩 모델 '{model_name}' 로드를 시작합니다...")
        
        # 1. GPU(cuda) 사용이 가능하면 GPU로, 아니면 CPU로 자동 설정
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 2. SentenceTransformer 모델 로드
            #    (최초 실행 시 모델 파일을 다운로드하므로 시간이 걸릴 수 있습니다)
            self.model = SentenceTransformer(model_name, device=self.device)
            
            log.info(f"✅ 임베딩 모델 로드 완료. (사용 디바이스: {self.device})")
            
        except Exception as e:
            log.error(f"❌ 임베딩 모델 '{model_name}' 로드 실패: {e}", exc_info=True)
            self.model = None
            
        self._initialized = True

    def embed_chunks(self, chunks: List[str]) -> Optional[List[List[float]]]:
        """
        텍스트 청크 리스트를 임베딩 벡터 리스트로 변환합니다.

        Args:
            chunks (List[str]): chunker.py에서 넘어온 텍스트 조각 리스트

        Returns:
            Optional[List[List[float]]]: 벡터 리스트 (실패 시 None)
        """
        if not self.model:
            log.error("임베딩 모델이 로드되지 않아 임베딩을 진행할 수 없습니다.")
            return None
        
        if not chunks:
            log.warning("임베딩할 텍스트 청크가 비어있습니다.")
            return []

        try:
            log.info(f"텍스트 청크 {len(chunks)}개에 대한 임베딩을 시작합니다...")
            
            # 3. (⭐️ 핵심) 모델의 .encode() 함수로 텍스트 리스트를 벡터 리스트로 변환
            #    (배치 처리되므로 매우 빠릅니다)
            embeddings = self.model.encode(
                chunks,
                batch_size=32,  # 32개씩 묶어서 처리
                show_progress_bar=False, # 로그가 지저분해지지 않도록 progress bar는 끔
                convert_to_tensor=False  # DB 저장을 위해 numpy/list 형태로 받음
            )
            
            log.info(f"✅ 임베딩 완료. (벡터 차원: {len(embeddings[0])})")
            
            # numpy array를 일반 list로 변환
            return embeddings.tolist()
        
        except Exception as e:
            log.error(f"청크 임베딩 중 오류 발생: {e}", exc_info=True)
            return None

def get_embedder() -> Embedder:
    """
    어디서든 이 함수를 호출하면,
    이미 메모리에 로드된 Embedder 인스턴스를 반환합니다.
    """
    return Embedder()



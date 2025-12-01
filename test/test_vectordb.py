import sys
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 설정
# ==========================================
# 구축된 VectorDB 경로
DB_PATH = "src/talaria/assets/vectordb"
COLLECTION_NAME = "meme_dictionary"
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-multitask"

def test_vectordb():
    print(f">>> [VectorDB Test] DB 로드 중... (경로: {DB_PATH})")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ VectorDB 폴더가 없습니다: {DB_PATH}")
        print("   -> 'build_vectordb_for_service.py'를 먼저 실행해주세요.")
        return

    # 1. 클라이언트 및 임베딩 모델 준비
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        print(f">>> 임베딩 모델 로드 중 ({EMBEDDING_MODEL_NAME})...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input):
                return embedding_model.encode(input).tolist()

        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=CustomEmbeddingFunction()
        )
        print(f">>> ✅ 컬렉션 로드 완료! (데이터 수: {collection.count()}개)")
        
    except Exception as e:
        print(f"❌ DB 로드 실패: {e}")
        return

    # 2. 검색 테스트 케이스
    # (검색어, 기대하는 결과 키워드)
    queries = [
        ("나락", "나락"),
        ("중꺽마", "중꺾마"),
        ("킹받네 진짜", "킹"), # 문장 검색 테스트
        ("갇생 살기", "갓생")
    ]

    print("\n" + "="*50)
    for query, expected in queries:
        print(f"🔍 검색어: '{query}'")
        
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # 결과 출력
        if results['ids']:
            top_meme = results['metadatas'][0][0]['primary_keyword']
            top_desc = results['metadatas'][0][0]['definition'][:30]
            dist = results['distances'][0][0]
            
            print(f"   👉 Top 1: {top_meme} (거리: {dist:.4f})")
            print(f"      뜻: {top_desc}...")
            
            # 간단 검증
            if expected in top_meme or top_meme in expected:
                print("   ✅ [PASS] 기대한 결과와 일치합니다.")
            else:
                print(f"   ⚠️ [CHECK] 기대한 결과('{expected}')와 다릅니다.")
        else:
            print("   ❌ 검색 결과 없음")
        print("-" * 50)

if __name__ == "__main__":
    test_vectordb()
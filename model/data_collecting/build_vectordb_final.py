import json
import os
import chromadb
try:
    from chromadb import errors as chroma_errors
except ImportError:
    chroma_errors = None

from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = "model/data_collecting"
DATA_FILE = os.path.join(BASE_DIR, "meme_dictionary_complete.json")
DB_PATH = "src/talaria/assets/vectordb"
COLLECTION_NAME = "meme_dictionary"

EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-multitask"

def build_database():
    # 1. 데이터 로드
    if not os.path.exists(DATA_FILE):
        print(f"❌ 오류: 데이터 파일이 없습니다.\n경로: {DATA_FILE}")
        return

    print(f">>> 데이터 로드 중: {DATA_FILE}")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   - 원본 밈 개수: {len(data)}개")

    # 2. 폴더 생성
    os.makedirs(DB_PATH, exist_ok=True)

    # 3. ChromaDB 클라이언트 설정
    print(f">>> VectorDB 생성 경로: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 기존 컬렉션 초기화
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"   - 🗑️ 기존 컬렉션 삭제 완료")
    except Exception:
        pass

    # 4. 임베딩 모델 로드
    print(f">>> 임베딩 모델 로드 중 ({EMBEDDING_MODEL_NAME})...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
        def __call__(self, input):
            # [핵심 수정] normalize_embeddings=True 추가! (거리 점수 안정화)
            return embedding_model.encode(input, normalize_embeddings=True).tolist()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=CustomEmbeddingFunction()
    )

    # 5. 데이터 저장 (정의 + 예시 문장 모두 포함)
    ids = []
    documents = [] 
    metadatas = []

    print(">>> 데이터 벡터화 및 저장 시작 (예시 문장 포함)...")
    
    for item in data:
        if 'primary_keyword' not in item: continue
            
        keyword = item['primary_keyword']
        definition = item.get('definition', '')
        examples = item.get('examples', [])
        
        # 공통 메타데이터 (어떤 데이터가 걸리든 원본 밈 정보를 알아야 함)
        base_meta = {
            "primary_keyword": keyword,
            "definition": definition,
            "tags": ", ".join(item.get('tags', [])),
            "source": item.get('source', 'unknown')
        }

        # (1) 정의 데이터 저장
        def_id = f"{item['id']}_def"
        def_text = f"{keyword}: {definition}" # "킹받네: 열받네"
        
        ids.append(def_id)
        documents.append(def_text)
        metadatas.append(base_meta)
        
        # (2) 예시 문장 데이터 저장 (핵심!)
        # 예시가 있으면 예시들도 다 DB에 넣습니다.
        for i, ex in enumerate(examples):
            # 예시 문장이 너무 짧거나 의미 없으면 스킵
            if len(ex) < 5: continue
            
            # 예시 문장을 저장하되, 이게 어떤 밈인지 알 수 있게 메타데이터는 동일하게 줌
            ex_id = f"{item['id']}_ex_{i}"
            
            # Q/A 형태면 A(답변)만 쓰는 게 문장 검색에 더 유리할 수 있음
            # 여기서는 전체를 다 넣되, 앞에 키워드 힌트를 살짝 줌
            ex_text = f"{keyword} 예시: {ex}" 
            
            ids.append(ex_id)
            documents.append(ex_text)
            metadatas.append(base_meta)

    print(f"   - 총 저장할 벡터 수: {len(ids)}개 (정의 + 예시)")

    # 배치 저장
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i+batch_size],
            documents=documents[i : i+batch_size],
            metadatas=metadatas[i : i+batch_size]
        )
        
    print(f"\n✅ VectorDB 구축 완료! 위치: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    build_database()
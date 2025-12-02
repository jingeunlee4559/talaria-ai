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

# ⭐️ [수정] 입력 파일을 'labeled' 버전으로 변경
DATA_FILE = os.path.join(BASE_DIR, "meme_dictionary_labeled.json") 

DB_PATH = "src/talaria/assets/vectordb"
COLLECTION_NAME = "meme_dictionary"

EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-multitask"

def build_database():
    # 1. 데이터 로드
    if not os.path.exists(DATA_FILE):
        print(f"❌ 오류: 데이터 파일이 없습니다.\n경로: {DATA_FILE}")
        print("💡 팁: 1단계 스크립트(add_category.py)를 먼저 실행해주세요.")
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
    
    # 기존 컬렉션 초기화 (재생성)
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
        
        # ⭐️ [수정] 메타데이터에 'category' 추가!
        # 여기서 저장해야 나중에 검색할 때 꺼내쓸 수 있음
        base_meta = {
            "primary_keyword": keyword,
            "definition": definition,
            "tags": ", ".join(item.get('tags', [])),
            "source": item.get('source', 'unknown'),
            "category": item.get('category', 'k-meme') # 기본값 설정
        }

        # (1) 정의 데이터 저장
        def_id = f"{item['id']}_def"
        def_text = f"{keyword}: {definition}"
        
        ids.append(def_id)
        documents.append(def_text)
        metadatas.append(base_meta)
        
        # (2) 예시 문장 데이터 저장
        for i, ex in enumerate(examples):
            if len(ex) < 5: continue
            
            ex_id = f"{item['id']}_ex_{i}"
            ex_text = f"{keyword} 예시: {ex}" 
            
            ids.append(ex_id)
            documents.append(ex_text)
            metadatas.append(base_meta)

    print(f"   - 총 저장할 벡터 수: {len(ids)}개 (정의 + 예시)")

    # 배치 저장 (ChromaDB 권장사항)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i+batch_size],
            documents=documents[i : i+batch_size],
            metadatas=metadatas[i : i+batch_size]
        )
        print(f"   - 저장 중... {i+len(ids[i:i+batch_size])}/{len(ids)}", end='\r')
        
    print(f"\n✅ VectorDB 구축 완료! 위치: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    build_database()
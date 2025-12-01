import json
import os
import chromadb
# [추가] 구체적인 에러를 잡기 위해 모듈 임포트
try:
    from chromadb import errors as chroma_errors
except ImportError:
    chroma_errors = None # 구버전 호환성

from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 경로 설정 (엄수)
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
    print(f"   - 총 데이터 개수: {len(data)}개")

    # 2. 폴더 생성
    os.makedirs(DB_PATH, exist_ok=True)

    # 3. ChromaDB 클라이언트 설정
    print(f">>> VectorDB 생성 경로: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # [수정됨] 안전한 초기화 로직
    print(f"   - 기존 컬렉션 '{COLLECTION_NAME}' 확인 중...")
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"   - 🗑️ 기존 컬렉션 삭제 완료 (초기화)")
    except Exception as e:
        # 에러 메시지에 'does not exist'나 'Found'가 포함되어 있으면 "없어서 못 지운 것"이니 안전함
        error_msg = str(e)
        if "does not exist" in error_msg or "NotFoundError" in str(type(e)):
            print(f"   - (기존 컬렉션이 없어서 새로 만듭니다)")
        else:
            # 진짜 위험한 에러는 출력하고 중단!
            print(f"❌ [치명적 오류] 컬렉션 삭제 중 예상치 못한 에러 발생!")
            print(f"   에러 내용: {e}")
            raise e # 프로그램 중단

    # 4. 임베딩 모델 로드
    print(f">>> 임베딩 모델 로드 중 ({EMBEDDING_MODEL_NAME})...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
        def __call__(self, input):
            return embedding_model.encode(input).tolist()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=CustomEmbeddingFunction()
    )

    # 5. 데이터 저장
    ids = []
    documents = [] 
    metadatas = []

    print(">>> 데이터 벡터화 및 저장 시작...")
    
    for item in data:
        if 'primary_keyword' not in item:
            continue
            
        doc_id = item.get('id', f"meme_{len(ids)}")
        
        # 검색용 텍스트
        search_text = item.get('embedding_text')
        if not search_text:
            search_text = f"{item['primary_keyword']}: {item.get('definition', '')}"

        ids.append(doc_id)
        documents.append(search_text)
        
        # 메타데이터 (리스트 -> 문자열 변환 필수)
        meta = {
            "primary_keyword": item['primary_keyword'],
            "definition": item.get('definition', ''),
            "tags": ", ".join(item.get('tags', [])) if isinstance(item.get('tags'), list) else str(item.get('tags', '')),
            "related_keywords": ", ".join(item.get('related_keywords', [])) if isinstance(item.get('related_keywords'), list) else str(item.get('related_keywords', '')),
            "source": item.get('source', 'unknown')
        }
        metadatas.append(meta)

    # 배치 저장
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        try:
            collection.add(
                ids=ids[i : i+batch_size],
                documents=documents[i : i+batch_size],
                metadatas=metadatas[i : i+batch_size]
            )
            print(f"   - {min(i+batch_size, len(ids))}/{len(ids)} 저장됨")
        except Exception as e:
            print(f"⚠️ 저장 중 에러 발생 (Skipping batch {i}): {e}")

    print(f"\n✅ VectorDB 구축 완료! 위치: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    build_database()
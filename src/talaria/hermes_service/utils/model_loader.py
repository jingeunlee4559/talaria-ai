# src/talaria/hermes_service/utils/model_loader.py

import os
import json # json 임포트 추가
import torch
import chromadb
from transformers import AutoTokenizer, ElectraModel, AutoConfig, ElectraPreTrainedModel
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import numpy as np

# ==========================================
# 1. 설정 (경로)
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MODEL_PATH = os.path.join(BASE_DIR, "src/talaria/assets/model")
DB_PATH = os.path.join(BASE_DIR, "src/talaria/assets/vectordb")
# ⭐️ [추가] 족보 데이터 경로
DATA_FILE = os.path.join(BASE_DIR, "model/data_collecting/meme_dictionary_complete.json")

COLLECTION_NAME = "meme_dictionary"
EMBEDDING_MODEL = "jhgan/ko-sbert-multitask"

# ==========================================
# 2. 모델 클래스 (학습 때와 동일 구조)
# ==========================================
class MemeMultiTaskModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_cls = 4 
        self.num_labels_ner = 3
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels_cls)
        self.ner_classifier = nn.Linear(config.hidden_size, self.num_labels_ner)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids)
        sequence_output = discriminator_hidden_states[0]
        pooled_output = sequence_output[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        logits_cls = self.classifier(pooled_output)
        logits_ner = self.ner_classifier(sequence_output)
        return logits_cls, logits_ner

# ==========================================
# 3. 추론기 (Scanner)
# ==========================================
class MemeScanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> [Loader] 모델 로딩 중... (Device: {self.device})")
        
        if not os.path.exists(MODEL_PATH):
             print(f"❌ [Error] 모델 경로 없음: {MODEL_PATH}")
             # 에러를 띄우지 않고 넘어가되, 기능이 작동 안 하도록 처리 (선택 사항)
             self.model = None
             return

        try:
            self.config = AutoConfig.from_pretrained(MODEL_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = MemeMultiTaskModel.from_pretrained(MODEL_PATH, config=self.config)
            self.model.to(self.device)
            self.model.eval()
            print(">>> [Loader] 모델 로딩 완료!")
        except Exception as e:
            print(f"❌ [Error] 모델 로드 실패: {e}")
            self.model = None

        self.id2label = {0: "SARCASM", 1: "ADMIRATION", 2: "SHOCK", 3: "NEUTRAL"}
        self.id2ner = {0: "O", 1: "B-MEME", 2: "I-MEME"}

    def predict(self, current_text, prev_text=""):
        if not self.model:
            return {"meme_words": [], "nuance": "NEUTRAL", "score": 0.0}

        context = f" | {prev_text}" 
        inputs = self.tokenizer(
            context, current_text, return_tensors="pt", truncation=True, max_length=128, return_offsets_mapping=True
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs["token_type_ids"].to(self.device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()
        
        with torch.no_grad():
            logits_cls, logits_ner = self.model(input_ids, attention_mask, token_type_ids)
        
        # 1. 뉘앙스
        probs_cls = torch.softmax(logits_cls, dim=1)
        top_prob, top_label_id = torch.max(probs_cls, dim=1)
        nuance_label = self.id2label[top_label_id.item()]
        confidence = top_prob.item()
        
        # 2. 밈 추출
        preds_ner = torch.argmax(logits_ner, dim=2)[0].cpu().numpy()
        meme_words = []
        curr_start, curr_end = -1, -1
        
        for idx, pred in enumerate(preds_ner):
            if token_type_ids[0][idx] != 1: continue
            tag = self.id2ner[pred]
            start, end = offset_mapping[idx]
            if start == end: continue

            if tag == "B-MEME":
                if curr_start != -1: meme_words.append(current_text[curr_start:curr_end])
                curr_start, curr_end = start, end
            elif tag == "I-MEME":
                if curr_start != -1: curr_end = end
                else: curr_start, curr_end = start, end 
            else:
                if curr_start != -1:
                    meme_words.append(current_text[curr_start:curr_end])
                    curr_start, curr_end = -1, -1
                    
        if curr_start != -1: meme_words.append(current_text[curr_start:curr_end])
        meme_words = list(set([w.strip() for w in meme_words if w.strip()]))
        
        return {"meme_words": meme_words, "nuance": nuance_label, "score": confidence}

# ==========================================
# 4. 검색기 (Retriever)
# ==========================================
class MemeRetriever:
    def __init__(self):
        print(f">>> [Loader] VectorDB 로딩 중... ({DB_PATH})")
        if not os.path.exists(DB_PATH):
            print(f"❌ [Error] DB 경로 없음: {DB_PATH}")
            self.collection = None
            return
            
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        
        class EmbedFn(chromadb.EmbeddingFunction):
            def __init__(self, model): self.model = model
            def __call__(self, input): 
                return self.model.encode(input, normalize_embeddings=True).tolist()
            
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME, 
                embedding_function=EmbedFn(self.embed_model)
            )
            print(">>> [Loader] VectorDB 로딩 완료!")
        except Exception as e:
            print(f"❌ [Error] 컬렉션 로드 실패: {e}")
            self.collection = None

    def search(self, queries, top_k=1):
        if not self.collection or not queries: return []
        
        try:
            results = self.collection.query(query_texts=queries, n_results=top_k)
            valid_results = []
            # 거리 기준 (1.0 미만이면 유사)
            for i, dist in enumerate(results['distances'][0]):
                if dist < 1.1: # 조금 여유 있게
                    valid_results.append(results['metadatas'][0][i])
            return valid_results
        except:
            return []

# ==========================================
# 5. 싱글톤 로더 (수정됨)
# ==========================================
class GlobalModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalModelLoader, cls).__new__(cls)
            cls._instance.scanner = None
            cls._instance.retriever = None
            cls._instance.meme_dict = {} # ⭐️ 족보 저장소
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        if self.initialized: return
        print(">>> [Loader] 리소스 초기화 시작...")
        
        self.scanner = MemeScanner()
        self.retriever = MemeRetriever()
        
        # ⭐️ [추가] JSON 족보 로딩 (Keyword Matching용)
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 검색 속도를 위해 {키워드: 데이터} 형태로 변환
                    for item in data:
                        # 메인 키워드
                        if 'primary_keyword' in item:
                            self.meme_dict[item['primary_keyword']] = item
                        # 연관 키워드 (유사어)도 다 매핑
                        for rel in item.get('related_keywords', []):
                            self.meme_dict[rel] = item
                print(f">>> [Loader] 밈 족보 로딩 완료 ({len(self.meme_dict)}개 키워드)")
            else:
                print(f"⚠️ [Loader] 족보 파일 없음: {DATA_FILE}")
        except Exception as e:
            print(f"❌ [Loader] 족보 로딩 실패: {e}")

        self.initialized = True

loader = GlobalModelLoader()

def get_scanner():
    if not loader.initialized: loader.initialize()
    return loader.scanner

def get_retriever():
    if not loader.initialized: loader.initialize()
    return loader.retriever

# ⭐️ [추가] 족보 가져오는 함수
def get_meme_dictionary():
    if not loader.initialized: loader.initialize()
    return loader.meme_dict
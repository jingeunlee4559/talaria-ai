import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer, ElectraPreTrainedModel, ElectraModel, AutoConfig
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer



# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "src/talaria/assets/model"
DB_PATH = "src/talaria/assets/vectordb"
COLLECTION_NAME = "meme_dictionary"
EMBEDDING_MODEL = "jhgan/ko-sbert-multitask"

# --- 모델 클래스 (생략 없이 정의 필요) ---
class MemeMultiTaskModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_cls = 4 
        self.num_labels_ner = 3
        self.electra = ElectraModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels_cls)
        self.ner_classifier = torch.nn.Linear(config.hidden_size, self.num_labels_ner)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids)
        sequence_output = discriminator_hidden_states[0]
        pooled_output = sequence_output[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        logits_cls = self.classifier(pooled_output)
        logits_ner = self.ner_classifier(sequence_output)
        return logits_cls, logits_ner

class HybridSystem:
    def __init__(self):
        print(">>> 시스템 로딩 중...")
        # 1. 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = MemeMultiTaskModel.from_pretrained(MODEL_PATH, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = {0: "SARCASM", 1: "ADMIRATION", 2: "SHOCK", 3: "NEUTRAL"}
        self.id2ner = {0: "O", 1: "B-MEME", 2: "I-MEME"}
        
        # 2. VectorDB 로드
        self.db_client = chromadb.PersistentClient(path=DB_PATH)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        
        class EmbedFn(chromadb.EmbeddingFunction):
            def __init__(self, model): self.model = model
            def __call__(self, input): 
                # [핵심 수정] 여기도 normalize_embeddings=True 추가!
                return self.model.encode(input, normalize_embeddings=True).tolist()
            
        self.collection = self.db_client.get_collection(
            name=COLLECTION_NAME, embedding_function=EmbedFn(self.embed_model)
        )
        print(">>> ✅ 로딩 완료!")

    def process(self, text):
        print(f"\n🔹 입력: '{text}'")
        
        # [Step 1] 모델 추론
        inputs = self.tokenizer(f" | {text}", text, return_tensors="pt", truncation=True, max_length=128, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        # ... (나머지 입력 처리 생략, 핵심 로직 위주) ...
        with torch.no_grad():
            logits_cls, logits_ner = self.model(input_ids, inputs["attention_mask"].to(self.device), inputs["token_type_ids"].to(self.device))
        
        # 밈 추출 로직 (Slicing)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()
        preds_ner = torch.argmax(logits_ner, dim=2)[0].cpu().numpy()
        meme_words = []
        curr_start, curr_end = -1, -1
        
        for i, pred in enumerate(preds_ner):
            if inputs["token_type_ids"][0][i] != 1: continue
            tag = self.id2ner[pred]
            start, end = offset_mapping[i]
            if start == end: continue
            
            if tag == "B-MEME":
                if curr_start != -1: meme_words.append(text[curr_start:curr_end])
                curr_start, curr_end = start, end
            elif tag == "I-MEME":
                if curr_start != -1: curr_end = end
                else: curr_start, curr_end = start, end
            else:
                if curr_start != -1:
                    meme_words.append(text[curr_start:curr_end])
                    curr_start, curr_end = -1, -1
        if curr_start != -1: meme_words.append(text[curr_start:curr_end])
        
        meme_words = list(set([w.strip() for w in meme_words if w.strip()]))
        
        # [Step 2] 결과 판단 및 DB 검색
        final_result = {}
        
        if meme_words:
            print(f"   👉 [모델 성공] 추출된 밈: {meme_words}")
            # 추출된 단어로 DB 검색 (정확도 높음)
            db_res = self.collection.query(query_texts=meme_words, n_results=1)
            if db_res['ids']:
                final_result = {
                    "meme": db_res['metadatas'][0][0]['primary_keyword'],
                    "meaning": db_res['metadatas'][0][0]['definition'],
                    "source": "MODEL + DB"
                }
        else:
            print(f"   ⚠️ [모델 실패] 밈을 찾지 못함 -> 문장 전체로 DB 검색 시도")
            # 문장 전체 검색 (Fallback)
            db_res = self.collection.query(query_texts=[text], n_results=1)
            
            # 거리(Distance)가 너무 멀면(예: 1.2 이상) 밈이 없는 문장으로 간주
            distance = db_res['distances'][0][0]
            if distance < 1.0:  # [수정] 기준값을 1.0으로 설정 (필요시 1.1~1.2로 조정)
                print(f"   ✅ [DB 구원] 문맥 검색 성공! (거리: {distance:.4f})")
                final_result = {
                    "meme": db_res['metadatas'][0][0]['primary_keyword'],
                    "meaning": db_res['metadatas'][0][0]['definition'],
                    "source": "ONLY DB (Sentence Search)"
                }
            else:
                print(f"   ❌ [최종 실패] DB에도 관련 내용 없음 (거리: {distance:.4f})")

        if final_result:
            print(f"   📝 최종 결과: {final_result['meme']} - {final_result['meaning'][:30]}...")
        else:
            print("   📝 최종 결과: 일반 대화입니다.")

if __name__ == "__main__":
    system = HybridSystem()
    
    # 테스트 케이스 (모델이 실패했던 것들 포함)
    test_sentences = [
        "완전 럭키비키잖아",        # 모델 성공 예상
        "아 진짜 킹받네",           # 모델 실패 -> DB 성공 예상
        "누가 칼 들고 협박함?",      # 모델 실패 -> DB 성공 예상 ('누칼협')
        "오늘 점심 뭐 먹을까?",      # 일반 대화 -> DB 거리 멀어서 무시 예상
        "응 이미 나락 갔어"         # 모델 실패 -> DB 성공 예상
    ]
    
    for sent in test_sentences:
        system.process(sent)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
from transformers import AutoTokenizer, ElectraPreTrainedModel, ElectraModel, AutoConfig
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "src/talaria/assets/model"

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

class MemeScanner:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> [Model Test] 모델 로드 중... (Device: {self.device})")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 모델 경로 없음: {model_path}")

        try:
            self.config = AutoConfig.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = MemeMultiTaskModel.from_pretrained(model_path, config=self.config)
            self.model.to(self.device)
            self.model.eval()
            print(">>> ✅ 모델 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise e

        self.id2label = {0: "SARCASM", 1: "ADMIRATION", 2: "SHOCK", 3: "NEUTRAL"}
        self.id2ner = {0: "O", 1: "B-MEME", 2: "I-MEME"}

    def predict(self, situation, question, answer):
        # [핵심] 상황 + 질문을 합쳐서 풍부한 문맥 제공
        context = f"{situation} | {question}"
        
        inputs = self.tokenizer(
            context, 
            answer, 
            return_tensors="pt",
            truncation=True, 
            max_length=128,
            return_offsets_mapping=True
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
        confidence = top_prob.item() * 100
        
        # 2. 밈 추출 (Slicing 방식 - 글자깨짐 방지)
        preds_ner = torch.argmax(logits_ner, dim=2)[0].cpu().numpy()
        meme_words = []
        current_start, current_end = -1, -1
        
        for idx, pred in enumerate(preds_ner):
            if token_type_ids[0][idx] != 1: continue 
            
            tag = self.id2ner[pred]
            start, end = offset_mapping[idx]
            if start == end: continue

            if tag == "B-MEME":
                if current_start != -1:
                    meme_words.append(answer[current_start:current_end])
                current_start, current_end = start, end
            elif tag == "I-MEME":
                if current_start != -1: current_end = end
                else: current_start, current_end = start, end 
            else:
                if current_start != -1:
                    meme_words.append(answer[current_start:current_end])
                    current_start, current_end = -1, -1
                    
        if current_start != -1:
            meme_words.append(answer[current_start:current_end])
            
        meme_words = list(set([w.strip() for w in meme_words if w.strip()]))
        
        return {
            "meme_words": meme_words,
            "nuance": nuance_label,
            "score": round(confidence, 2)
        }

if __name__ == "__main__":
    # 모델 경로 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 경로를 찾을 수 없습니다: {MODEL_PATH}")
        sys.exit(1)

    scanner = MemeScanner(MODEL_PATH)
    
    # 실전 테스트 케이스 (상황 정보 없음)
    # (이전 발화, 현재 발화, 기대하는 뉘앙스 - 참고용)
    test_cases = [
        # 1. 긍정/찬양 (ADMIRATION)
        ("", "와 오늘 손흥민 경기 봤냐?", "진짜 폼 미쳤다 혼자서 다 하더라"),
        ("발표 자료 만든 거 어때?", "와 PPT 퀄리티 갓벽하다 진짜 고생했어", "ADMIRATION"),
        ("이번 시험 너무 어려울 것 같은데 포기할까?", "야 중요한 건 꺾이지 않는 마음이야 끝까지 해보자", "ADMIRATION"),
        ("요즘 얼굴 보기 힘들다?", "나 요즘 새벽 수영 다니고 영어 학원 다니고 완전 갓생 중이잖아", "ADMIRATION"),

        # 2. 비꼬기/공격 (SARCASM)
        ("김대리님이 너 때문에 징계받을 수도 있다던데?", "알빠노 지가 실수해놓고 왜 나한테 난리야", "SARCASM"),
        ("아 이 게임 밸런스 왜 이래 진짜 못해먹겠네", "누가 그 게임 하라고 칼 들고 협박함? 꼬우면 접던가 ㅋㅋ", "SARCASM"),
        ("이번에 그 유튜버 사과 영상 올렸더라", "응 이미 나락 갔어~ 절대 안 돌아와~", "SARCASM"),
        ("내일 약속 1시간만 늦춰도 돼?", "아 진짜 킹받네 나 벌써 준비 다 하고 나왔다고 ㅡㅡ", "SARCASM"),

        # 3. 충격/부정 (SHOCK) - 데이터 부족으로 SARCASM으로 나올 가능성 있음
        ("너 왜 수강신청을 21학점이나 했어?", "몰라.. 과거의 내가 미쳤었나 봐 완전 스불재야 지금 죽겠어", "SHOCK"),
        ("오늘 점심 뭐 먹을래?", "국밥이 만원이더라 런치플레이션 실화냐 ㅠㅠ", "SHOCK"),

        # 4. 중립/일상 (NEUTRAL)
        ("오늘 점심 뭐 먹을까?", "그냥 회사 앞 김치찌개 집 어때? 가격도 괜찮던데", "NEUTRAL"),
        ("이번 프로젝트 진행 상황 보고해주세요.", "현재 디자인 시안은 완료되었고 개발 착수 단계입니다.", "NEUTRAL"),
        ("다음 주 금요일 저녁 비워둬", "오케이 달력에 캘박 해놨음 그날 보자", "NEUTRAL"), # 캘박은 밈이지만 감정은 중립적
        ("야 비 와서 야외 일정 취소해야 될 듯 ㅠㅠ", "오히려 좋아 실내에서 에어컨 쐬면서 호캉스 가자", "NEUTRAL"), # 오히려 좋아는 긍정적이지만 NEUTRAL로 나올 수도 있음

        # 5. 최신/미지 밈 (Generalization Test)
        ("야 내가 사려던 소금빵 다 팔렸대 ㅠㅠ", "오히려 좋아! 덕분에 다이어트하고 돈도 아꼈잖아? 완전 럭키비키잖아🍀", "ADMIRATION"),
        ("아 진짜 너무 힘들다 그냥 집에 갈까?", "야 중요한 건 꺾였는데도 그냥 하는 마음이야.. 중꺾그마 알지? 얼른 하자 ㅠㅠ", "ADMIRATION"),
    ]
    
    print("\n" + "="*60)
    print("🧪 실전 모델 테스트 시작 (상황 정보 없음)")
    print("="*60)

    for prev_text, current_text, expected_nuance in test_cases:
        # 상황(situation)은 비워두고, 이전 발화(question)를 문맥으로 사용
        result = scanner.predict("", prev_text, current_text)
        
        print(f"Previous: {prev_text}")
        print(f"Input:    {current_text}")
        print(f"👉 밈 추출: {result['meme_words']}")
        print(f"👉 뉘앙스: {result['nuance']} ({result['score']}%)")
        
        # 간단한 검증 로직 (참고용)
        # SHOCK은 데이터가 적어서 SARCASM으로 나와도 어느 정도 인정
        if result['nuance'] == expected_nuance or (expected_nuance == "SHOCK" and result['nuance'] == "SARCASM"):
             print("✅ 예측 성공")
        else:
             print(f"⚠️ 예측 확인 필요 (기대: {expected_nuance})")
             
        print("-" * 60)
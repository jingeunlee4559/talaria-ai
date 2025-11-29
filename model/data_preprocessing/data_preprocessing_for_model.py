import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import os
import ast

# ==========================================
# 1. 설정
# ==========================================
# 라벨링이 완료된 파일 경로 (라벨링 끝나면 이 경로에 파일이 있겠죠?)
INPUT_FILE = "model/data_preprocessing/final_labeled_dataset.csv"
OUTPUT_DIR = "model/train_data"

# 사용할 모델 (토크나이저용)
MODEL_NAME = "beomi/KcELECTRA-base"

# 라벨 맵핑 (우리가 정한 4가지 감정)
label2id = {
    "SARCASM": 0,
    "ADMIRATION": 1,
    "SHOCK": 2,
    "NEUTRAL": 3
}

# NER 태그 맵핑 (0:아님, 1:시작, 2:중간)
ner_label2id = {"O": 0, "B-MEME": 1, "I-MEME": 2}

def preprocess_data():
    # 1. 파일 로드
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 파일이 아직 없습니다. 라벨링이 끝나면 실행하세요.")
        return

    print(f">>> 데이터 로드 중: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # 품질 필터링: Score가 3점 이상인 것만 학습 (데이터 퀄리티 업!)
    # score 컬럼이 문자열일 수도 있어서 숫자로 변환
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    original_len = len(df)
    df = df[df['score'] >= 3].copy()
    
    # 라벨 이상한 거 제거
    df = df[df['label'].isin(label2id.keys())].copy()
    
    print(f">>> 데이터 필터링: {original_len}개 -> {len(df)}개 (Score 3점 미만 제외)")

    # 2. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    processed_samples = []
    
    print(">>> 전처리(토큰화 & NER 태깅) 시작...")
    
    for idx, row in df.iterrows():
        try:
            # -----------------------------------------------------------
            # [핵심 로직] 문자를 토큰으로 바꾸고, 밈 단어 위치 찾기
            # -----------------------------------------------------------
            meme_word = str(row['meme_word'])
            target_text = str(row['target_a']) # 답변 문장
            
            # 문맥 정보 합치기: "상황 | 질문"
            context_text = f"{row.get('situation', '')} | {row.get('context_q', '')}"
            
            # 토크나이징 (정답 문장 target_text를 text_pair로 넣음)
            # return_offsets_mapping=True: 각 토큰이 원문의 몇 번째 글자인지 알려줌
            tokenized = tokenizer(
                context_text,
                target_text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_offsets_mapping=True
            )
            
            # 밈 단어가 답변 문장(target_text) 안에서 어디 있는지 찾기 (글자 기준)
            start_char = target_text.find(meme_word)
            
            ner_labels = []
            
            if start_char == -1:
                # 못 찾으면 학습 데이터로 쓰기 애매하므로 건너뜀 (혹은 전체 O 처리)
                continue
                
            end_char = start_char + len(meme_word)
            
            # 토큰별로 루프를 돌면서 태그 달기
            # sequence_ids: 0은 상황/질문, 1은 답변, None은 특수토큰
            sequence_ids = tokenized.sequence_ids()
            offsets = tokenized["offset_mapping"]
            
            for i, (start, end) in enumerate(offsets):
                # 답변(1)이 아니거나 특수토큰이면 무시 (-100)
                if sequence_ids[i] != 1:
                    ner_labels.append(-100)
                    continue
                
                # 이 토큰이 밈 단어 범위 안에 있는가?
                # 주의: KcELECTRA 토크나이저 오프셋 보정 필요할 수 있음 (일단 기본 로직 적용)
                if start >= start_char and start < end_char:
                    if start == start_char:
                        ner_labels.append(ner_label2id["B-MEME"]) # 시작
                    else:
                        ner_labels.append(ner_label2id["I-MEME"]) # 중간
                else:
                    ner_labels.append(ner_label2id["O"])      # 아님

            # 결과 저장
            processed_samples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "token_type_ids": tokenized["token_type_ids"],
                "labels": label2id[row['label']], # 감정 라벨
                "ner_labels": ner_labels          # 위치 라벨
            })
            
        except Exception as e:
            continue

    print(f">>> 전처리 완료: 총 {len(processed_samples)}개 샘플 생성")
    
    # 3. Train / Val 나누기 및 저장
    final_df = pd.DataFrame(processed_samples)
    train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_pickle(os.path.join(OUTPUT_DIR, "train.pkl"))
    val_df.to_pickle(os.path.join(OUTPUT_DIR, "val.pkl"))
    
    print(f">>> 저장 완료! 폴더 확인: {OUTPUT_DIR}")
    print(f"    - Train: {len(train_df)}개")
    print(f"    - Val: {len(val_df)}개")

if __name__ == "__main__":
    preprocess_data()
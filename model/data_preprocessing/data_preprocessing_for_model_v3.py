import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. 설정
# ==========================================
INPUT_FILE = "model/data_preprocessing/final_labeled_dataset.csv"
OUTPUT_DIR = "model/train_data"
MODEL_NAME = "beomi/KcELECTRA-base"

label2id = {"SARCASM": 0, "ADMIRATION": 1, "SHOCK": 2, "NEUTRAL": 3}
ner_label2id = {"O": 0, "B-MEME": 1, "I-MEME": 2}

def find_meme_span(target_text, meme_word):
    # (이전과 동일한 유연한 검색 함수)
    meme_word = str(meme_word).strip()
    target_text = str(target_text)
    
    start = target_text.find(meme_word)
    if start != -1:
        return start, start + len(meme_word)
    
    min_len = 2 
    if len(meme_word) <= min_len: return -1, -1

    current_word = meme_word
    while len(current_word) > min_len:
        current_word = current_word[:-1]
        start = target_text.find(current_word)
        if start != -1:
            return start, start + len(current_word)
            
    return -1, -1

def preprocess_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 없음")
        return

    print(f">>> 데이터 로드 중...")
    df = pd.read_csv(INPUT_FILE)
    
    # 필터링
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df[df['score'] >= 3].copy()
    df = df[df['label'].isin(label2id.keys())].copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processed_samples = []
    
    success_count = 0
    
    print(">>> 전처리(Overlap 로직 적용) 시작...")
    
    for idx, row in df.iterrows():
        try:
            meme_word = row['meme_word']
            target_text = str(row['target_a'])
            
            # 1. 밈 위치 찾기
            start_char, end_char = find_meme_span(target_text, meme_word)
            if start_char == -1: continue # 못 찾으면 스킵
            
            # 2. 토크나이징
            context_text = f"{row.get('situation', '')} | {row.get('context_q', '')}"
            tokenized = tokenizer(
                context_text,
                target_text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_offsets_mapping=True
            )
            
            ner_labels = []
            sequence_ids = tokenized.sequence_ids()
            offsets = tokenized["offset_mapping"]
            
            # [핵심 수정] B-MEME 태그가 이미 등장했는지 체크하는 플래그
            found_start_token = False

            for i, (start, end) in enumerate(offsets):
                # 답변 문장(ID=1)이 아니면 무시
                if sequence_ids[i] != 1:
                    ner_labels.append(-100)
                    continue
                
                # 특수 토큰(SEP 등)은 오프셋이 (0,0)임 -> 무시
                if start == 0 and end == 0:
                    ner_labels.append(-100)
                    continue

                # -------------------------------------------------------
                # [수정됨] 겹치기(Overlap) 로직
                # 토큰 구간(start~end)과 밈 구간(start_char~end_char)이 교집합이 있는지 확인
                # -------------------------------------------------------
                # 교집합 길이 계산
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                
                if overlap_start < overlap_end: # 겹치는 부분이 있다면 태깅
                    if not found_start_token:
                        ner_labels.append(ner_label2id["B-MEME"]) # 첫 겹침 -> 시작
                        found_start_token = True
                    else:
                        ner_labels.append(ner_label2id["I-MEME"]) # 그 이후 -> 중간
                else:
                    ner_labels.append(ner_label2id["O"])

            processed_samples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "token_type_ids": tokenized["token_type_ids"],
                "labels": label2id[row['label']],
                "ner_labels": ner_labels
            })
            success_count += 1
            
        except Exception:
            continue

    print(f">>> 전처리 완료: 총 {len(processed_samples)}개 (성공: {success_count}개)")
    
    final_df = pd.DataFrame(processed_samples)
    train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_pickle(os.path.join(OUTPUT_DIR, "train.pkl"))
    val_df.to_pickle(os.path.join(OUTPUT_DIR, "val.pkl"))
    print(">>> 저장 완료.")

if __name__ == "__main__":
    preprocess_data()
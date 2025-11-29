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

# ==========================================
# 2. 강력한 밈 단어 찾기 함수 (핵심 추가)
# ==========================================
def find_meme_span(target_text, meme_word):
    """
    문장에서 밈 단어의 위치(start, end)를 찾습니다.
    활용형(conjugation)을 고려하여, 뒤에서부터 한 글자씩 줄여가며 매칭을 시도합니다.
    예: meme='어마무시하다', text='어마무시하네' -> '어마무시'로 매칭 성공
    """
    meme_word = str(meme_word).strip()
    target_text = str(target_text)
    
    # 1. 원본 그대로 찾아보기
    start = target_text.find(meme_word)
    if start != -1:
        return start, start + len(meme_word)
    
    # 2. 글자 수 줄여가며 찾기 (최소 2글자까지)
    # 예: 어마무시하다 -> 어마무시하 -> 어마무시 (매칭!)
    min_len = 2 
    # 단어 자체가 2글자 미만이면(예: '각') 더 줄일 수 없으므로 포기
    if len(meme_word) <= min_len:
        return -1, -1

    current_word = meme_word
    while len(current_word) > min_len:
        current_word = current_word[:-1] # 뒤에 한 글자 자르기
        start = target_text.find(current_word)
        if start != -1:
            return start, start + len(current_word)
            
    return -1, -1

# ==========================================
# 3. 전처리 메인 함수
# ==========================================
def preprocess_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 파일이 없습니다.")
        return

    print(f">>> 데이터 로드 중: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # 데이터 필터링
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    original_len = len(df)
    df = df[df['score'] >= 3].copy()
    df = df[df['label'].isin(label2id.keys())].copy()
    
    print(f">>> 데이터 필터링: {original_len}개 -> {len(df)}개 (Score 3점 미만 제외)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processed_samples = []
    
    success_count = 0
    fail_count = 0
    
    print(">>> 전처리(토큰화 & 유연한 NER 태깅) 시작...")
    
    for idx, row in df.iterrows():
        try:
            meme_word = row['meme_word']
            target_text = str(row['target_a'])
            
            # -----------------------------------------------------------
            # [수정됨] 유연한 검색 함수 사용
            # -----------------------------------------------------------
            start_char, end_char = find_meme_span(target_text, meme_word)
            
            # 그래도 못 찾으면 어쩔 수 없이 Skip (완전히 다른 단어로 변형된 경우)
            if start_char == -1:
                fail_count += 1
                continue
            
            # 문맥 정보 합치기
            context_text = f"{row.get('situation', '')} | {row.get('context_q', '')}"
            
            # 토크나이징
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
            
            for i, (start, end) in enumerate(offsets):
                if sequence_ids[i] != 1:
                    ner_labels.append(-100)
                    continue
                
                # 찾은 위치(start_char, end_char)를 기준으로 태깅
                if start >= start_char and start < end_char:
                    if start == start_char:
                        ner_labels.append(ner_label2id["B-MEME"])
                    else:
                        ner_labels.append(ner_label2id["I-MEME"])
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
            
        except Exception as e:
            fail_count += 1
            continue

    print(f">>> 전처리 완료: 총 {len(processed_samples)}개 샘플 생성")
    print(f">>> (성공: {success_count}개, 실패/건너뜀: {fail_count}개)")
    
    # 저장
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
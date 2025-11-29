import pandas as pd
import json
import os

# ==========================================
# 1. 설정
# ==========================================
INPUT_CSV = "model/data_preprocessing/final_labeled_dataset.csv"  # 업로드한 CSV 파일명
OUTPUT_JSON = "model/data_collecting/meme_dictionary_final.json"

def convert_csv_to_json():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 오류: '{INPUT_CSV}' 파일이 없습니다.")
        return

    print(f">>> 데이터 로드 중: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 데이터가 잘 들어왔는지 확인
    print(f"   - 총 데이터 행 수: {len(df)}개")

    # 2. 단어별로 그룹화 (Grouping)
    # meme_word가 같은 것끼리 묶습니다.
    grouped = df.groupby('meme_word')
    
    json_list = []
    
    print(f">>> 변환 작업 시작 (총 {len(grouped)}개의 밈 단어)...")

    # enumerate로 ID 생성을 위해 인덱스 사용
    for idx, (word, group) in enumerate(grouped):
        # (1) 기본 정보 추출 (첫 번째 행의 정보 사용)
        first_row = group.iloc[0]
        definition = str(first_row['meaning']).strip()
        
        # (2) 태그 수집 (중복 제거)
        # 해당 단어에 달린 모든 label을 모아서 리스트로 만듦
        tags = group['label'].unique().tolist()
        # 혹시 Nan이나 빈 값이 있으면 제거
        tags = [t for t in tags if pd.notna(t) and t != "ERROR"]
        
        # (3) 예시 문장 수집 (최대 5개)
        examples = []
        for _, row in group.head(5).iterrows():
            q = str(row['context_q']).strip()
            a = str(row['target_a']).strip()
            
            # 질문이 있으면 Q/A 형태로, 없으면 A만 저장
            if q and q != "nan":
                example_text = f"Q: {q} / A: {a}"
            else:
                example_text = a
            examples.append(example_text)

        # (4) 임베딩용 텍스트 생성
        # 검색이 잘 되도록 "단어: 뜻" 형태로 만듭니다.
        embedding_text = f"{word}: {definition}"

        # (5) 최종 딕셔너리 생성
        meme_entry = {
            "id": f"meme_{idx:04d}",
            "primary_keyword": word,
            "related_keywords": [], # CSV에는 없으므로 빈 리스트 (추후 LLM으로 채움)
            "definition": definition,
            "tags": tags,
            "examples": examples,
            "embedding_text": embedding_text,
            "source": "final_labeled_dataset"
        }
        
        json_list.append(meme_entry)

    # 3. JSON 파일로 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 변환 완료!")
    print(f"   - 저장 경로: {OUTPUT_JSON}")
    print(f"   - 총 밈 단어 수: {len(json_list)}개")
    
    # 샘플 출력
    if json_list:
        print("\n[샘플 데이터 확인]")
        print(json.dumps(json_list[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    convert_csv_to_json()
import pandas as pd
import json
import os

ORIGINAL_CSV = "model/data_preprocessing/final_labeled_dataset.csv"
OUTPUT_JSONL = "model/data_preprocessing/batch_output.jsonl"

def merge_results():
    if not os.path.exists(OUTPUT_JSONL):
        print("❌ 결과 파일(batch_output.jsonl)이 없습니다. 2단계가 완료되었는지 확인하세요.")
        return

    print(">>> 데이터 병합을 시작합니다...")
    
    # 1. 원본 CSV 로드
    df = pd.read_csv(ORIGINAL_CSV)
    
    # 2. 결과 JSONL 로드 및 파싱
    results_map = {} # {인덱스: {label, score, reason}}
    
    print(">>> 결과 파일 파싱 중...")
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                res = json.loads(line)
                
                # custom_id에서 인덱스 추출 ("row_123" -> 123)
                custom_id = res['custom_id']
                idx = int(custom_id.split('_')[1])
                
                # 결과 내용 추출
                # Batch API 응답 구조는 조금 깊습니다.
                response_body = res['response']['body']
                content_str = response_body['choices'][0]['message']['content']
                content_json = json.loads(content_str)
                
                results_map[idx] = content_json
                
            except Exception as e:
                print(f"⚠️ 파싱 에러 (무시함): {e}")
                continue

    print(f">>> 파싱 완료. {len(results_map)}개의 결과를 CSV에 반영합니다.")

    # 3. DataFrame 업데이트
    success_count = 0
    for idx, data in results_map.items():
        if idx in df.index:
            df.at[idx, 'label'] = data.get('label')
            df.at[idx, 'score'] = data.get('score')
            df.at[idx, 'reason'] = data.get('reason')
            success_count += 1
            
    # 4. 저장
    df.to_csv(ORIGINAL_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ 병합 완료! 총 {success_count}개 행이 업데이트되었습니다.")
    print(f">>> 최종 파일: {ORIGINAL_CSV}")
    
    # 라벨 분포 확인
    print("\n[최종 라벨 분포]")
    print(df['label'].value_counts())

if __name__ == "__main__":
    merge_results()
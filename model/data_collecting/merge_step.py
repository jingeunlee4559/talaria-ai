import json
import pandas as pd
import os

# ==========================================
# 1. 파일 경로 설정 (사용자님 환경에 맞게 수정)
# ==========================================
# 기존에 만들어둔 353개 파일
EXISTING_FILE = "model/data_collecting/meme_dictionary_final.json" 

# 방금 GPT로 만든 172개 파일 (폴더 위치 확인 필수!)
# 사용자님 로그를 보니 'model/data_collecting/meme_dictionary_expanded.json'에 저장되었다고 뜸
NEW_FILE = "model/data_collecting/meme_dictionary_expanded.json"

# 최종 결과 파일
FINAL_FILE = "model/data_collecting/meme_dictionary_complete.json"

def merge_two_json_files():
    # 1. 파일 로드
    if not os.path.exists(EXISTING_FILE):
        print(f"❌ 오류: 기존 파일('{EXISTING_FILE}')을 찾을 수 없습니다.")
        return
    if not os.path.exists(NEW_FILE):
        print(f"❌ 오류: 새 파일('{NEW_FILE}')을 찾을 수 없습니다.")
        return

    with open(EXISTING_FILE, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    print(f">>> 기존 데이터 로드: {len(existing_data)}개")

    with open(NEW_FILE, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    print(f">>> 새 데이터 로드: {len(new_data)}개")

    # 2. 합치기 및 중복 제거
    combined = existing_data + new_data
    
    df = pd.DataFrame(combined)
    # 공백 제거 및 소문자화하여 키 정규화
    df['key_norm'] = df['primary_keyword'].astype(str).str.replace(" ", "").str.lower()
    
    # 중복 제거 (기존 데이터 우선 유지)
    df = df.drop_duplicates(subset=['key_norm'], keep='first')
    
    # 딕셔너리 변환
    final_list = df.drop(columns=['key_norm']).to_dict(orient='records')
    
    # 3. ID 재정렬 (깔끔하게 0000부터 다시 매김)
    for i, item in enumerate(final_list):
        item['id'] = f"meme_{i:04d}"
        
        # 필수 필드 빈값 채우기 (안전장치)
        if 'related_keywords' not in item: item['related_keywords'] = []
        if 'tags' not in item: item['tags'] = ["NEUTRAL"]
        if 'examples' not in item: item['examples'] = []
        if 'embedding_text' not in item:
            item['embedding_text'] = f"{item.get('primary_keyword','')}: {item.get('definition','')}"

    print(f"\n>>> 최종 병합 결과: 총 {len(final_list)}개 (중복 제거됨)")
    
    # 4. 저장
    with open(FINAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
        
    print(f"✅ 저장 완료: {FINAL_FILE}")

if __name__ == "__main__":
    merge_two_json_files()
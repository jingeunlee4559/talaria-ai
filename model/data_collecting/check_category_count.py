import json
import os
from collections import Counter

# 파일 경로 설정 (방금 생성한 파일 경로)
FILE_PATH = "model/data_collecting/meme_dictionary_labeled.json"

def check_category_distribution():
    # 1. 파일 존재 여부 확인
    if not os.path.exists(FILE_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {FILE_PATH}")
        return

    # 2. 데이터 로드
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return

    print(f"📊 전체 데이터 개수: {len(data)}개")
    print("-" * 40)

    # 3. 카테고리 개수 세기
    # 데이터에 'category' 키가 없는 경우 'Unknown'으로 처리
    categories = [item.get('category', 'Unknown') for item in data]
    category_counts = Counter(categories)

    # 4. 결과 출력 (많은 순서대로 정렬)
    # 총 합계 검증용 변수
    total_count = 0
    
    print(f"{'Category':<20} | {'Count':<5} | {'Percentage'}")
    print("-" * 40)
    
    for category, count in category_counts.most_common():
        percentage = (count / len(data)) * 100
        print(f"{category:<20} | {count:<5} | {percentage:.1f}%")
        total_count += count
        
    print("-" * 40)
    print(f"{'TOTAL':<20} | {total_count:<5}")

if __name__ == "__main__":
    check_category_distribution()
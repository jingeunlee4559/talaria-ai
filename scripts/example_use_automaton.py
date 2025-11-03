# 아호-코라식 오토마톤 활용 예시 코드

import ahocorasick
import pickle
from src.talaria.core.database import get_db_cursor

# 1. 저장된 파일을 로드
try:
    with open('./output/expert_dict.pkl', 'rb') as f:
        automaton = pickle.load(f)
    print("✅ 오토마톤 로드 완료!")
except FileNotFoundError:
    print("❌ 'expert_dict.pkl' 파일이 없습니다. build_automaton.py를 먼저 실행하세요.")

input_sentence = "요즘 IT 트렌드는 agile입니다. 한국어로는 애자일이죠."

# 2. automaton.iter()를 사용해 문장 속 키워드 찾기
found_keywords = {} 

for end_index, concept_id in automaton.iter(input_sentence):
    if concept_id not in found_keywords:
        # 3. A안에 따라, 찾은 concept_id로 DB에서 뜻을 조회 (DB 로직 필요)
        # description = get_description_from_db(concept_id) 
        found_keywords[concept_id] = "DB에서_가져온_뜻" # 예시
        
        # (원본 키워드가 필요하면 end_index를 역추적해야 하지만, 
        #  concept_id만 있어도 충분할 수 있습니다.)
        
print(f"찾은 개념 ID 목록: {list(found_keywords.keys())}")
# 출력 예: 찾은 개념 ID 목록: [101, 205] 
# (LiveKit/라이브킷이 101, WebRTC가 205라고 가정)
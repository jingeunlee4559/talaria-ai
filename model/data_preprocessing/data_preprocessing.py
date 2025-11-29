import json
import pandas as pd

# 1. 파일 경로 설정 (업로드한 파일명)
file_path = 'merged_final.json'

def load_and_flatten_data(path):
    # JSON 파일 로드
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, "파일을 찾을 수 없습니다."

    flattened_list = []

    # 2. 데이터 평탄화 (Flattening)
    # 중첩된 구조를 한 줄씩 풉니다.
    for entry in data:
        meme_word = entry.get('title', '')       # 밈 단어
        meaning = entry.get('description', '')   # 밈 뜻

        conversations = entry.get('conversations', [])

        for conv in conversations:
            # 3. 필요한 데이터만 추출
            # 모델 학습 시, Q(질문)가 Context가 되고 A(답변)가 Target 문장이 됩니다.
            row = {
                "meme_word": meme_word,      # 타겟 단어 (예: 어마무시하다)
                "meaning": meaning,          # 단어 뜻 (프롬프트 힌트용)
                "context_q": conv.get('q'),  # 대화 맥락 (질문)
                "target_a": conv.get('a'),   # 밈이 포함된 발화 (답변)
                "situation": conv.get('context', '') # 상황 설명 (있으면 도움됨)
            }
            flattened_list.append(row)

    # DataFrame으로 변환
    df = pd.DataFrame(flattened_list)
    return df, "변환 성공"

# 실행
df, message = load_and_flatten_data(file_path)

if df is not None:
    print(f">>> 총 데이터 개수: {len(df)}개")
    print("\n>>> 변환된 데이터 예시 (상위 3개):")
    # 보기 좋게 출력
    pd.set_option('display.max_colwidth', 50) # 컬럼 길이 조절
    print(df[['meme_word', 'meaning', 'target_a']].head(3))

    # 4. 나중에 LLM에 넣기 좋게 CSV로 임시 저장
    df.to_csv("ready_for_llm_labeling.csv", index=False, encoding='utf-8-sig')
    print("\n>>> 'ready_for_llm_labeling.csv' 파일로 저장되었습니다.")
else:
    print(message)
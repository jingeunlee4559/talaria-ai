import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 1. 설정
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

INPUT_CSV = "model/data_preprocessing/final_labeled_dataset.csv"
JSONL_FILE = "model/data_preprocessing/batch_requests.jsonl"
BATCH_ID_FILE = "model/data_preprocessing/batch_id.txt"

MODEL_NAME = "gpt-4o-mini"

# 사용자님이 만족하신 그 프롬프트 그대로 사용
SYSTEM_PROMPT = """
너는 한국어 '밈(Meme)'과 '신조어'의 미묘한 뉘앙스를 완벽하게 분석하는 언어학 전문가야.
주어지는 [상황, 질문, 답변, 밈 단어, 사전적 의미]를 종합적으로 고려하여,
해당 단어가 답변(Answer)에서 어떤 '감정'과 '의도'로 쓰였는지 분석해.

반드시 다음 4가지 라벨 중 하나로 분류하고 JSON으로 출력해.

1. SARCASM (비꼼/냉소): 겉으로는 칭찬 같지만 실제로는 비난하거나 조롱함, 혹은 자조적인 상황.
2. ADMIRATION (감탄/찬양): 긍정적인 의미로 매우 놀라거나, 칭찬하거나, 동의함.
3. SHOCK (충격/부정): 부정적인 상황에서 놀라거나, 공포, 혐오, 어이없음을 느낌.
4. NEUTRAL (중립/강조): 특별한 감정 없이 크기, 정도, 사실만을 강조하거나 단순한 추임새.

[출력 포맷 (JSON)]
{
  "label": "SARCASM",
  "score": 5,
  "reason": "상황이 망했는데 '대단하다'라고 했으므로 반어법임."
}
"""

def prepare_and_send_batch():
    # 1. CSV 로드
    if not os.path.exists(INPUT_CSV):
        print("❌ CSV 파일이 없습니다.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 2. 작업 대상 필터링 (이미 된 건 제외)
    # 컬럼이 없으면 생성
    for col in ['label', 'score', 'reason']:
        if col not in df.columns:
            df[col] = None
            
    # 라벨이 없거나 에러인 것들만 추출
    todo_df = df[
        df['label'].isnull() | 
        (df['label'] == 'ERROR') |
        (df['label'].notnull() & df['reason'].isnull())
    ].copy()
    
    print(f">>> 총 {len(df)}개 중 작업 대상: {len(todo_df)}개")
    
    if len(todo_df) == 0:
        print(">>> 모든 작업이 완료되어 보낼 게 없습니다!")
        return

    # 3. JSONL 파일 생성 (OpenAI Batch 포맷)
    print(">>> JSONL 파일 생성 중...")
    
    with open(JSONL_FILE, 'w', encoding='utf-8') as f:
        for idx, row in todo_df.iterrows():
            # 사용자가 만족한 프롬프트 내용 (다이어트 X)
            user_content = f"""
            [분석 데이터]
            1. Situation (상황): {row.get('situation', '정보 없음')}
            2. Question (맥락): {row.get('context_q', '')}
            3. Target Word (밈): {row.get('meme_word', '')}
            4. Dictionary Meaning (뜻): {row.get('meaning', '')}
            5. Answer (분석할 문장): "{row.get('target_a', '')}"

            위 Answer 문장에서 '{row.get('meme_word')}'가 사용된 구체적인 뉘앙스를 분석해줘.
            """
            
            # Request 객체 생성
            request_obj = {
                "custom_id": f"row_{idx}", # 나중에 합칠 때 인덱스로 씁니다
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0
                }
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    print(f">>> JSONL 파일 준비 완료: {JSONL_FILE}")

    # 4. 파일 업로드
    print(">>> OpenAI에 파일 업로드 중...")
    batch_input_file = client.files.create(
        file=open(JSONL_FILE, "rb"),
        purpose="batch"
    )
    file_id = batch_input_file.id
    print(f">>> 파일 업로드 완료 ID: {file_id}")

    # 5. 배치 생성 (실제 실행 요청)
    print(">>> 배치 작업 생성 중...")
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h", # 24시간 내 처리 (보통 1-2시간)
        metadata={"description": "meme_nuance_analysis"}
    )
    
    batch_id = batch_job.id
    print(f"\n✅ 배치 요청 성공! Batch ID: {batch_id}")
    
    # 6. ID 저장 (나중에 확인용)
    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch_id)
    print(f">>> Batch ID가 '{BATCH_ID_FILE}'에 저장되었습니다.")
    print(">>> 이제 컴퓨터를 끄고 쉬셔도 됩니다. 나중에 확인 스크립트를 실행하세요.")

if __name__ == "__main__":
    prepare_and_send_batch()
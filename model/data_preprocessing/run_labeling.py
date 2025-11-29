import os
import json
import pandas as pd
import asyncio
import random
import time
from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from dotenv import load_dotenv

# ==========================================
# 1. 설정 및 초기화
# ==========================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("⚠️ [경고] .env 파일에서 'OPENAI_API_KEY'를 찾을 수 없습니다.")

MODEL_NAME = "gpt-4o-mini"

# [속도 조절] 
# TPM(토큰 제한) 때문에 RPM을 더 낮춰야 안전합니다.
# 200 RPM (약 0.3초당 1개) 정도로 수정
TARGET_RPM = 350 
REQUEST_INTERVAL = 60.0 / TARGET_RPM 

# 파일 경로
OUTPUT_FILE = "model/data_preprocessing/final_labeled_dataset.csv"
INPUT_FILE = "model/data_preprocessing/ready_for_llm_labeling.csv"

# ==========================================
# 2. 시스템 프롬프트
# ==========================================
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

# ==========================================
# 3. 비동기 분석 함수
# ==========================================
# 기존 analyze_nuance 함수를 이걸로 교체하세요.

async def analyze_nuance(client, index, row, semaphore):
    async with semaphore:
        # [다이어트 1] 뜻(meaning)을 50글자로 제한
        meaning_short = str(row.get('meaning', ''))[:50]
        
        # [다이어트 2] 프롬프트를 아주 간결하게 변경 (토큰 절약)
        user_content = f"""
        상황:{row.get('situation','')}
        맥락:{row.get('context_q','')}
        단어:{row.get('meme_word','')}
        뜻:{meaning_short}
        문장:"{row.get('target_a','')}"
        
        위 문장에서 단어의 뉘앙스를 분석해.
        """

        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)
                
                print(".", end="", flush=True) 
                return index, result_json

            except RateLimitError:
                print("R", end="", flush=True)
                # 대기 시간을 조금 더 길게 잡아서 확실히 식힘
                wait_time = retry_delay * (2 ** attempt) + 2 
                await asyncio.sleep(wait_time)
            
            except APITimeoutError:
                print("T", end="", flush=True) 
                await asyncio.sleep(1)
                
            except Exception as e:
                print("E", end="", flush=True)
                return index, {"label": "ERROR", "score": 0, "reason": str(e)}
        
        print("X", end="", flush=True)
        return index, {"label": "ERROR", "score": 0, "reason": "Max retries exceeded"}

# ==========================================
# 4. 메인 프로세싱 함수
# ==========================================
async def run_labeling_process(df, is_test=False):
    client = AsyncOpenAI(api_key=API_KEY, timeout=30.0)
    # [수정] 세마포어(동시 접속)를 10 -> 5로 줄임 (안전 제일)
    semaphore = asyncio.Semaphore(5) 
    
    try:
        # 컬럼 초기화
        for col in ['label', 'score', 'reason']:
            if col not in df.columns:
                df[col] = None

        # 작업 대상 필터링
        todo_indices = df[
            df['label'].isnull() | 
            (df['label'] == 'ERROR') |
            (df['label'].notnull() & df['reason'].isnull()) 
        ].index.tolist()
        
        todo_count = len(todo_indices)
        if todo_count == 0:
            return df

        if not is_test:
            print(f">>> 작업을 시작합니다. 대상: {todo_count}개")
            print(">>> (범례: . = 성공 / R = 속도제한대기 / T = 타임아웃 / E = 에러)")

        tasks = []
        
        for i, idx in enumerate(todo_indices):
            row = df.loc[idx]
            task = asyncio.create_task(analyze_nuance(client, idx, row, semaphore))
            tasks.append(task)
            
            # 요청 간격 조절
            await asyncio.sleep(REQUEST_INTERVAL)

        if not is_test:
            print("\n>>> 모든 요청 큐 등록 완료. 응답 대기 중...")
        
        results = await asyncio.gather(*tasks)
        
        # 결과 반영
        for idx, res_json in results:
            df.at[idx, 'label'] = res_json.get('label')
            df.at[idx, 'score'] = res_json.get('score')
            df.at[idx, 'reason'] = res_json.get('reason')

        return df

    finally:
        await client.close()

# ==========================================
# 5. 실행 진입점
# ==========================================
if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f">>> 기존 파일 로드: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
    elif os.path.exists(INPUT_FILE):
        print(f">>> 원본 파일 로드: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
    else:
        print(f"❌ 오류: 입력 파일('{INPUT_FILE}')이 없습니다.")
        exit()

    for col in ['label', 'score', 'reason']:
        if col not in df.columns:
            df[col] = None

    print("\n======= [전체 데이터 실행 (안전 모드)] =======")
    
    # [수정] 배치 사이즈를 50개로 줄여서 자주 저장하게 함
    BATCH_SIZE = 50
    
    while True:
        todo_mask = (
            df['label'].isnull() | 
            (df['label'] == 'ERROR') | 
            (df['label'].notnull() & df['reason'].isnull())
        )
        todo_indices = df[todo_mask].index
        
        if len(todo_indices) == 0:
            print("\n>>> [완료] 모든 데이터 처리 끝!")
            break
            
        print(f"\n>>> 남은 데이터: {len(todo_indices)}개. 이번 배치: {BATCH_SIZE}개")
        
        current_batch_indices = todo_indices[:BATCH_SIZE]
        batch_df = df.loc[current_batch_indices].copy()
        
        # 실행
        batch_df = asyncio.run(run_labeling_process(batch_df, is_test=False))
        
        # 결과 반영
        cols_to_update = ['label', 'score', 'reason']
        df.loc[current_batch_indices, cols_to_update] = batch_df[cols_to_update]
        
        # 저장
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n>>> [저장 완료] {BATCH_SIZE}개 저장됨.")
        
        time.sleep(2)
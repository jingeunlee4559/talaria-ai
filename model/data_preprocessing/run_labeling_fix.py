import os
import json
import pandas as pd
import asyncio
import random
import time
from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

# ==========================================
# 1. 설정 및 초기화
# ==========================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("⚠️ [경고] .env 파일에서 'OPENAI_API_KEY'를 찾을 수 없습니다.")

MODEL_NAME = "gpt-4o-mini"
TARGET_RPM = 350
REQUEST_INTERVAL = 60.0 / TARGET_RPM 

OUTPUT_FILE = "final_labeled_dataset.csv"
INPUT_FILE = "ready_for_llm_labeling.csv"

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
async def analyze_nuance(client, index, row, semaphore):
    async with semaphore:
        user_content = f"""
        [분석 데이터]
        1. Situation (상황): {row.get('situation', '정보 없음')}
        2. Question (맥락): {row.get('context_q', '')}
        3. Target Word (밈): {row.get('meme_word', '')}
        4. Dictionary Meaning (뜻): {row.get('meaning', '')}
        5. Answer (분석할 문장): "{row.get('target_a', '')}"

        위 Answer 문장에서 '{row.get('meme_word')}'가 사용된 구체적인 뉘앙스를 분석해줘.
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
                return index, result_json

            except RateLimitError:
                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            except Exception as e:
                return index, {"label": "ERROR", "score": 0, "reason": str(e)}
        
        return index, {"label": "ERROR", "score": 0, "reason": "Max retries exceeded"}

# ==========================================
# 4. 메인 프로세싱
# ==========================================
async def run_labeling_process(df, is_test=False):
    client = AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(10) 
    
    try:
        # -------------------------------------------------------
        # [수정됨] 재작업 대상 선정 로직 강화
        # -------------------------------------------------------
        # 1. label이 없는 경우
        # 2. label은 있는데 reason이 없는 경우 (아까 잘못 저장된 것들 복구)
        # 3. ERROR 인 경우
        
        # 확실하게 컬럼이 있는지 확인
        for col in ['label', 'score', 'reason']:
            if col not in df.columns:
                df[col] = None

        # 작업 대상 인덱스 필터링
        todo_indices = df[
            df['label'].isnull() | 
            (df['label'] == 'ERROR') |
            (df['label'].notnull() & df['reason'].isnull()) # <--- [핵심] 라벨만 있고 이유가 없으면 다시 함
        ].index.tolist()
        
        total_count = len(df)
        todo_count = len(todo_indices)
        
        if todo_count == 0:
            return df

        if not is_test:
            print(f">>> 작업을 시작합니다. 남은 수량: {todo_count}개")

        tasks = []
        
        for i, idx in enumerate(todo_indices):
            row = df.loc[idx]
            task = asyncio.create_task(analyze_nuance(client, idx, row, semaphore))
            tasks.append(task)
            
            await asyncio.sleep(REQUEST_INTERVAL)
            
            if not is_test and (i + 1) % 50 == 0:
                print(f"   - [진행중] {i+1}/{todo_count} 요청 발송")

        if not is_test:
            print(">>> 응답 대기 중...")
        
        results = await asyncio.gather(*tasks)
        
        # 결과 DataFrame에 반영
        for idx, res_json in results:
            df.at[idx, 'label'] = res_json.get('label')
            df.at[idx, 'score'] = res_json.get('score')
            df.at[idx, 'reason'] = res_json.get('reason')

        return df

    finally:
        # [추가] 이벤트 루프 에러 방지용
        await client.close()

# ==========================================
# 5. 실행 진입점
# ==========================================
if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 1. 파일 로드
    if os.path.exists(OUTPUT_FILE):
        print(f">>> 기존 파일('{OUTPUT_FILE}') 로드")
        df = pd.read_csv(OUTPUT_FILE)
    elif os.path.exists(INPUT_FILE):
        print(f">>> 원본 파일('{INPUT_FILE}') 로드")
        df = pd.read_csv(INPUT_FILE)
    else:
        print("❌ 파일 없음")
        exit()

    # [핵심 수정] 메인 DataFrame에 필수 컬럼 강제 생성
    # 이 부분이 없어서 아까 저장이 안 된 것입니다.
    for col in ['label', 'score', 'reason']:
        if col not in df.columns:
            df[col] = None

    # 2. 전체 실행 (배치 처리)
    print("\n======= [전체 데이터 실행 (이어하기 및 복구)] =======")
    
    BATCH_SIZE = 300 # 배치 사이즈 조금 줄임 (안전하게)
    
    while True:
        # 작업 필요한 것들 다시 계산 (label 있어도 reason 없으면 포함됨)
        todo_mask = (
            df['label'].isnull() | 
            (df['label'] == 'ERROR') | 
            (df['label'].notnull() & df['reason'].isnull())
        )
        todo_indices = df[todo_mask].index
        
        if len(todo_indices) == 0:
            print(">>> [완료] 모든 데이터가 정상적으로 저장되었습니다!")
            break
            
        print(f"\n>>> 남은 데이터: {len(todo_indices)}개. 이번 배치: {BATCH_SIZE}개")
        
        current_batch_indices = todo_indices[:BATCH_SIZE]
        batch_df = df.loc[current_batch_indices].copy()
        
        # 실행
        batch_df = asyncio.run(run_labeling_process(batch_df, is_test=False))
        
        # [핵심 수정] update() 대신 직접 대입 사용 (이게 제일 확실함)
        # 인덱스를 기준으로 정확하게 꽂아 넣습니다.
        df.loc[current_batch_indices, ['label', 'score', 'reason']] = batch_df[['label', 'score', 'reason']]
        
        # 저장
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f">>> [저장 완료] '{OUTPUT_FILE}' 갱신됨. (reason 컬럼 확인하세요!)")
        
        time.sleep(2)
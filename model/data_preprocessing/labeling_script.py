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
# .env 파일 로드 (같은 폴더에 .env 파일이 있어야 함)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# API 키 확인
if not API_KEY:
    print("⚠️ [경고] .env 파일에서 'OPENAI_API_KEY'를 찾을 수 없습니다.")
    print("설정을 확인하거나, 코드에 직접 키를 입력해서 테스트하세요.")
    # API_KEY = "sk-..." # 필요 시 주석 해제 후 입력

MODEL_NAME = "gpt-4o-mini"

# [속도 조절 설정]
# Tier 1 한도(500 RPM)를 고려하여 분당 350회로 제한
TARGET_RPM = 350
REQUEST_INTERVAL = 60.0 / TARGET_RPM 

# ==========================================
# 2. 시스템 프롬프트 정의
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
# 3. 비동기 분석 함수 (개별 요청 처리)
# ==========================================
async def analyze_nuance(client, index, row, semaphore):
    # 네트워크 연결 폭주 방지용 세마포어
    async with semaphore:
        user_content = f"""
        [분석 데이터]
        1. Situation (상황): {row.get('situation', '정보 없음')}
        2. Question (맥락): {row.get('context_q', '')}
        3. Target Word (밈): {row.get('meme_word', '')}
        4. Dictionary Meaning (뜻): {row.get('meaning', '')}
        5. Answer (분석할 문장): "{row.get('target_a', '')}"

        위 Question, Answer 문장에서 '{row.get('meme_word')}'가 사용된 구체적인 뉘앙스를 분석해줘.
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
                # 429 에러 시 잠시 대기 (Jitter 포함)
                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                # 기타 에러 처리
                return index, {"label": "ERROR", "score": 0, "reason": str(e)}
        
        return index, {"label": "ERROR", "score": 0, "reason": "Max retries exceeded"}

# ==========================================
# 4. 메인 프로세싱 함수 (속도 제어 루프)
# ==========================================
async def run_labeling_process(target_df, desc_text="처리 중"):
    if not API_KEY:
        raise ValueError("API Key가 없습니다.")

    client = AsyncOpenAI(api_key=API_KEY)
    
    # 세마포어는 동시에 열리는 소켓 수 제한 (RPM 제한과는 별개)
    semaphore = asyncio.Semaphore(10) 
    
    tasks = []
    total_count = len(target_df)
    print(f"\n>>> [{desc_text}] 총 {total_count}개 데이터 분석 시작")
    
    if total_count > 100:
        print(f">>> (예상 소요 시간: 약 {total_count / TARGET_RPM:.1f}분 / 목표 속도: 분당 {TARGET_RPM}회)")

    start_time = time.time()
    
    # [핵심 로직] for문을 돌면서 요청을 생성하되, REQUEST_INTERVAL 만큼 쉼
    for index, row in target_df.iterrows():
        # 태스크 생성 (비동기 호출)
        task = asyncio.create_task(analyze_nuance(client, index, row, semaphore))
        tasks.append(task)
        
        # 속도 조절 (Throttling) - 여기서 멈춰서 RPM을 맞춤
        await asyncio.sleep(REQUEST_INTERVAL)
        
        # 진행 상황 로그 (데이터가 많을 때만)
        if total_count > 50 and (index + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   - 진행률: {index+1}/{total_count} 요청 발송 완료 (경과: {elapsed:.1f}초)")

    print(f">>> [{desc_text}] 모든 요청 발송 완료! 응답 수신 대기 중...")
    
    # 모든 결과가 돌아올 때까지 대기
    results = await asyncio.gather(*tasks)
    
    # 결과를 인덱스 순서대로 정렬 (비동기라 순서 섞임 방지)
    results.sort(key=lambda x: x[0])
    
    # 결과 리스트 추출
    labels = [res[1].get('label') for res in results]
    scores = [res[1].get('score') for res in results]
    reasons = [res[1].get('reason') for res in results]
    
    # 원본 데이터프레임에 컬럼 추가
    target_df['label'] = labels
    target_df['score'] = scores
    target_df['reason'] = reasons
    
    return target_df

# ==========================================
# 5. 실행 진입점 (Main Block)
# ==========================================
if __name__ == "__main__":
    # 1. 파일 로드
    file_name = 'model/data_preprocessing/ready_for_llm_labeling.csv'
    if not os.path.exists(file_name):
        print(f"❌ 오류: '{file_name}' 파일을 찾을 수 없습니다.")
        exit()
        
    print(f">>> '{file_name}' 파일을 로드합니다...")
    df = pd.read_csv(file_name)
    print(f">>> 데이터 로드 완료: 총 {len(df)}개 행")

    # 윈도우/리눅스 환경에 따른 이벤트 루프 정책 설정 (필요 시)
    # Windows Python 3.8+에서는 SelectorEventLoop가 기본이 아닐 수 있음
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # -----------------------------------------------------------
    # [Step 1] 테스트 실행 (상위 5개)
    # -----------------------------------------------------------
    print("\n" + "="*50)
    print("   [1단계] 상위 5개 데이터 테스트 실행")
    print("="*50)
    
    test_df = df.head(5).copy()
    
    # 실행
    test_df = asyncio.run(run_labeling_process(test_df, desc_text="테스트"))
    
    # 결과 출력
    print("\n[테스트 결과 확인]")
    # 보기 좋게 컬럼 선택해서 출력
    print(test_df[['meme_word', 'target_a', 'label', 'reason']])

    # -----------------------------------------------------------
    # [Step 2] 사용자 확인 및 전체 실행
    # -----------------------------------------------------------
    print("\n" + "="*50)
    user_input = input(">>> 위 테스트 결과를 확인하셨습니까?\n>>> 나머지 전체 데이터(11,000개)를 실행하시겠습니까? (y/n): ")
    
    if user_input.lower().strip() == 'y':
        print("\n" + "="*50)
        print("   [2단계] 전체 데이터 실행 (스마트 속도 제어 모드)")
        print("="*50)
        
        # 전체 데이터 실행
        final_df = asyncio.run(run_labeling_process(df, desc_text="전체 처리"))
        
        # 저장
        output_file = "final_labeled_dataset.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f">>> [완료] 모든 작업이 끝났습니다.")
        print(f">>> 결과 파일 저장됨: {output_file}")
        print(">>> 라벨 분포 요약:")
        print(final_df['label'].value_counts())
        print("="*50)
        
    else:
        print("\n>>> 작업을 중단합니다. (테스트만 수행됨)")
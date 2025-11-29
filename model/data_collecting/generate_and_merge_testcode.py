import os
import json
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ==========================================
# 1. 설정
# ==========================================
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 기존 데이터 파일 (없으면 빈 리스트로 처리됨)
EXISTING_FILE = "model/data_preprocessing/merged_final.json"
# 최종 저장될 파일
FINAL_OUTPUT_FILE = "model/data_preprocessing/final_meme_dictionary.json"

# [핵심] 주제 리스트 (최대한 구체적으로!)
TOPICS = [
    # 2024-2025 최신
    "2024년 유튜브 쇼츠 유행어 (럭키비키, 원영적 사고 등)",
    "2024년 틱톡 챌린지 밈",
    "2024년 MZ세대 카카오톡 말투",
    
    # 커뮤니티별
    "디시인사이드/에펨코리아 등 남초 커뮤니티 용어",
    "더쿠/인스티즈/트위터 등 여초 커뮤니티 용어 (덕질 용어)",
    "개발자 유머 및 슬랭 (디버깅, 배포 관련)",
    "주식/코인 투자자들의 은어 (한강, 떡상 등)",
    "리그오브레전드(LoL) 게임 용어 및 욕설 순화어",
    "오버워치/발로란트 게임 채팅 용어",
    
    # 상황별
    "직장인이 싫어하는 상황에서 쓰는 자조적 밈",
    "대학생 시험기간/수강신청 관련 밈",
    "K-POP 아이돌 팬들이 쓰는 칭찬/주접 멘트",
    "인터넷 방송(치지직, 아프리카) 채팅 밈"
]

# 시스템 프롬프트 (엄격한 스키마 요구)
SYSTEM_PROMPT = """
You are an expert AI creating a database for a 'Korean Meme Translation Service'.
Your goal is to generate a list of Korean memes/slang based on the given topic.

[Required Output Schema - JSON List]
[
  {
    "primary_keyword": "Standard form of the meme (e.g., 중꺾마)",
    "related_keywords": ["List", "of", "variations", "or", "typos"],
    "definition": "Detailed definition and origin.",
    "description_for_llm": "English explanation for the translator AI to understand the nuance.",
    "examples": ["Example sentence 1", "Example sentence 2"],
    "tags": ["Choose one or more: SARCASM, ADMIRATION, SHOCK, NEUTRAL"],
    "embedding_text": "Combine keyword and definition for vector search."
  }
]

[Rules]
1. Output MUST be a valid JSON list.
2. 'embedding_text' should be formatted like: "Keyword: Definition".
3. Provide at least 15 items per topic.
"""

# ==========================================
# 2. 생성 함수
# ==========================================
async def generate_memes_by_topic(topic):
    print(f"🚀 생성 시작: {topic}")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o", # 창의력 필요 -> 4o 권장
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate Korean memes about: '{topic}'"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # GPT가 {"memes": [...]} 또는 {"data": [...]} 형태로 줄 경우 처리
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, list):
                    return value
        
        return result if isinstance(result, list) else []
        
    except Exception as e:
        print(f"❌ 에러 발생 ({topic}): {e}")
        return []

# ==========================================
# 3. 통합 및 중복 제거 (Dedup)
# ==========================================
def merge_and_deduplicate(generated_data):
    print("\n>>> 데이터 통합 및 중복 제거 시작...")
    
    combined_data = []
    
    # 1. 기존 파일 로드 (merged_final.json)
    # 기존 파일은 스키마가 다를 수 있으므로 변환이 필요할 수 있습니다.
    # 여기서는 기존 파일도 우리가 원하는 스키마로 변환되었다고 가정하거나,
    # 현장에서 변환 로직을 추가해야 합니다. 
    # (일단은 생성된 데이터 위주로 처리하는 코드를 작성합니다)
    if os.path.exists(EXISTING_FILE):
        try:
            with open(EXISTING_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                # 기존 데이터 포맷이 다르다면 여기서 변환 로직 필요
                # 예시: 단순 {title, description} 형태라면 변환
                for item in existing:
                    if 'primary_keyword' not in item: # 구버전 포맷인 경우
                        combined_data.append({
                            "primary_keyword": item.get('title'),
                            "related_keywords": [],
                            "definition": item.get('description'),
                            "description_for_llm": item.get('description'), # 임시
                            "examples": [conv['q'] for conv in item.get('conversations', [])][:2],
                            "tags": ["NEUTRAL"], # 기본값
                            "embedding_text": f"{item.get('title')}: {item.get('description')}",
                            "source": "original_dataset"
                        })
                    else:
                        combined_data.append(item)
            print(f"   - 기존 데이터 {len(existing)}개 로드 완료")
        except Exception as e:
            print(f"⚠️ 기존 파일 로드 실패 (무시함): {e}")

    # 2. 생성된 데이터 추가
    for item in generated_data:
        item['source'] = 'generated_gpt4o'
        combined_data.append(item)
        
    # 3. 중복 제거 (primary_keyword 기준)
    # Pandas를 쓰면 편합니다.
    df = pd.DataFrame(combined_data)
    
    # 공백 제거 및 소문자화 등 정규화
    df['key_norm'] = df['primary_keyword'].astype(str).str.replace(" ", "").str.lower()
    
    before_count = len(df)
    # key_norm이 같은 것 중 첫 번째만 남김
    df = df.drop_duplicates(subset=['key_norm'], keep='first')
    
    # 정제된 리스트로 변환
    final_list = df.drop(columns=['key_norm']).to_dict(orient='records')
    
    print(f"   - 중복 제거 결과: {before_count}개 -> {len(final_list)}개 (삭제됨: {before_count - len(final_list)}개)")
    return final_list

# ==========================================
# 4. 메인 실행
# ==========================================
async def main():
    # 1. 생성 (병렬 처리)
    tasks = [generate_memes_by_topic(topic) for topic in TOPICS]
    results = await asyncio.gather(*tasks)
    
    all_generated = []
    for res in results:
        all_generated.extend(res)
        
    print(f"\n>>> 총 {len(all_generated)}개의 데이터가 생성되었습니다.")
    
    # 2. 병합 및 저장
    final_data = merge_and_deduplicate(all_generated)
    
    # 저장
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 최종 데이터셋 구축 완료! 저장 경로: {FINAL_OUTPUT_FILE}")
    print(f"   총 데이터 개수: {len(final_data)}개")

if __name__ == "__main__":
    asyncio.run(main())
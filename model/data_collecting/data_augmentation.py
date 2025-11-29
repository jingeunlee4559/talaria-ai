import os
import json
import asyncio
import pandas as pd
import random
import re
from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

# ==========================================
# 1. 설정
# ==========================================
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "model/data_collecting/meme_dictionary_final.json"
OUTPUT_FILE = "model/data_collecting/meme_dictionary_expanded.json"

# 토픽 리스트 (30개 그대로 유지)
TOPICS = [
    # (위의 30개 토픽 리스트 그대로 사용)
    "2025년 유행할 것으로 예상되는 Zalpha(Z+Alpha) 세대 신조어",
    "2024년 하반기 유튜브 쇼츠 및 릴스 챌린지 유행어 (럭키비키 등)",
    "2025년 트렌드 코리아 키워드 관련 파생 유행어",
    "2024-2025 아이돌 4세대/5세대 팬덤(라이즈, 투어스 등) 고유 용어",
    "2024년 틱톡에서 유행하는 10대들의 은어 및 줄임말",
    "최신 힙합/국힙 씬에서 래퍼들이 쓰는 추임새 및 유행어",
    "2024년 인스타그램 스토리 및 릴스용 감성/주접 멘트",
    "2025년 대비 갓생/자기계발 트렌드 관련 신조어 (미라클모닝 등)",
    "AI 및 챗GPT 발전으로 생긴 새로운 기술 유행어",
    "최근 유행하는 '밈 화법' (원영적 사고, 희진적 사고 등 사고방식 밈)",
    "디시인사이드 식물/반려동물/취미 갤러리 용어",
    "디시인사이드 국내야구/해외축구 갤러리 드립 및 유행어",
    "더쿠/인스티즈 등 여초 커뮤니티의 연애/고민 상담 용어",
    "에펨코리아/루리웹 등 남초 커뮤니티의 유머/게임 용어",
    "트위터(X) 실시간 트렌드 및 '트위터 예절' 관련 용어",
    "치지직/아프리카TV 인터넷 방송 시청자(트수)들의 채팅 용어",
    "유튜브 댓글창에서 자주 보이는 드립 및 고정 멘트",
    "당근마켓/중고나라 거래 시 자주 쓰는 매너/비매너 용어",
    "블라인드(Blind) 앱 직장인들이 쓰는 회사 생활 은어",
    "대학생 에브리타임(에타) 시험기간/수강신청/학식 관련 용어",
    "리그오브레전드(LoL) 티어/라인/운영 관련 은어 (숟가락, 백정 등)",
    "오버워치/발로란트 등 FPS 게임의 브리핑 및 정치질 용어",
    "메이플스토리/로스트아크 등 RPG 유저들의 강화/득템 관련 용어",
    "극도의 분노나 어이없음을 표현하는 창의적인 욕설 순화어",
    "상대방을 돌려까거나 비꼬는 고단수 풍자 밈",
    "무한 긍정/회복탄력성을 보여주는 '중꺾마' 류의 희망 밈",
    "월요병/야근/퇴사 등 K-직장인의 애환을 담은 밈",
    "연애/썸/이별 상황에서 쓰는 최신 플러팅 멘트 및 용어",
    "MBTI(T발 C야 등) 과몰입 관련 밈 및 성격 묘사 용어",
    "맛집/카페 탐방 시 쓰는 '맛도리', '웨이팅' 관련 유행어"
]

# [수정] 프롬프트 강화: 무조건 리스트만 뱉어라!
SYSTEM_PROMPT = """
You are a database generator.
Generate 15~20 Korean memes for the given topic.
Output MUST be a raw JSON list. Do not include any other text.

[Format]
[
  {
    "primary_keyword": "Word",
    "related_keywords": ["Var1", "Var2"],
    "definition": "Definition",
    "tags": ["NEUTRAL"],
    "examples": ["Ex1"],
    "embedding_text": "Word: Definition"
  }
]
"""

# ==========================================
# 2. 생성 함수 (파싱 로직 강화)
# ==========================================
async def generate_memes_by_topic(topic, semaphore):
    async with semaphore:
        print(f"🚀 생성 요청: {topic}")
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Topic: {topic}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.8
                )
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # [핵심 수정] 다양한 JSON 구조 대응
                items = []
                if isinstance(result, list):
                    items = result
                elif isinstance(result, dict):
                    # 딕셔너리라면 내부의 리스트를 찾아서 꺼냄
                    for val in result.values():
                        if isinstance(val, list):
                            items = val
                            break
                
                # 결과가 리스트가 아니거나 비었으면 재시도
                if not items or not isinstance(items, list):
                    raise ValueError("JSON output is not a valid list")

                print(f"✅ 완료 ({topic}): {len(items)}개 생성됨")
                await asyncio.sleep(random.uniform(1, 2))
                return items
                
            except Exception as e:
                print(f"⚠️ 재시도 ({topic}): {e}")
                await asyncio.sleep(2)
        
        print(f"❌ 실패 ({topic}): 재시도 초과")
        return []

# ==========================================
# 3. 통합 및 저장 (에러 방지 로직 추가)
# ==========================================
def merge_and_save(existing_data, generated_data):
    print("\n>>> 데이터 통합 및 중복 제거 시작...")
    
    # 1. 마지막 ID 찾기
    max_id_num = -1
    for item in existing_data:
        # item이 딕셔너리인지 확인
        if not isinstance(item, dict): continue
        
        if 'id' in item and str(item['id']).startswith("meme_"):
            try:
                num = int(item['id'].split('_')[1])
                if num > max_id_num: max_id_num = num
            except: pass
    
    print(f"   - 기존 마지막 ID: meme_{max_id_num:04d}")
    
    combined = list(existing_data)
    new_items_to_add = []
    
    # 2. 생성 데이터 처리
    for item in generated_data:
        # [핵심 수정] 문자열이나 이상한 데이터가 들어오면 스킵
        if not isinstance(item, dict):
            continue
            
        item['source'] = 'gpt4o_generated'
        # 필수 필드 채우기
        item.setdefault('related_keywords', [])
        item.setdefault('tags', ["NEUTRAL"])
        item.setdefault('examples', [])
        if 'embedding_text' not in item:
            item['embedding_text'] = f"{item.get('primary_keyword','')}: {item.get('definition','')}"
            
        new_items_to_add.append(item)

    # 3. 중복 제거
    existing_keys = set()
    for item in existing_data:
        if isinstance(item, dict) and 'primary_keyword' in item:
            key = str(item['primary_keyword']).replace(" ", "").lower()
            existing_keys.add(key)
            
    final_new_items = []
    skipped_count = 0
    
    for item in new_items_to_add:
        if 'primary_keyword' not in item: continue
        
        key = str(item['primary_keyword']).replace(" ", "").lower()
        if key in existing_keys:
            skipped_count += 1
            continue
            
        existing_keys.add(key)
        max_id_num += 1
        item['id'] = f"meme_{max_id_num:04d}"
        final_new_items.append(item)
        
    final_list = combined + final_new_items
    
    print(f"   - 생성된 데이터(유효): {len(new_items_to_add)}개")
    print(f"   - 중복 제외 후 추가됨: {len(final_new_items)}개")
    print(f"   - 최종 데이터 총 개수: {len(final_list)}개")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 저장 완료: {OUTPUT_FILE}")

# ==========================================
# 4. 메인 실행
# ==========================================
async def main():
    existing_data = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    
    # 30개 토픽 병렬 실행
    semaphore = asyncio.Semaphore(5) # 동시 5개로 늘림
    tasks = [generate_memes_by_topic(topic, semaphore) for topic in TOPICS]
    results = await asyncio.gather(*tasks)
    
    all_generated = []
    for res in results:
        all_generated.extend(res)
        
    print(f"\n>>> 총 생성 시도 결과: {len(all_generated)}개 데이터 수집됨")
    
    merge_and_save(existing_data, all_generated)

if __name__ == "__main__":
    asyncio.run(main())
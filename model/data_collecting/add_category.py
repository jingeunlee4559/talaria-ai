import json
import os
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# --- 1. 설정 ---
INPUT_FILE = "model/data_collecting/meme_dictionary_complete.json"
OUTPUT_FILE = "model/data_collecting/meme_dictionary_labeled.json"

CATEGORIES = [
    'k-meme', 'global-meme', 'overseas-meme', 'game', 'music', 
    'mukbang', 'travel', 'vlog', 'talk', 'study'
]

BATCH_SIZE = 10 
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- 2. 분류 함수 (Retry 로직 강화) ---
async def classify_batch(batch_memes: List[Dict]) -> List[str]:
    """
    밈 리스트(Batch)를 받아서 카테고리 리스트를 반환합니다.
    실패 시(Rate Limit 등) 성공할 때까지 재시도합니다.
    """
    items_text = ""
    for idx, item in enumerate(batch_memes):
        keyword = item.get('primary_keyword', '')
        definition = item.get('definition', '')
        items_text += f"{idx+1}. {keyword}: {definition}\n"

    prompt = f"""
    You are a Meme Classifier. Classify each of the following memes into exactly ONE category from the list below.
    
    [Categories]
    {", ".join(CATEGORIES)}
    
    [Rules]
    1. Reply with a JSON list of strings.
    2. The list must contain exactly {len(batch_memes)} categories.
    3. Use 'k-meme' or 'talk' for general slang if unsure.
    
    [Input Memes]
    {items_text}
    
    [Output Format]
    ["category1", "category2", ...]
    """

    retry_count = 0
    while True:  # ⭐️ 성공할 때까지 무한 반복 (안전장치)
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            categories = json.loads(content)
            
            if len(categories) != len(batch_memes):
                print(f"⚠️ 개수 불일치 재시도 ({len(categories)} vs {len(batch_memes)})...")
                retry_count += 1
                await asyncio.sleep(1)
                continue # 개수가 다르면 재시도
                
            return categories # 성공 시 반환

        except Exception as e:
            error_msg = str(e)
            retry_count += 1
            wait_time = min(retry_count * 2, 30) # 점진적 대기 (최대 30초)
            
            if "429" in error_msg:
                print(f"⏳ Rate Limit 감지! {wait_time}초 대기 후 재시도... (Error: {e})")
            else:
                print(f"❌ API 오류 발생. {wait_time}초 대기 후 재시도... (Error: {e})")
            
            await asyncio.sleep(wait_time)

# --- 3. 메인 실행 로직 ---
async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일이 없습니다: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📂 데이터 로드 완료: {len(data)}개")
    
    labeled_data = []
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"🚀 분류 시작 (총 {total_batches} 배치)...")
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        batch_idx = (i // BATCH_SIZE) + 1
        
        print(f"   - Processing Batch {batch_idx}/{total_batches} ({len(batch)} items)...", end="\r")
        
        # 여기서 실패하면 성공할 때까지 내부에서 재시도함
        categories = await classify_batch(batch)
        
        for item, category in zip(batch, categories):
            if category not in CATEGORIES:
                category = 'k-meme'
            item['category'] = category
            labeled_data.append(item)
            
        # 기본 대기 (Rate Limit 예방용)
        await asyncio.sleep(0.5)

    print("\n✅ 모든 분류 완료!")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=4)
        
    print(f"💾 저장 완료: {OUTPUT_FILE}")
    
    print("\n[샘플 데이터 확인]")
    for item in labeled_data[:5]:
        print(f"- {item['primary_keyword']} -> {item['category']}")

if __name__ == "__main__":
    asyncio.run(main())
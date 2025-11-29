import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

BATCH_ID_FILE = "model/data_preprocessing/batch_id.txt"
OUTPUT_JSONL = "model/data_preprocessing/batch_output.jsonl"

def check_and_download():
    if not os.path.exists(BATCH_ID_FILE):
        print("❌ Batch ID 파일이 없습니다. 1단계부터 실행하세요.")
        return

    with open(BATCH_ID_FILE, "r") as f:
        batch_id = f.read().strip()

    print(f">>> Batch ID 조회 중: {batch_id}")
    batch_job = client.batches.retrieve(batch_id)
    
    status = batch_job.status
    print(f"\n📊 현재 상태: [{status.upper()}]")
    print(f"   - 처리된 요청: {batch_job.request_counts.completed} / {batch_job.request_counts.total}")
    
    if status == "completed":
        print("\n✅ 작업이 완료되었습니다! 결과 다운로드 중...")
        output_file_id = batch_job.output_file_id
        
        if output_file_id:
            file_response = client.files.content(output_file_id)
            with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
                f.write(file_response.text)
            print(f">>> 결과 저장 완료: {OUTPUT_JSONL}")
            print(">>> 이제 3단계(병합) 스크립트를 실행하세요.")
        else:
            print("⚠️ 완료되었으나 결과 파일 ID가 없습니다. (오류 가능성 있음)")
            
    elif status == "failed":
        print("❌ 작업 실패. OpenAI 대시보드를 확인하세요.")
        print(batch_job.errors)
    
    else:
        print("\n⏳ 아직 작업 중입니다. 나중에 다시 실행해 보세요.")
        print("   (validating -> in_progress -> finalizing -> completed 순서로 진행됩니다)")

if __name__ == "__main__":
    check_and_download()
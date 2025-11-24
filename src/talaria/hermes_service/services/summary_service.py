import logging
from ..schemas.summary import MeetingSummaryIn, MeetingSummaryOut
from ..summary_graph import app as summary_graph

log = logging.getLogger(__name__)

async def process_meeting_summary(payload: MeetingSummaryIn) -> MeetingSummaryOut:
    # 1. 프롬프트 엔지니어링 (텍스트 조립)
    formatted_text = f"""[회의 정보]
기준 날짜: {payload.current_date}
회의 제목: {payload.meeting_title}
참석자: {payload.participants}
[스크립트]
{payload.transcript}"""

    # 2. 초기 상태 설정
    initial_state = {
        "input_transcript": formatted_text,
        "metadata": {
            "current_date": payload.current_date,
            "meeting_id": payload.meeting_id
        },
        "summary_retry_count": 0,
        "extraction_retry_count": 0,
        "summary_feedback": "",
        "extraction_feedback": ""
    }

    # 3. 그래프 실행 (비동기)
    log.info(f"[{payload.meeting_id}] 요약 그래프 실행 시작...")
    result = await summary_graph.ainvoke(initial_state)
    
    final_response = result.get("final_response", {})
    
    # 메타데이터 보강
    if "meta" in final_response:
        final_response["meta"]["meeting_id"] = payload.meeting_id

    return final_response
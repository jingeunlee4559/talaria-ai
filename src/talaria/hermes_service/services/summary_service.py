import logging
import time
from ..schemas.summary import MeetingSummaryIn, MeetingSummaryOut
from ..summary_graph import app as summary_graph

log = logging.getLogger(__name__)

async def process_meeting_summary(payload: MeetingSummaryIn) -> MeetingSummaryOut:
    # 1. 프롬프트 엔지니어링 (텍스트 조립)
    # (참고: 단순 텍스트보다 구조화된 텍스트가 LLM 인식에 더 좋습니다)
    formatted_text = f"""[회의 정보]
기준 날짜: {payload.current_date}
회의 제목: {payload.meeting_title}
참석자: {payload.participants}

[스크립트 본문]
{payload.transcript}"""

    # 2. 참가자 목록 리스트로 변환 (쉼표 구분 가정)
    participant_list = [p.strip() for p in payload.participants.split(",") if p.strip()]

    # 3. [핵심 수정] 초기 상태(State) 설정
    # state.py의 SummaryState 정의와 키(Key) 이름을 정확히 일치시켜야 합니다.
    initial_state = {
        # (수정 전: "input_transcript") -> (수정 후: "transcript")
        "transcript": formatted_text, 
        
        # (수정 전: "metadata" 내부에 존재) -> (수정 후: 최상위 레벨로 이동)
        "current_date": payload.current_date,
        "participants": participant_list,

        # 나머지 필드 초기화
        "summary_intro": "",
        "keywords": [],
        "decisions": [],
        "track_a_feedback": "",
        "track_a_retry": 0,

        "action_items": [],
        "track_b_feedback": "",
        "track_b_retry": 0,

        "content_score": 0.0,
        "content_risks": [],
        "content_improvements": [],
        "process_score": 0.0,
        "process_risks": [],
        "coach_next_steps": [],
        
        "final_output": {}
    }

    # 4. 그래프 실행 (비동기)
    log.info(f"[{payload.meeting_id}] 요약 그래프 실행 시작...")
    
    # invoke 결과는 State의 최종 상태입니다.
    final_state = await summary_graph.ainvoke(initial_state)
    
    # 5. 최종 결과 반환
    # node_final_merger에서 만든 'final_output'을 꺼냅니다.
    final_response_data = final_state.get("final_output", {})
    
    # 메타데이터 보강 (필요 시)
    meta_info = {
        "meeting_id": payload.meeting_id,
        "status": "success",
        "generated_at": str(int(time.time()))
    }

    return MeetingSummaryOut(
        meta=meta_info,
        result=final_response_data
    )
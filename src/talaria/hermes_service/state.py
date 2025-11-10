#LangGraph의 AgentState (세션 상태) 정의

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from typing import Optional, List
from typing import Literal

class FoundTerm(TypedDict):
    term: str       # 찾은 단어 (예: "CR")
    meaning: str    # DB에서 조회한 의미 (예: "변경 요청서")

class TranslateState(TypedDict):
    stt_text : str # 사용자 발화 STT 내용
    translated_text : str # 사용자 발화 파파고 단순 번역 내용

    context : List[Document] # RAG 검색 결과 청크
    
    meeting_id : int # 현재 진행중인 회의 ID
    docs_id : Optional[List[int]] # 회의에 매핑된 문서들의 ID
    chat_history : Optional[List[str]] # 이전 회의 대화 10개
    found_terms: List[FoundTerm] # 찾은 용어 리스트

    should_context_translate: bool # 문맥 번역 조건 만족 여부

    answer : str # llm에 의해 보정된 번역

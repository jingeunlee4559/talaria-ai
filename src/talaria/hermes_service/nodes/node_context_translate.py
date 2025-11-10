#문맥 기반 번역

import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# state.py의 위치에 따라 상대 경로 조정이 필요할 수 있습니다.
from ..state import TranslateState, FoundTerm

log = logging.getLogger(__name__)

# --- (싱글톤) LLM 모델 초기화 ---
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=2)
    log.info("✅ OpenAI LLM(gpt-4o-mini) 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    log.critical(f"❌ OpenAI LLM 로드 실패. API 키 또는 환경 변수를 확인하세요: {e}", exc_info=True)
    llm = None


def _format_context_for_prompt(
    history: Optional[List[str]], 
    terms: List[FoundTerm], 
    docs: List[Document]
) -> str:
    """State의 컨텍스트 리스트들을 LLM 프롬프트에 넣기 좋은 단일 문자열로 포맷팅합니다."""
    
    # 1. 대화 기록
    if history:
        history_str = "\n".join(f"- {h}" for h in history)
    else:
        history_str = "No recent chat history available."
    
    # 2. 전문 용어 (Aho-Corasick)
    if terms:
        terms_str = "\n".join(f"- {t['term']} (Meaning: {t['meaning']})" for t in terms)
    else:
        terms_str = "No specific project terms found in this utterance."
    
    # 3. RAG 문서 (우리의 "약한 신호" 전략 반영)
    if docs:
        doc_str = "\n".join(
            # "약한 신호"일 수 있으므로 LLM이 참고하도록 distance 값을 명시
            f"- (Distance: {doc.metadata.get('distance', 999):.4f}) \"{doc.page_content}\"" 
            for doc in docs
        )
    else:
        doc_str = "No relevant document excerpts found."
    
    # 3가지 컨텍스트를 하나의 문자열로 결합
    return f"""
[Context 1: Recent Chat History]
{history_str}

[Context 2: Project Terminology Found in Utterance]
{terms_str}

[Context 3: Relevant Document Excerpts (RAG)]
{doc_str}
    """

# --- 프롬프트 템플릿 정의 ---
SYSTEM_PROMPT = """You are an expert translator and proofreader for a real-time multilingual meeting.
Your task is to correct potential STT (Speech-to-Text) errors and improve a simple, machine-translated text based on all available context.

Here is all the context you MUST use to make your decision:
{formatted_context}

Here is the user's utterance you must process:
[Original STT (May contain errors)]: {stt_text}
[Simple Translation (From Papago)]: {translated_text}

Your instructions are:
1.  Analyze the [Original STT]. It may contain errors (e.g., "헤매다" instead of "Hermes").
2.  Use all three contexts ([Chat History], [Project Terminology], [Relevant Documents]) to determine the *true intended meaning* of the [Original STT].
3.  Pay close attention to [Relevant Documents]. Even excerpts with high distance (e.g., > 0.7) are included as "weak signals" because they might contain the correct spelling of a misheard word (e.g., "Hermes").
4.  Based on this corrected understanding, refine and improve the [Simple Translation] to be as accurate as possible.
5.  If the [Simple Translation] is already 100% accurate and contextually correct, return it exactly as is, without any changes.
6.  **You MUST respond ONLY with the final, corrected translation text.** Do not add any conversational phrases, explanations, or labels like "Corrected Translation:".
"""

# ChatPromptTemplate 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT)
])

# LangChain Expression Language (LCEL) 체인 정의
# 이 체인은 'invoke'될 때 TranslateState 딕셔너리를 입력으로 받습니다.
chain = (
    {
        # 1. 헬퍼 함수를 호출하여 3가지 컨텍스트를 1개의 문자열로 포맷팅
        "formatted_context": lambda x: _format_context_for_prompt(
            x.get("chat_history"), 
            x.get("found_terms", []), 
            x.get("context", [])
        ),
        # 2. 나머지 프롬프트 변수들을 state에서 그대로 가져옴
        "stt_text": lambda x: x["stt_text"],
        "translated_text": lambda x: x["translated_text"]
    }
    | prompt  # 포맷팅된 딕셔너리를 프롬프트에 삽입
    | llm     # LLM 호출
    | StrOutputParser() # LLM의 응답(AIMessage)을 문자열(str)로 파싱
)

def context_translate_node(state: TranslateState) -> TranslateState:
    """
    (IF) 'should_context_translate'가 True일 때 실행되는 노드.
    
    LLM을 사용하여 STT 오류를 보정하고 번역 품질을 향상시켜
    그 결과를 'answer' 필드에 저장합니다.
    """
    if not llm:
        log.error("LLM이 로드되지 않았습니다. 문맥 기반 번역을 건너뛰고 기본 번역으로 폴백합니다.")
        # LLM 실패 시, 안전하게 파파고 원본으로 폴백
        state['answer'] = state['translated_text']
        return state

    log.info(f"문맥 기반 번역 노드 실행: (STT: '{state['stt_text']}')")

    try:
        # LCEL 체인 실행. 'state' 객체는 딕셔너리처럼 동작하므로
        # 체인에 필요한 키(stt_text, chat_history 등)를 자동으로 추출하여 사용합니다.
        corrected_translation = chain.invoke(state)
        
        # LLM 응답 앞뒤의 불필요한 공백이나 따옴표 제거
        state['answer'] = corrected_translation.strip().strip('"') 
        log.info(f"LLM 보정 완료: '{state['answer']}'")

    except Exception as e:
        log.error(f"LLM 문맥 번역 중 오류 발생: {e}", exc_info=True)
        # LLM 호출 실패 시, 안전하게 파파고 원본 번역으로 폴백
        state['answer'] = state['translated_text']

    return state
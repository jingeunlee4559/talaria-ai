# src/talaria/hermes_service/nodes/node_context_translate.py
# 문맥 기반 번역 (수정: JSON 출력으로 원문 보정본과 번역본 동시 생성)

import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# 🔴 [삭제] from langchain_core.output_parsers import StrOutputParser
# 🟢 [추가] JSON 파서 사용
from langchain_core.output_parsers.json import JsonOutputParser 
from dotenv import load_dotenv

load_dotenv()

from ..state import TranslateState, FoundTerm

log = logging.getLogger(__name__)

try:
    llm = ChatOpenAI(model="gpt-5.1", temperature=0.1, max_retries=2)
    log.info("✅ OpenAI LLM(gpt-4o-mini) 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    log.critical(f"❌ OpenAI LLM 로드 실패: {e}", exc_info=True)
    llm = None


def _format_context_for_prompt(
    history: Optional[List[str]], 
    terms: List[FoundTerm], 
    docs: List[Document]
) -> str:
    # (기존 로직과 동일하므로 생략 가능, 혹은 그대로 유지)
    if history:
        history_str = "\n".join(f"- {h}" for h in history)
    else:
        history_str = "No recent chat history available."
    
    if terms:
        terms_str = "\n".join(f"- {t['term']} (Meaning: {t['meaning']})" for t in terms)
    else:
        terms_str = "No specific project terms found in this utterance."
    
    if docs:
        doc_str = "\n".join(
            f"- (Distance: {doc.metadata.get('distance', 999):.4f}) \"{doc.page_content}\"" 
            for doc in docs
        )
    else:
        doc_str = "No relevant document excerpts found."
    
    return f"""
[Context 1: Recent Chat History]
{history_str}

[Context 2: Project Terminology Found in Utterance]
{terms_str}

[Context 3: Relevant Document Excerpts (RAG)]
{doc_str}
    """

# 🟢 [수정] 프롬프트: JSON 출력을 지시하고, STT 보정(correction) 작업을 명시
SYSTEM_PROMPT = """You are an expert translator and proofreader for a real-time multilingual meeting.
Your task is to correct potential STT (Speech-to-Text) errors based on context and provide a high-quality translation.

Here is the context:
{formatted_context}

Here is the user's utterance:
[Original STT]: {stt_text}
[Simple Translation]: {translated_text}
[Target Language]: {target_lang}

You MUST perform two tasks:
1. **Correct STT**: Analyze the [Original STT]. Fix any typos, wrong technical terms (refer to Context 2), or context errors (refer to Context 1 & 3).
   - Example: "A 테스트" -> "A/B 테스트" (if "A/B Testing" is in context)
   - Example: "API가 뻗어서" -> "API가 뻗어서" (Keep slang if it conveys meaning, but fix obvious typos)
2. **Refine Translation**: Based on the *Corrected STT*, provide the best possible translation in [Target Language].

**Output Format**:
Respond ONLY with a valid JSON object containing these two keys:
{{
  "corrected_stt": "The corrected version of Original STT in its original language",
  "final_translation": "The final refined translation in Target Language"
}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT)
])

# 🟢 [수정] 파서를 JsonOutputParser로 변경
parser = JsonOutputParser()

chain = (
    {
        "formatted_context": lambda x: _format_context_for_prompt(
            x.get("chat_history"), 
            x.get("found_terms", []), 
            x.get("context", [])
        ),
        "stt_text": lambda x: x["stt_text"],
        "translated_text": lambda x: x["translated_text"],
        "target_lang": lambda x: x["target_lang"]
    }
    | prompt
    | llm
    | parser # 🟢 StrOutputParser() 대신 사용
)

def context_translate_node(state: TranslateState) -> TranslateState:
    """
    [Business Mode - A-3a] (LLM 보정 노드)
    STT 오류를 보정하여 'corrected_stt'에 저장하고,
    그 결과를 바탕으로 번역하여 'final_answer'에 저장합니다.
    """
    if not llm:
        log.error("LLM 로드 실패. 기본 번역으로 폴백합니다.")
        state['final_answer'] = state['translated_text']
        state['corrected_stt'] = state['stt_text'] # LLM 없으면 원본 사용
        return state

    log.info(f"Biz 모드 (Slow Path): LLM 보정 실행 (STT: '{state['stt_text']}')")

    try:
        # LLM 호출 (JSON 반환)
        response = chain.invoke(state)
        
        # 🟢 [수정] 응답 파싱 및 State 업데이트
        corrected_stt = response.get("corrected_stt")
        final_translation = response.get("final_translation")

        # 결과가 비어있을 경우 대비 (Optional)
        if not corrected_stt: corrected_stt = state['stt_text']
        if not final_translation: final_translation = state['translated_text']

        state['corrected_stt'] = corrected_stt
        state['final_answer'] = final_translation
        
        log.info(f"✅ Biz 보정 완료:\n - STT: {state['stt_text']} -> {state['corrected_stt']}\n - Trans: {state['final_answer']}")

    except Exception as e:
        log.error(f"LLM 문맥 번역 중 오류 발생: {e}", exc_info=True)
        # 오류 발생 시 폴백
        state['final_answer'] = state['translated_text']
        state['corrected_stt'] = state['stt_text']

    return state
# src/talaria/hermes_service/nodes/node_manner_guard.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from ..state import TranslateState

log = logging.getLogger(__name__)

# 1. LLM 설정 (빠른 속도를 위해 gpt-4o-mini 사용)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0) # 온도를 0으로 하여 일관성 유지
    log.info("✅ Manner Guard LLM 로드 완료.")
except Exception as e:
    llm = None
    log.critical(f"❌ Manner Guard LLM 로드 실패: {e}")

# 2. 출력 스키마 정의 (JSON으로 확실하게 받기 위함)
class SafetyCheckResult(BaseModel):
    is_safe: bool = Field(description="True if the text is safe, False otherwise.")
    filtered_text: str = Field(description="The sanitized text. Use '***' for hate speech, and rewording for mild slang.")
    reason: str = Field(description="Reason for filtering (e.g., PROFANITY, HATE_SPEECH, NONE).")

parser = JsonOutputParser(pydantic_object=SafetyCheckResult)

# 3. 프롬프트 정의 (하이브리드 전략 적용)
GUARD_PROMPT = """You are the 'Manner Guard' AI of a real-time translation service.
Your job is to screen the translated text for **unethical content**.

### Rules for Filtering:
1. **SAFE Content:** If the text is clean or just has harmless jokes, keep `is_safe` = true and `filtered_text` as is.
2. **MILD Slang (Vibe):** If it contains mild slang (e.g., 'damn', 'shit', '존나', '개웃겨'), **REWRITE** it wititly to maintain the fun vibe but make it less offensive. (e.g., 'Freaking funny', '겁나 웃겨').
3. **EXTREME Hate/Profanity:** If it contains hate speech, sexual harassment, or severe curses (e.g., 'Fuck you', 'Cunt', '개새끼', 패드립), **MASK** the specific word with `***`. (e.g., 'You ***').

### Input Text:
"{input_text}"

### Output Format (JSON):
{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(GUARD_PROMPT, partial_variables={"format_instructions": parser.get_format_instructions()})
chain = prompt | llm | parser

def manner_guard_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - Post-Processing]
    최종 번역문에 대해 윤리적 가드레일을 적용합니다.
    """
    if not llm:
        return state

    current_text = state.get("final_answer", "")
    
    # 입력이 비어있으면 패스
    if not current_text.strip():
        state['is_safe'] = True
        return state

    log.info(f"🛡️ Manner Guard 검사 시작: '{current_text}'")

    try:
        # LLM 호출
        result = chain.invoke({"input_text": current_text})
        
        # 결과 파싱
        is_safe = result.get("is_safe", True)
        filtered_text = result.get("filtered_text", current_text)
        reason = result.get("reason", "NONE")

        # State 업데이트
        state['is_safe'] = is_safe
        state['safety_reason'] = reason
        state['original_text_before_guard'] = current_text # 원본 백업
        
        if not is_safe:
            state['final_answer'] = filtered_text # 필터링된 텍스트로 교체
            log.warning(f"🚨 유해 콘텐츠 감지 ({reason})! 필터링 적용: '{current_text}' -> '{filtered_text}'")
        else:
            log.info("✅ 안전한 텍스트입니다.")

    except Exception as e:
        log.error(f"❌ Manner Guard 실행 중 오류: {e}")
        # 오류 시 원본 그대로 내보냄 (Fail-Open 정책)
        state['is_safe'] = True 
        
    return state
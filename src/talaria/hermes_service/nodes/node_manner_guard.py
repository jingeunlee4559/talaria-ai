# src/talaria/hermes_service/nodes/node_manner_guard.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from ..state import TranslateState

log = logging.getLogger(__name__)

try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
except:
    llm = None

# 언어별 표준 차단 메시지 정의
BLOCK_MESSAGES = {
    "ko": "🚫 부적절한 표현이 포함된 메시지입니다.",
    "en": "🚫 Message hidden due to inappropriate content.",
    "ja": "🚫 不適切な表現が含まれているため表示されません。",
    "zh": "🚫 该消息因包含不当内容而被隐藏。",
    "default": "🚫 Message hidden."
}

def get_block_msg(lang_code: str) -> str:
    """언어 코드를 기반으로 적절한 차단 메시지를 반환"""
    code = lang_code.lower()
    if "ko" in code or "korean" in code:
        return BLOCK_MESSAGES["ko"]
    elif "ja" in code or "japanese" in code:
        return BLOCK_MESSAGES["ja"]
    elif "zh" in code or "chinese" in code:
        return BLOCK_MESSAGES["zh"]
    return BLOCK_MESSAGES["en"]

class SafetyCheckResult(BaseModel):
    severity: str = Field(description="One of: SAFE, WORD_MASK, BLOCK")
    filtered_source_text: str = Field(description="Filtered source text (only for WORD_MASK).")
    filtered_target_text: str = Field(description="Filtered target text (only for WORD_MASK).")
    reason: str = Field(description="Reason for filtering.")

parser = JsonOutputParser(pydantic_object=SafetyCheckResult)

GUARD_PROMPT = """You are a Content Moderator.
Your goal is to block Hate Speech while protecting Free Speech.

### 🛡️ WHITELIST (Authorized Memes):
Words in [{whitelist}] are ALWAYS SAFE.

### 🚨 DECISION GUIDELINES:

**1. WHAT IS "SAFE"? (Do NOT Block)**
- **Criticism/Boycott:** "I will boycott...", "This sucks". -> **SAFE**
- **Personal Anger:** "I'm pissed off". -> **SAFE**
- **Metaphors:** "You are a potato". -> **SAFE**
- **Simple Insults:** "Idiot", "Stupid". -> **SAFE**

**2. WHAT IS "WORD_MASK"? (Partial Censorship)**
- Only specific vulgar slurs (e.g., F*ck, Sh*t, 씨발, 개새끼).
- Action: Set severity='WORD_MASK', replace bad words with `***`.

**3. WHAT IS "BLOCK"? (Total Ban)**
- **Hate Speech:** Attacks on Race, Gender, Religion, Disability.
- **Sexual Violence/Harassment.**
- **Realistic Threats.**
- Action: Set severity='BLOCK'. (No need to generate text, just flag it).

---
**Input Source:** "{source_text}" (Lang: {source_lang})
**Input Target:** "{target_text}" (Lang: {target_lang})

Output JSON: {format_instructions}"""

prompt = ChatPromptTemplate.from_template(GUARD_PROMPT, partial_variables={"format_instructions": parser.get_format_instructions()})
chain = prompt | llm | parser

def manner_guard_agent(state: TranslateState) -> TranslateState:
    if not llm: 
        state['is_safe'] = True
        return state

    src = state.get("corrected_stt") or state.get("stt_text", "")
    tgt = state.get("final_answer", "")
    
    src_lang = state.get("stt_lang", "Korean")
    tgt_lang = state.get("target_lang", "English")
    
    found_memes = state.get("found_terms", [])
    whitelist_str = ", ".join([f"'{item['term']}'" for item in found_memes])
    
    if not src.strip() or not tgt.strip(): 
        state['is_safe'] = True
        return state

    log.info(f"🛡️ Guard Check: '{src}' <-> '{tgt}'")

    try:
        res = chain.invoke({
            "source_text": src, 
            "target_text": tgt,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "whitelist": whitelist_str
        })
        
        severity = res.get("severity", "SAFE")
        reason = res.get("reason", "NONE")
        
        if severity == "BLOCK":
            state['is_safe'] = False
            state['corrected_stt'] = get_block_msg(src_lang)
            state['final_answer'] = get_block_msg(tgt_lang)
            log.warning(f"🚨 [BLOCK] {reason} (Src: {src})")
            
        elif severity == "WORD_MASK":
            state['is_safe'] = False
            state['corrected_stt'] = res.get("filtered_source_text", src)
            state['final_answer'] = res.get("filtered_target_text", tgt)
            log.warning(f"⚠️ [MASK] {reason}")
            
        else:
            state['is_safe'] = True
            state['corrected_stt'] = res.get("filtered_source_text", src)
            state['final_answer'] = res.get("filtered_target_text", tgt)
            log.info(f"✅ [SAFE] {reason}")

        state['safety_reason'] = f"{severity}:{reason}"
        

    except Exception as e:
        log.error(f"Guard Error: {e}")
        state['is_safe'] = True
    
    state['final_emoji'] = None
    
    return state
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

class SafetyCheckResult(BaseModel):
    severity: str = Field(description="One of: SAFE, WORD_MASK, BLOCK")
    filtered_source_text: str
    filtered_target_text: str
    blocking_message_source: str
    blocking_message_target: str
    reason: str

parser = JsonOutputParser(pydantic_object=SafetyCheckResult)

# ⭐️ [핵심 수정] 혐오와 비판을 명확히 구분하는 프롬프트
GUARD_PROMPT = """You are a rational Content Moderator.
Your goal is to block Hate Speech while protecting Free Speech (Criticism, Anger, Satire).

### 🛡️ WHITELIST (Authorized Memes):
Words in [{whitelist}] are ALWAYS SAFE.

### 🚨 DECISION GUIDELINES (Read Carefully):

**1. WHAT IS "SAFE"? (Do NOT Block)**
- **Criticism/Boycott:** "I will boycott Clova", "This service sucks", "Worst movie ever". -> **SAFE**
- **Personal Anger:** "I'm so pissed off", "How annoying". -> **SAFE**
- **Metaphors:** "You are a potato", "He is a snake". -> **SAFE**
- **Simple Insults:** "Idiot", "Stupid", "Fool". -> **SAFE** (Do not block, at most mask if vulgar)

**2. WHAT IS "WORD_MASK"? (Partial Censorship)**
- Only specific vulgar slurs (e.g., F*ck, Sh*t, 씨발, 개새끼).
- Action: Replace ONLY the swear word with `***`. The sentence remains visible.

**3. WHAT IS "BLOCK"? (Total Ban)**
- **Hate Speech:** Attacks based on **Race, Gender, Religion, Disability, Sexual Orientation**. (e.g., "Kill all [Group]", "Dirty [Race]").
- **Sexual Violence/Harassment:** Explicit sexual descriptions or threats.
- **Realistic Threats:** "I will kill you and your family".

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
    
    # 밈 화이트리스트 주입
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
        
        final_src = res.get("filtered_source_text", src)
        final_tgt = res.get("filtered_target_text", tgt)
        
        if severity == "BLOCK":
            state['is_safe'] = False
            # 차단 메시지 생성
            state['corrected_stt'] = res.get("blocking_message_source", "🚫 부적절한 메시지입니다.")
            state['final_answer'] = res.get("blocking_message_target", "🚫 Message hidden.")
            log.warning(f"🚨 [BLOCK] {reason} (Src: {src})")
            
        elif severity == "WORD_MASK":
            state['is_safe'] = False
            state['corrected_stt'] = final_src
            state['final_answer'] = final_tgt
            log.warning(f"⚠️ [MASK] {reason}")
            
        else:
            state['is_safe'] = True
            # SAFE라도 필터링 텍스트가 있으면 반영 (마스킹 없는 수정 등)
            state['corrected_stt'] = final_src
            state['final_answer'] = final_tgt
            log.info(f"✅ [SAFE] {reason}")

        state['safety_reason'] = f"{severity}:{reason}"

    except Exception as e:
        log.error(f"Guard Error: {e}")
        state['is_safe'] = True
        
    return state
# src/talaria/hermes_service/nodes/node_vibe_enrich.py
import logging
import json
import os
from ..state import TranslateState
from typing import Dict, List, Any, Optional
from ...core.database import get_db_cursor 
import re # ⭐️ 정규식(Regex) 임포트

log = logging.getLogger(__name__)

# --- 1. K-Culture Rule Caching (Singleton Pattern) ---

# ⭐️ MannerChecker.MannerRule의 핵심 로직을 차용
class _CultureRule:
    """
    K-Culture DB 규칙을 메모리에 캐시하고 'contains' 검사를 수행하는 헬퍼 클래스
    """
    def __init__(self, rule_data: Dict[str, Any]):
        self.pattern_text: str = rule_data.get('pattern_text', '')
        self.pattern_type: str = rule_data.get('pattern_type', 'contains')
        self.case_sensitive: bool = bool(rule_data.get('case_sensitive', 0))
        self.explanation_tpl: str = rule_data.get('explanation_tpl', '')
        self.suggestion_tpl: str = rule_data.get('suggestion_tpl', '') # 힌트
        
        # MannerChecker의 _compile 로직과 동일
        self._regex: Optional[re.Pattern] = None
        self._pattern_ci: Optional[str] = None # Case-Insensitive (소문자) 패턴
        self._pattern_cs: Optional[str] = None # Case-Sensitive (원본) 패턴
        
        if self.pattern_type == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                self._regex = re.compile(self.pattern_text, flags=flags)
            except re.error as e:
                log.error(f"Regex 컴파일 실패 (Rule: {self.pattern_text}): {e}")
        else:
            # 'contains' 또는 'exact'
            if self.case_sensitive:
                self._pattern_cs = self.pattern_text
            else:
                self._pattern_ci = self.pattern_text.lower()
                
    def _norm_text(self, text: str) -> str:
         return text if self.case_sensitive else text.lower()

    # MannerChecker._find_matches의 핵심 로직
    def find_match(self, text: str) -> Optional[str]:
        """주어진 텍스트에서 이 규칙의 패턴을 찾습니다."""
        t = self._norm_text(text) # STT 텍스트를 소문자화 (필요시)
        
        if self.pattern_type == "contains":
            pat = self._pattern_ci if not self.case_sensitive else self._pattern_cs
            # ⭐️ "킹받"(pat) in "아 진짜 킹받네"(t) -> True
            if pat and pat in t: 
                return self.pattern_text # 매칭된 원본 "킹받" 반환
        
        elif self.pattern_type == "regex" and self._regex:
            match = self._regex.search(text)
            if match:
                return match.group(0) # 정규식 매칭 부분 반환

        # (기타: exact 등)
        return None

class _CultureChecker:
    """
    K-Culture DB 규칙을 시작 시 메모리에 로드하고 'contains' 검사를 제공하는 싱글톤
    """
    _instance = None
    _rules: List[_CultureRule] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_CultureChecker, cls).__new__(cls)
            cls._instance._load_rules_from_db()
        return cls._instance

    def _load_rules_from_db(self):
        log.info("K-Culture DB 규칙 로드를 시작합니다...")
        rules_data = []
        try:
            with get_db_cursor(dictionary=True) as cursor:
                TABLE_NAME = "tl_culture_dict" 
                
                # ⭐️ [수정] image_45d967.png의 K-meme, K-game 등
                sql = f"""
                    SELECT pattern_text, pattern_type, case_sensitive, 
                           explanation_tpl, suggestion_tpl
                    FROM {TABLE_NAME}
                    WHERE active = 1
                """
                cursor.execute(sql)
                rules_data = cursor.fetchall()
            
            # ⭐️ DB에서 가져온 룰을 파이썬 객체로 컴파일 (MannerChecker 방식)
            self._rules = [self.compile_rule(data) for data in rules_data]
            log.info(f"✅ K-Culture DB 규칙 {len(self._rules)}개 로드 및 캐시 완료.")
            
        except Exception as e:
            log.error(f"K-Culture DB 로드 실패: {e}", exc_info=True)
            
    def compile_rule(self, data: Dict[str, Any]) -> _CultureRule:
        return _CultureRule(data)

    def find_all_matches(self, corrected_stt: str) -> List[Dict[str, str]]:
        """LLM 1이 보정한 텍스트로 'contains' 검사를 수행"""
        if not corrected_stt:
            return []
            
        found_matches: List[Dict[str, str]] = []
        
        # 1. 캐시된 모든 룰을 순회 (DB 조회 X)
        for rule in self._rules:
            match_text = rule.find_match(corrected_stt)
            if match_text:
                # ⭐️ 찾은 룰의 정보를 반환
                found_matches.append({
                    "term": match_text,
                    "explanation": rule.explanation_tpl.format(match=match_text),
                    "hint": rule.suggestion_tpl or rule.explanation_tpl.split(".")[0]
                })
        
        return found_matches

# --- 2. Eager-loading (싱글톤 인스턴스 생성) ---
try:
    _checker_instance = _CultureChecker()
except Exception as e:
    log.critical(f"❌ _CultureChecker 인스턴스 생성 실패: {e}", exc_info=True)
    _checker_instance = None

# --- 3. EMOJI 맵 로드 (기존과 동일) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "..", "..", "assets")

def load_json_asset(filename: str) -> Dict:
    filepath = os.path.join(ASSETS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.error(f"❌ {filename} 로드 실패: {e}", exc_info=True)
        return {}
        
EMOJI_MAP = load_json_asset("emotion_emoji_map.json")

# --- 4. LangGraph 노드 함수 정의 ---

# ⭐️ [핵심 수정] context_enrich_agent (Contains 로직)
def context_enrich_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-2] (규칙: Contains)
    LLM 1이 보정한 'corrected_stt'를 받아, K-Culture DB(캐시)와 'contains' 비교
    """
    if not _checker_instance:
        log.error("Vibe 힌트: _CultureChecker가 초기화되지 않았습니다.")
        state['vibe_context'] = "No K-Culture rules loaded."
        return state

    # ⭐️ [수정] LLM 1이 보정한 STT를 가져옴
    corrected_stt = state.get("corrected_stt") or state.get("stt_text", "")
    
    # ⭐️ _checker_instance를 통해 'contains' 검사 실행
    matches = _checker_instance.find_all_matches(corrected_stt)
    
    if not matches:
        log.info("Vibe 힌트: 'contains' 매칭 결과 없음.")
        state['vibe_context'] = "No specific K-Culture terms identified."
        return state
        
    explanations_for_ui = [m['explanation'] for m in matches]
    hints_for_llm = [f"'{m['term']}' means '{m['hint']}'" for m in matches]

    state['final_explanation'] = " | ".join(explanations_for_ui)
    state['vibe_context'] = "Context: " + ". ".join(hints_for_llm)
    log.info(f"Vibe 힌트(LLM 2용) 생성 (Contains 기반): {state['vibe_context']}")

    # ⭐️ [신규] 프론트엔드에 전달할 수 있도록 'vibe_triggers'에도 저장
    state['vibe_triggers'] = [m['term'] for m in matches]

    return state

# [수정 없음] emoji_agent
def emoji_agent(state: TranslateState) -> TranslateState:
    """
    [Vibe Mode - B-3] (규칙 기반)
    LLM 1이 분석한 'vibe_emotion_tag'를 'emotion_emoji_map.json'과 매핑합니다.
    """
    emotion = state.get("vibe_emotion_tag", "default")

    if emotion == "neutral" or emotion == "default":
        state['final_emoji'] = None
        log.info(f"Vibe 이모지 매핑: {emotion} -> (이모지 없음)")
        return state

    # 그 외의 감정(joy, anger 등)일 때만 이모지 매핑
    emoji = EMOJI_MAP.get(emotion, EMOJI_MAP.get("default", "💬"))
    
    # 'neutral'이 아닌데도 'default'(💬)가 매핑되는 경우 방지 (선택적)
    if emoji == EMOJI_MAP.get("default"):
         state['final_emoji'] = None
         log.warning(f"Vibe 이모지 매핑: {emotion}에 해당하는 이모지가 없어 건너뜁니다.")
    else:
        state['final_emoji'] = emoji
        log.info(f"Vibe 이모지 매핑: {emotion} -> {emoji}")
    
    return state
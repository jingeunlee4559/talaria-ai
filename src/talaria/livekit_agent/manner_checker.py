# src/talaria/livekit_agent/manner_checker.py
import asyncio
import aiohttp
import logging
import time
import fnmatch
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class MannerRule:
    rule_id: int
    active: int
    agent_code: str
    label: str
    severity: int
    weight: float
    priority: int
    pattern_type: str           # 'regex' | 'contains' | 'exact' | 'glob'
    pattern_text: str
    case_sensitive: int
    negate: int
    speaker_locale: Optional[str]
    listener_locale: Optional[str]
    culture_pair: Optional[str]
    explanation_tpl: str
    suggestion_tpl: str
    cooldown_sec: int
    version_tag: str

    # 내부 캐시(성능)
    _regex: Optional[re.Pattern] = None
    _pattern_ci: Optional[str] = None
    _pattern_cs: Optional[str] = None

    def compile(self):
        if self.pattern_type == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._regex = re.compile(self.pattern_text, flags=flags)
        else:
            if self.case_sensitive:
                self._pattern_cs = self.pattern_text
            else:
                self._pattern_ci = self.pattern_text.lower()


def _norm_text(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


class MannerChecker:
    """
    - 백엔드에서 룰을 주기적으로 가져와 로컬 캐시
    - 텍스트 매칭 후, 룰 hit마다 백엔드 쿨다운 API 호출
    - 허용(allowed=True)이면 DataChannel로 알림 보낼 수 있도록 payload 반환
    """
    def __init__(
        self,
        base_url: str,
        api_token: Optional[str] = None,
        refresh_sec: int = 30,
        agent_codes: Optional[List[str]] = None,
        app_locale_prefix: Optional[str] = None,
        target_locale_prefix: Optional[str] = None,
        culture_pair: Optional[str] = None,
        http_timeout_sec: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.refresh_sec = max(5, refresh_sec)
        self.agent_codes = agent_codes or []
        self.app_locale_prefix = app_locale_prefix
        self.target_locale_prefix = target_locale_prefix
        self.culture_pair = culture_pair
        self.http_timeout_sec = http_timeout_sec

        self._rules: List[MannerRule] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._stop = asyncio.Event()

    async def start(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout_sec))
        await self.refresh_rules()
        asyncio.create_task(self._auto_refresh())

    async def close(self):
        self._stop.set()
        if self._session:
            await self._session.close()
            self._session = None

    async def _auto_refresh(self):
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.refresh_sec)
                await self.refresh_rules()
            except asyncio.CancelledError:
                break
            except Exception:
                logging.exception("[MannerChecker] auto refresh failed")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_token:
            h["Authorization"] = f"Bearer {self.api_token}"
        return h

    async def refresh_rules(self):
        """
        GET /api/manner/rules?active=1&locale_prefix=ko&agent_code=direct&agent_code=address...
        backend는 page/size가 있다면 큰 size로 가져오도록 구현되어 있다고 가정
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout_sec))

        params = {"active": "1"}
        if self.app_locale_prefix:
            params["locale_prefix"] = self.app_locale_prefix
        for code in self.agent_codes:
            params.setdefault("agent_code", [])
            params["agent_code"].append(code)

        url = f"{self.base_url}/api/manner/rules"
        try:
            async with self._session.get(url, headers=self._headers(), params=params) as resp:
                if resp.status != 200:
                    logging.error(f"[MannerChecker] rules fetch failed: {resp.status}")
                    return
                data = await resp.json()
        except Exception:
            logging.exception("[MannerChecker] rules fetch exception")
            return

        # 백엔드 응답 스키마: { items: [...], page:..., size:..., total:... } 또는 list로 직반환
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            logging.warning("[MannerChecker] rules response has no items")
            return

        rules: List[MannerRule] = []
        for r in items:
            try:
                mr = MannerRule(
                    rule_id=int(r["rule_id"]),
                    active=int(r.get("active", 1)),
                    agent_code=r.get("agent_code") or "",
                    label=r.get("label") or "",
                    severity=int(r.get("severity", 1)),
                    weight=float(r.get("weight", 1.0)),
                    priority=int(r.get("priority", 100)),
                    pattern_type=r.get("pattern_type") or "contains",
                    pattern_text=r.get("pattern_text") or "",
                    case_sensitive=int(r.get("case_sensitive", 0)),
                    negate=int(r.get("negate", 0)),
                    speaker_locale=r.get("speaker_locale"),
                    listener_locale=r.get("listener_locale"),
                    culture_pair=r.get("culture_pair"),
                    explanation_tpl=r.get("explanation_tpl") or "",
                    suggestion_tpl=r.get("suggestion_tpl") or "",
                    cooldown_sec=int(r.get("cooldown_sec", 0)),
                    version_tag=r.get("version_tag") or "v1",
                )
                mr.compile()
                rules.append(mr)
            except Exception:
                logging.exception("[MannerChecker] invalid rule item skipped")

        # 우선순위 순 정렬(낮을수록 우선)
        rules.sort(key=lambda x: x.priority)
        self._rules = rules
        logging.info(f"[MannerChecker] loaded rules: {len(self._rules)}")

    def _locale_accept(self, rule: MannerRule) -> bool:
        # speaker_locale prefix
        if rule.speaker_locale and self.app_locale_prefix:
            if not self.app_locale_prefix.startswith(rule.speaker_locale):
                return False
        # listener_locale prefix
        if rule.listener_locale and self.target_locale_prefix:
            if not self.target_locale_prefix.startswith(rule.listener_locale):
                return False
        # culture_pair exact
        if rule.culture_pair and self.culture_pair:
            if rule.culture_pair != self.culture_pair:
                return False
        return True

    def _find_matches(self, text: str) -> List[Tuple[MannerRule, str]]:
        out: List[Tuple[MannerRule, str]] = []
        for rule in self._rules:
            if not rule.active:
                continue
            if not self._locale_accept(rule):
                continue

            t = _norm_text(text, bool(rule.case_sensitive))
            m: Optional[str] = None

            if rule.pattern_type == "regex" and rule._regex:
                match = rule._regex.search(text)
                if match:
                    m = match.group(0)
            elif rule.pattern_type == "contains":
                pat = rule._pattern_ci if not rule.case_sensitive else rule._pattern_cs
                if pat and pat in t:
                    # best-effort match text
                    m = rule.pattern_text
            elif rule.pattern_type == "exact":
                pat = rule._pattern_ci if not rule.case_sensitive else rule._pattern_cs
                if pat and t == pat:
                    m = rule.pattern_text
            elif rule.pattern_type == "glob":
                # fnmatch는 기본적으로 case-sensitive. 필요시 소문자화
                pat = rule.pattern_text if rule.case_sensitive else rule.pattern_text.lower()
                if fnmatch.fnmatchcase(t, pat):
                    m = rule.pattern_text

            if m:
                # negate==1이면 '매칭 시 제외' 규칙 → 히트에서 제거
                if rule.negate:
                    continue
                out.append((rule, m))

        return out

    @staticmethod
    def room_name_to_session_id(room_name: str) -> int:
        """meetingId 없을 때 임시 세션ID 생성(충돌 최소화)"""
        h = hashlib.sha1(room_name.encode("utf-8")).hexdigest()
        return int(h[:12], 16)  # 48bit 정도

    async def _cooldown_allowed(self, rule_id: int, session_id: Optional[int], speaker_id: Optional[str]) -> bool:
        """
        POST /api/manner/events/check-and-mark
        body: { ruleId, sessionId, speakerId }
        resp: { allowed: bool, lastAlertAt?: string }
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout_sec))
        url = f"{self.base_url}/api/manner/events/check-and-mark"
        payload = {
            "ruleId": rule_id,
            "sessionId": session_id,
            "speakerId": speaker_id,
        }
        try:
            async with self._session.post(url, json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    logging.warning(f"[MannerChecker] cooldown check failed status={resp.status}")
                    return True  # 백엔드 오류 시 알림 자체는 허용(관측 가능성 높이기)
                data = await resp.json()
                return bool(data.get("allowed", True))
        except Exception:
            logging.exception("[MannerChecker] cooldown check exception")
            return True

    def _render(self, tpl: str, match_text: str) -> str:
        try:
            return tpl.format(match=match_text)
        except Exception:
            return tpl

    async def process_text(
        self,
        text: str,
        *,
        room_name: str,
        speaker_identity: Optional[str],
        speaker_username: Optional[str],
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        텍스트 하나에 대해 룰 매칭 후, 쿨다운 허용된 알림 payload 리스트 반환
        (호출 측에서 DataChannel publish)
        """
        res: List[Dict[str, Any]] = []
        if not text:
            return res

        session_id = self.room_name_to_session_id(room_name)
        hits = self._find_matches(text)

        for rule, mtxt in hits:
            if rule.cooldown_sec > 0:
                allowed = await self._cooldown_allowed(rule.rule_id, session_id, speaker_identity)
                if not allowed:
                    continue

            explanation = self._render(rule.explanation_tpl, mtxt)
            suggestion = self._render(rule.suggestion_tpl, mtxt)
            score = round(rule.severity * rule.weight, 2)

            payload = {
                "type": "manner_alert",
                "rule_id": rule.rule_id,
                "label": rule.label,
                "agent_code": rule.agent_code,
                "severity": rule.severity,
                "weight": rule.weight,
                "score": score,
                "priority": rule.priority,
                "match": mtxt,
                "explanation": explanation,
                "suggestion": suggestion,
                "identity": speaker_identity or "",
                "username": speaker_username or (speaker_identity or ""),
                "language": language or "",
                "room_name": room_name,
                "ts": time.time(),
                "version": rule.version_tag,
            }
            res.append(payload)

        return res

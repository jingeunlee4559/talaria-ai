# src/talaria/livekit_agent/handler.py
import json
import asyncio
import logging
import time
import uuid
from typing import Optional, Any, List, Dict, Set

import httpx
from livekit import agents, rtc
from talaria.livekit_agent.manner_checker import MannerChecker
from .utils import locale_to_clova_lang, locale_to_papago_lang

from .utils import locale_to_clova_lang
from .integrations import (
    translate_to_multiple_languages,
    http_client,
    HERMES_SERVICE_URL
)

log = logging.getLogger(__name__)

# [신규 함수] 텍스트 기반 언어 감지 (간이 LID)
def detect_dominant_language(text: str, default_locale: str) -> str:
    """
    텍스트 기반 언어 감지 (Papago용 코드 반환)
    - 한글 포함 시 'ko'
    - 아니면 default_locale의 언어 코드 사용
    - ⭐️ 절대 'enko'를 반환하지 않도록 방어
    """
    # 1. 텍스트가 없으면 기본값
    if not text or not text.strip():
        lang = locale_to_papago_lang(default_locale)
        return "ko" if lang == "enko" else lang

    # 2. 한글 감지 (AC00-D7A3)
    korean_char_count = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
    
    if korean_char_count > 0:
        return "ko"
        
    # 3. 한글이 없으면 기본 Locale 따라감
    detected_lang = locale_to_papago_lang(default_locale)
    
    # 🚨 [핵심 안전장치] 만약 유틸 함수가 'enko'를 뱉으면 강제로 보정
    if detected_lang == "enko":
        # enko가 떴다는 건, 한글이 없는데 enko라는 뜻 -> 영어로 간주
        return "en"
        
    return detected_lang


class ParticipantSTTHandler:
    # ( ... __init__, start, _process_stream, _get_current_locales, _get_metadata_field 함수는 모두 동일 ... )
    def __init__(
        self,
        participant,
        track,
        participant_locale: str,
        stt_model,
        room,
        publish_callback,
        *,
        manner_checker: Optional[MannerChecker] = None,
        publish_manner_callback=None,
        translation_enabled: bool = False,
        supported_locales: Set[str] = None,
    ):
        self.participant = participant
        self.track = track
        self.participant_locale = participant_locale
        self.stt = stt_model
        self.room = room
        self.publish_callback = publish_callback
        
        # ⭐️ [수정] 비즈니스용 매너 체커로 이름 명시
        self.mchecker_business = manner_checker
        self.publish_manner_callback = publish_manner_callback
        
        self.translation_enabled = translation_enabled
        self.supported_locales = supported_locales or set()
        clova_lang = locale_to_clova_lang(participant_locale)
        log.info(f"🎯 ParticipantSTTHandler: locale={participant_locale}, clova_lang={clova_lang}, translation={translation_enabled}")
        self.stream = None
        self.task = None
        self._is_running = True

    async def start(self):
        # ( ... 기존 코드와 동일 ... )
        audio_stream = rtc.AudioStream(self.track)
        pid = getattr(self.participant, "identity", "unknown")
        try:
            while self._is_running:
                log.info(f"[{pid}] 🔄 Starting new STT stream...")
                self.stream = self.stt.stream()
                self.task = asyncio.create_task(self._process_stream())
                try:
                    async for event in audio_stream:
                        if not self._is_running:
                            break
                        if self.stream:
                            self.stream.push_frame(event.frame)
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "is closed" in msg:
                        log.warning(f"[{pid}] ⚠️ Stream closed (likely silence). Restarting...")
                    else:
                        log.error(f"[{pid}] Audio push runtime error: {msg}", exc_info=True)
                        raise
                await self._cleanup_stream_task()
                if not self._is_running:
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            log.info(f"[{pid}] STT handler task cancelled")
        except Exception as e:
            log.error(f"[{pid}] STT handler failed catastrophically: {e}", exc_info=True)
        finally:
            log.info(f"[{pid}] 🛑 Cleaning up STT handler...")
            self._is_running = False
            await self._cleanup_stream_task()
            await audio_stream.aclose()

    async def _process_stream(self):
        # ( ... 기존 코드와 동일 ... )
        pid = getattr(self.participant, "identity", "unknown")
        try:
            async for event in self.stream:
                if not self._is_running:
                    break
                await self._handle_transcription(event)
        except StopAsyncIteration:
            log.info(f"[{pid}] STT stream ended normally")
        except Exception as e:
            error_msg = str(e)
            try:
                if "Unexpected response" in error_msg and "'text': ''" in error_msg:
                    log.debug(f"[{pid}] 💤 Clova: empty response (silence)")
                elif "Confidence" in error_msg and "bellow threshold" in error_msg:
                    log.debug(f"[{pid}] ⚠️ low-confidence result ignored: {error_msg}")
                else:
                    log.error(f"[{pid}] ❌ STT stream error: {error_msg}")
            except Exception as log_error:
                log.error(f"[{pid}] 🚨 error while logging: {log_error}")

    def _get_current_locales(self) -> Set[str]:
        # ( ... 기존 코드와 동일 ... )
        locales = set()
        if self.room.local_participant and self.room.local_participant.metadata:
            try:
                meta = json.loads(self.room.local_participant.metadata)
                if "appLocale" in meta:
                    locales.add(meta["appLocale"])
            except:
                pass
        for participant in self.room.remote_participants.values():
            if participant.metadata:
                try:
                    meta = json.loads(participant.metadata)
                    if "appLocale" in meta:
                        locales.add(meta["appLocale"])
                except:
                    pass
        return locales.intersection(self.supported_locales)

    def _get_metadata_field(self, field_name: str, default: Any = None) -> Any:
        # ( ... 기존 코드와 동일 ... )
        try:
            if self.participant.metadata:
                meta = json.loads(self.participant.metadata)
                return meta.get(field_name, default)
        except Exception:
            log.warning(f"[{self.participant.identity}] metadata 파싱 실패")
        return default


    # ⭐️ [수정] Task 1: 번역 + "Vibe 모드 설명 발행"
    async def _run_translation_pipeline(self, text: str, username: str, confidence: float, start_time_ms: int, end_time_ms: int, **kwargs):
        """
        Task 1: (Papago) -> (1차 Publish) -> (Hermes) -> (2차 Publish)
        """
        pid = getattr(self.participant, "identity", "unknown")
        
        try:
            # --- 0. 고유 발화 ID 생성 ---
            utterance_id = f"utt_{self.participant.sid}_{start_time_ms}_{str(uuid.uuid4())[:4]}"

            # text(STT 결과)를 보고, 한글이 섞여 있으면 'ko', 아니면 원래 locale 언어로 설정
            real_source_lang_code = detect_dominant_language(text, self.participant_locale)
            
            log.debug(f"[{pid}] 🔍 Lang Detect: User({self.participant_locale}) -> Detected({real_source_lang_code}) | Text: {text[:20]}...")

            # --- 1. 모드 및 대상 언어 결정 ---
            mode_flag = self._get_metadata_field("meeting_mode", "B") # 'B' or 'C'
            translation_mode = "vibe" if mode_flag == "C" else "business"
            log.info(f"[{pid}] 🎤 Meeting Mode Detected: {mode_flag} -> {translation_mode}")

            current_locales = self._get_current_locales()
            target_locales_list = [loc for loc in current_locales if loc != self.participant_locale]

            log.info(f"[{pid}] 🔍 Lang Check: Text='{text}' | UserLocale={self.participant_locale} | DetectedSource={real_source_lang_code} | Targets={target_locales_list}")
            
            primary_target_lang = "en-US" 
            if target_locales_list:
                primary_target_lang = target_locales_list[0] 
            
            log.debug(f"[{pid}] Primary Target(LLM): {primary_target_lang} | Papago Targets: {target_locales_list}")

            # --- 2. Papago 1차 번역 ---
            simple_translations = {}
            
            # ⭐️ [수정] Vibe 모드(elif) 분기 제거, if 조건 단순화
            if self.translation_enabled and target_locales_list:
                log.debug(f"[{pid}] Running Papago (1st translation) for mode: {translation_mode}...")
                try:
                    simple_translations = await translate_to_multiple_languages(
                        text, 
                        real_source_lang_code,
                        target_locales_list 
                    )
                except Exception as trans_err:
                    log.error(f"❌ Papago (1st translation) error: {trans_err}", exc_info=True)
            else:
                 log.debug(f"[{pid}] Papago (1st translation) skipped (enabled={self.translation_enabled}, targets={len(target_locales_list)})")
            
            # --- 3. 1차 게시 (즉시) ---
            log.info(f"[{pid}] 1st Publish (Papago): {utterance_id}")
            await self.publish_callback(
                text=text,  # 원본 STT
                final=True,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                locale=self.participant_locale,
                translations=simple_translations,  # ⭐️ 1차 Papago 번역본
                utterance_id=utterance_id         # ⭐️ 고유 ID
            )

            # --- 3. hermes_service로 [문맥 보정 + DB 저장] 요청 ---
            meeting_id = self._get_metadata_field("meetings_id")
            final_translations = simple_translations 
            final_emoji = None
            corrected_stt_from_hermes = None
            
            if not meeting_id:
                log.error(f"❌ hermes_service 호출 중단: meeting_id를 metadata에서 찾을 수 없습니다.")
                return
            else:
                try:
                    payload = {
                        "meeting_id": int(meeting_id),
                        "speaker_id": username,
                        "track_sid": self.track.sid,
                        "stt_text": text,
                        "lang": real_source_lang_code,
                        "confidence": confidence,
                        "started_ms": start_time_ms,
                        "ended_ms": end_time_ms,
                        "simple_translations": simple_translations, 
                        "target_lang": primary_target_lang,
                        "translation_mode": translation_mode
                    }
                    
                    response = await http_client.post(
                        f"{HERMES_SERVICE_URL}/utterances/process",
                        json=payload
                    )
                    response.raise_for_status()
                    result_data = response.json()
                    
                    # hermes 결과를 변수에 저장
                    final_translations = result_data.get("contextual_translations", simple_translations)
                    final_emoji = result_data.get("final_emoji")
                    corrected_stt_from_hermes = result_data.get("corrected_stt")
                    vibe_triggers = result_data.get("vibe_triggers")
                    final_explanation = result_data.get("final_explanation")
                    
                    #  Vibe 모드이고, 설명이 있고, 콜백이 존재하면
                    # "밈 관련 정보"을 'topic=manner'로 발행
                    if translation_mode == "vibe" and final_explanation and self.publish_manner_callback:
                        log.info(f"[{pid}] ⭐️ Vibe '문화 설명'을 topic=manner로 발행합니다.")
                        match_text = vibe_triggers[0] if vibe_triggers else ""
                        # (MannerChecker 페이로드와 유사하게 생성)
                        fake_payload = {
                            "type": "manner_alert",
                            "label": "K-Culture Tip", # (프론트가 표시할 라벨)
                            "explanation": final_explanation, # (LangGraph가 DB에서 생성한 설명)
                            
                            # (나머지 필드는 MannerChecker와 동일하게 채움)
                            "agent_code": "K-meme",
                            "severity": 1,
                            "match": match_text,
                            "suggestion": "",
                            "identity": self.participant.identity,
                            "username": username,
                            "language": self.participant_locale,
                            "room_name": self.room.name,
                            "ts": time.time(),
                            "version": "v1-culture",
                        }
                        await self.publish_manner_callback(fake_payload)

                except httpx.HTTPStatusError as e:
                    log.error(f"❌ hermes_service API 오류: {e.response.status_code} - {e.response.text}")
                    final_translations = simple_translations
                except Exception as e:
                    log.error(f"❌ hermes_service 호출 실패: {e}")
                    final_translations = simple_translations

            # --- 5.  2차 게시 (보정본 업데이트) ---
            final_text_to_publish = corrected_stt_from_hermes if corrected_stt_from_hermes is not None else text
            
            if final_emoji:
                final_text_to_publish = f"{final_text_to_publish} {final_emoji}"
                for lang_code in final_translations:
                    final_translations[lang_code] = f"{final_translations[lang_code]} {final_emoji}"

            final_translations.pop(self.participant_locale, None)

            log.info(f"[{pid}] 2nd Publish (LLM Corrected): {utterance_id}")
            await self.publish_callback(
                text=final_text_to_publish,        # 보정된 STT + 이모지
                final=True,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                locale=self.participant_locale,
                translations=final_translations, # LLM 보정본 + 이모지
                utterance_id=utterance_id         # 1차와 동일한 ID
            )
        except Exception as e:
            log.error(f"❌ [Translation Pipeline] Failed: {e}", exc_info=True)


    # ⭐️ [수정] Task 2: "비즈니스 모드" 매너 체크 파이프라인
    async def _run_manner_check_pipeline(self, text: str, username: str, **kwargs):
        """
        Task 2: STT -> MannerCheck -> Publish (Manner)
        ⭐️ (신규) 'business' 모드일 때만 실행됩니다.
        """
        pid = getattr(self.participant, "identity", "unknown")

        # --- 1. 모드 감지 (Task 1과 동일) ---
        mode_flag = self._get_metadata_field("meeting_mode", "B") # 'B' or 'C'
        translation_mode = "vibe" if mode_flag == "C" else "business"
        
        # ⭐️ [핵심 수정] Vibe 모드이면, 비즈니스 매너 체커를 실행하지 않음
        if translation_mode == "vibe":
            log.debug(f"[{pid}] Vibe 모드이므로 Business 매너 체크를 건너뜁니다.")
            return

        # --- (이하는 기존과 동일) Business 모드일 때만 실행 ---
        if not (self.mchecker_business and self.publish_manner_callback):
            return 
            
        log.debug(f"[{pid}] Business 매너 체크 실행...")
        try:
            clova_lang = locale_to_clova_lang(self.participant_locale) 
            alerts = await self.mchecker_business.process_text(
                text=text,
                room_name=self.room.name,
                speaker_identity=self.participant.identity,
                speaker_username=username,
                language=clova_lang,
            )
            for payload in alerts:
                await self.publish_manner_callback(payload)
        except Exception:
            log.exception(f"[{pid}] manner checking failed")


    # [메인 핸들러 (병렬 실행)] (수정 없음)
    async def _handle_transcription(self, event: agents.stt.SpeechEvent):
        # ( ... 기존 코드와 동일 ... )
        pid = getattr(self.participant, "identity", "unknown")
        try:
            if not event.alternatives:
                return 
            alternative = event.alternatives[0]
            text = alternative.text
            confidence = alternative.confidence
            start_time_ms = int(alternative.start_time * 1000)
            end_time_ms = int(alternative.end_time * 1000)
            is_final = event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT
            if not text or not text.strip():
                return
            username = self._get_username()
            
            if not is_final:
                await self.publish_callback(
                    text=text, final=False,
                    identity=self.participant.identity,
                    psid=self.participant.sid,
                    username=username,
                    locale=self.participant_locale,
                    translations={},
                )
                return

            task_args = {
                "text": text,
                "username": username,
                "confidence": confidence,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms
            }
            await asyncio.gather(
                self._run_translation_pipeline(**task_args),
                self._run_manner_check_pipeline(**task_args)
            )
        except Exception as e:
            log.error(f"[{pid}] 🧩 _handle_transcription 처리 중 심각한 오류: {e}", exc_info=True)

    # ( ... _get_username, _cleanup_stream_task, stop 함수는 모두 동일 ... )
    def _get_username(self) -> str:
        try:
            metadata = json.loads(self.participant.metadata or '{}')
            if "username" in metadata:
                return metadata["username"]
        except Exception:
            pass
        return self.participant.name or self.participant.identity

    async def _cleanup_stream_task(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        if self.stream:
            try:
                await self.stream.aclose()
            except Exception:
                pass
            self.stream = None

    async def stop(self):
        pid = getattr(self.participant, "identity", "unknown")
        log.info(f"[{pid}] Received stop signal. Cleaning up...")
        self._is_running = False
        await self._cleanup_stream_task()
        log.info(f"[{pid}] STT handler stopped cleanly")
# src/talaria/livekit_agent/handler.py
import json
import asyncio
import logging
import time
from typing import Optional, Any, List, Dict, Set

import httpx
from livekit import agents, rtc
from talaria.livekit_agent.manner_checker import MannerChecker

from .utils import locale_to_clova_lang
from .integrations import (
    translate_to_multiple_languages,
    http_client,
    HERMES_SERVICE_URL
)

log = logging.getLogger(__name__)


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
        self.mchecker = manner_checker
        self.publish_manner_callback = publish_manner_callback
        self.translation_enabled = translation_enabled
        self.supported_locales = supported_locales or set()
        clova_lang = locale_to_clova_lang(participant_locale)
        log.info(f"🎯 ParticipantSTTHandler: locale={participant_locale}, clova_lang={clova_lang}, translation={translation_enabled}")
        self.stream = None
        self.task = None
        self._is_running = True

    async def start(self):
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
        try:
            if self.participant.metadata:
                meta = json.loads(self.participant.metadata)
                return meta.get(field_name, default)
        except Exception:
            log.warning(f"[{self.participant.identity}] metadata 파싱 실패")
        return default


    # ⭐️ [수정] Task 1: 번역 파이프라인 (Autumn님 요청 - "첫 번째 언어만 보정")
    async def _run_translation_pipeline(self, text: str, username: str, confidence: float, start_time_ms: int, end_time_ms: int, **kwargs):
        """
        Task 1: STT -> (Papago) -> Hermes(LangGraph) -> Publish (STT/Translation)
        """
        pid = getattr(self.participant, "identity", "unknown")
        
        try:
            # --- 1. 모드 및 대상 언어 결정 ---
            mode_flag = self._get_metadata_field("meeting_mode", "B") # 'B' or 'C'
            translation_mode = "vibe" if mode_flag == "C" else "business"
            log.info(f"[{pid}] 🎤 Meeting Mode Detected: {mode_flag} -> {translation_mode}")

            # [핵심 수정] 
            # 1. Papago는 모든 대상 언어로 번역
            # 2. LLM 보정은 "목록의 첫 번째" 언어로 지정
            
            current_locales = self._get_current_locales()
            target_locales_list = [loc for loc in current_locales if loc != self.participant_locale]
            
            # (규칙) 1. 목록이 있으면, 첫 번째 언어를 LLM 보정 대상으로
            #       2. 목록이 없으면, "en-US"를 기본값으로
            
            primary_target_lang = "en-US" # (기본값)
            if target_locales_list:
                primary_target_lang = target_locales_list[0] # ⭐️ 목록의 첫 번째 언어
            
            log.debug(f"[{pid}] Primary Target(LLM): {primary_target_lang} | Papago Targets: {target_locales_list}")

            # --- 2. Papago 1차 번역 (Business 모드일 때만 호출) ---
            simple_translations = {}
            if self.translation_enabled and translation_mode == "business":
                log.debug(f"[{pid}] Running Papago (Business Mode 'Fast Path')...")
                if target_locales_list:
                    try:
                        # ⭐️ Papago는 목록의 '모든' 언어로 번역
                        simple_translations = await translate_to_multiple_languages(
                            text, 
                            self.participant_locale,
                            target_locales_list 
                        )
                    except Exception as trans_err:
                        log.error(f"❌ Papago 번역 중 예외: {trans_err}", exc_info=True)
            
            elif self.translation_enabled and translation_mode == "vibe":
                log.debug(f"[{pid}] Skipping Papago (Vibe Mode). LLM will translate.")
            
            # --- 3. hermes_service로 [문맥 보정 + DB 저장] 요청 ---
            meeting_id = self._get_metadata_field("meetings_id")
            
            if not meeting_id:
                log.error(f"❌ hermes_service 호출 중단: meeting_id를 metadata에서 찾을 수 없습니다.")
                final_translations = simple_translations
            else:
                try:
                    payload = {
                        "meeting_id": int(meeting_id),
                        "speaker_id": self.participant.identity,
                        "track_sid": self.track.sid,
                        "stt_text": text,
                        "lang": self.participant_locale,
                        "confidence": confidence,
                        "started_ms": start_time_ms,
                        "ended_ms": end_time_ms,
                        "simple_translations": simple_translations, 
                        
                        # ⭐️ [핵심 수정] 목록의 첫 번째 언어를 보정 대상으로 전달
                        "target_lang": primary_target_lang,
                        "translation_mode": translation_mode
                    }
                    
                    response = await http_client.post(
                        f"{HERMES_SERVICE_URL}/utterances/process",
                        json=payload
                    )
                    response.raise_for_status()
                    
                    result_data = response.json()
                    
                    # ⭐️ [수정] hermes는 'primary_target_lang'만 보정했으므로, 
                    # 1차 번역본(simple_translations)을 기반으로 덮어써야 함
                    final_translations = simple_translations.copy()
                    
                    # LLM이 보정한 'final_answer' (e.g., 'zh-CN')를 가져옴
                    corrected_answer = result_data.get("contextual_translations", {}).get(primary_target_lang)
                    final_emoji = result_data.get("final_emoji")

                    if corrected_answer:
                        if final_emoji:
                             corrected_answer = f"{corrected_answer} {final_emoji}"
                        
                        # ⭐️ 'primary_target_lang' 키의 값만 LLM 보정본으로 덮어쓰기
                        final_translations[primary_target_lang] = corrected_answer
                    
                    # (결과) final_translations = {'zh-CN': 'LLM_Result', 'ja-JP': 'Papago...'}

                except httpx.HTTPStatusError as e:
                    log.error(f"❌ hermes_service API 오류: {e.response.status_code} - {e.response.text}")
                    final_translations = simple_translations
                except Exception as e:
                    log.error(f"❌ hermes_service 호출 실패: {e}")
                    final_translations = simple_translations

            # 4. (중요) 최종 결과를 UI (DataChannel)로 전송
            await self.publish_callback(
                text=text,
                final=True,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                locale=self.participant_locale,
                translations=final_translations, # (1개 언어만 보정된 맵)
            )
        except Exception as e:
            log.error(f"❌ [Translation Pipeline] Failed: {e}", exc_info=True)


    # [Task 2: 매너 체크 파이프라인] (수정 없음)
    async def _run_manner_check_pipeline(self, text: str, username: str, **kwargs):
        pid = getattr(self.participant, "identity", "unknown")
        if not (self.mchecker and self.publish_manner_callback):
            return 
        try:
            clova_lang = locale_to_clova_lang(self.participant_locale) 
            alerts = await self.mchecker.process_text(
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
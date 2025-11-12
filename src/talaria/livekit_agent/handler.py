# src/talaria/livekit_agent/handler.py
import json
import asyncio
import logging
import time
from typing import Optional, Any, List, Dict, Set

import httpx
from livekit import agents, rtc
from talaria.livekit_agent.manner_checker import MannerChecker

# (신규) 분리된 파일에서 import
from .utils import locale_to_clova_lang
from .integrations import (
    translate_to_multiple_languages,
    http_client,
    HERMES_SERVICE_URL
)

log = logging.getLogger(__name__)


class ParticipantSTTHandler:
    """Handle realtime STT for a participant's audio track."""

    def __init__(
        self,
        participant,
        track,
        participant_locale: str,  # 🔥 참가자의 locale
        stt_model,  # 🔥 참가자별 STT 모델
        room,
        publish_callback,
        *,
        manner_checker: Optional[MannerChecker] = None,
        publish_manner_callback=None,
        translation_enabled: bool = False,
        supported_locales: Set[str] = None,  # 🔥 회의의 모든 locale
    ):
        self.participant = participant
        self.track = track
        self.participant_locale = participant_locale
        self.stt = stt_model
        self.room = room
        self.publish_callback = publish_callback

        # ✅ Manner
        self.mchecker = manner_checker
        self.publish_manner_callback = publish_manner_callback

        # ✅ Translation
        self.translation_enabled = translation_enabled
        self.supported_locales = supported_locales or set()
        
        clova_lang = locale_to_clova_lang(participant_locale) # (utils.py 사용)
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
        """(실시간) 현재 회의의 모든 참가자 locale (지원 언어만) 반환"""
        locales = set()
        
        # 로컬 참가자 (에이전트 자신 - 보통 metadata가 없음)
        if self.room.local_participant and self.room.local_participant.metadata:
            try:
                meta = json.loads(self.room.local_participant.metadata)
                if "appLocale" in meta:
                    locales.add(meta["appLocale"])
            except:
                pass
        
        # 원격 참가자들 (실제 사용자)
        for participant in self.room.remote_participants.values():
            if participant.metadata:
                try:
                    meta = json.loads(participant.metadata)
                    if "appLocale" in meta:
                        locales.add(meta["appLocale"])
                except:
                    pass
        
        # 지원하는 locale만 필터링
        return locales.intersection(self.supported_locales)

    # (신규) 참가자 메타데이터에서 meeting_id를 파싱하는 헬퍼 함수
    def _get_metadata_field(self, field_name: str, default: Any = None) -> Any:
        try:
            if self.participant.metadata:
                meta = json.loads(self.participant.metadata)
                return meta.get(field_name, default)
        except Exception:
            log.warning(f"[{self.participant.identity}] metadata 파싱 실패")
        return default

    async def _handle_transcription(self, event: agents.stt.SpeechEvent): # (타입 명시)
        pid = getattr(self.participant, "identity", "unknown")
        try:
            if not event.alternatives:
                log.debug(f"[{pid}] STT event에 alternatives가 없어 건너뜁니다.")
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

            # --- 이하는 'final' (최종) 결과일 때만 실행 ---

            # 1. Papago로 1차 단순 번역 수행 (기존 로직)
            simple_translations = {}
            if self.translation_enabled:
                current_locales = self._get_current_locales()
                target_locales = [loc for loc in current_locales if loc != self.participant_locale]
                if target_locales:
                    try:
                        # (integrations.py 사용)
                        simple_translations = await translate_to_multiple_languages(
                            text, 
                            self.participant_locale,
                            target_locales
                        )
                    except Exception as trans_err:
                        log.error(f"❌ Papago 번역 중 예외: {trans_err}", exc_info=True)

            # 2. (신규) hermes_service로 [문맥 보정 + DB 저장] 요청
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
                        "simple_translations": simple_translations
                    }
                    
                    # (integrations.py 사용)
                    response = await http_client.post(
                        f"{HERMES_SERVICE_URL}/utterances/process",
                        json=payload
                    )
                    response.raise_for_status()
                    
                    result_data = response.json()
                    final_translations = result_data.get("contextual_translations", simple_translations)
                    
                except httpx.HTTPStatusError as e:
                    log.error(f"❌ hermes_service API 오류: {e.response.status_code} - {e.response.text}")
                    final_translations = simple_translations
                except Exception as e:
                    log.error(f"❌ hermes_service 호출 실패: {e}")
                    final_translations = simple_translations

            # 3. (추가) Manner Check 로직
            if self.mchecker and self.publish_manner_callback:
                try:
                    clova_lang = locale_to_clova_lang(self.participant_locale) # (utils.py 사용)
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

            # 4. (수정) 최종 결과를 UI (DataChannel)로 전송
            await self.publish_callback(
                text=text,
                final=True,
                identity=self.participant.identity,
                psid=self.participant.sid,
                username=username,
                locale=self.participant_locale,
                translations=final_translations, # (보정된 번역본 사용)
            )
            
        except Exception as e:
            log.error(f"[{pid}] 🧩 _handle_transcription 처리 중 심각한 오류: {e}", exc_info=True)

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
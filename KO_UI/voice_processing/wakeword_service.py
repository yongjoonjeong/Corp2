from __future__ import annotations

import io
import os
import re
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable

from .openai_transcriber import TranscriptionError, transcribe_audio_bytes

EventCallback = Callable[[str, dict[str, Any]], None]


class WakeWordService:
    """Wake-word listener that accepts exactly one command per activation.

    The first activation uses the local full wake-word model. Every later
    activation uses the shorter "케이오" phrase, verified through STT.
    """

    def __init__(self, model_path: Path, on_event: EventCallback):
        self.model_path = Path(model_path)
        self.on_event = on_event
        self.display_name = os.environ.get("WAKEWORD_DISPLAY_NAME", "웨이크 업 케이오").strip() or "웨이크 업 케이오"
        self.followup_display_name = os.environ.get("FOLLOWUP_WAKEWORD_DISPLAY_NAME", "케이오").strip() or "케이오"
        self.threshold = float(os.environ.get("WAKEWORD_THRESHOLD", "0.65"))
        self.sample_rate = int(os.environ.get("WAKEWORD_SAMPLE_RATE", "48000"))
        self.frame_ms = int(os.environ.get("WAKEWORD_FRAME_MS", "80"))
        self.frame_size = max(960, int(self.sample_rate * self.frame_ms / 1000))
        self.device_index = self._optional_int(os.environ.get("WAKEWORD_DEVICE_INDEX"))
        self.command_max_sec = float(os.environ.get("WAKEWORD_COMMAND_MAX_SEC", "8.0"))
        self.command_min_sec = float(os.environ.get("WAKEWORD_COMMAND_MIN_SEC", "1.2"))
        self.command_silence_sec = float(os.environ.get("WAKEWORD_COMMAND_SILENCE_SEC", "1.45"))
        self.rms_min = float(os.environ.get("WAKEWORD_COMMAND_RMS_MIN", "420"))
        self.rms_multiplier = float(os.environ.get("WAKEWORD_COMMAND_RMS_MULTIPLIER", "2.6"))
        self.cooldown_sec = float(os.environ.get("WAKEWORD_COOLDOWN_SEC", "5.0"))
        self.confirm_hits = max(1, int(os.environ.get("WAKEWORD_CONFIRM_HITS", "2")))
        self.confirm_window = max(self.confirm_hits, int(os.environ.get("WAKEWORD_CONFIRM_WINDOW", "3")))
        self.download_models = os.environ.get("WAKEWORD_DOWNLOAD_MODELS", "1") != "0"

        self.session_timeout_sec = max(10.0, float(os.environ.get("VOICE_SESSION_TIMEOUT_SEC", "30")))
        self.session_max_sec = max(self.session_timeout_sec, float(os.environ.get("VOICE_SESSION_MAX_SEC", "3600")))
        self.session_followup_delay_sec = max(0.2, float(os.environ.get("VOICE_SESSION_FOLLOWUP_DELAY_SEC", "0.8")))
        self.session_voice_hits = max(1, int(os.environ.get("VOICE_SESSION_VOICE_HITS", "2")))
        self.followup_wake_max_sec = max(1.0, float(os.environ.get("FOLLOWUP_WAKEWORD_MAX_SEC", "2.5")))
        self.followup_wake_min_sec = max(0.2, float(os.environ.get("FOLLOWUP_WAKEWORD_MIN_SEC", "0.35")))
        self.followup_wake_silence_sec = max(0.3, float(os.environ.get("FOLLOWUP_WAKEWORD_SILENCE_SEC", "0.65")))

        self._enabled = os.environ.get("WAKEWORD_ENABLED", "1") != "0"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._suppressed_until = 0.0
        self._session_active = False
        self._session_deadline = 0.0
        self._session_source = ""
        self._initial_wake_completed = False
        self._status: dict[str, Any] = {
            "available": False,
            "running": False,
            "enabled": self._enabled,
            "state": "starting",
            "message": "음성 기능 준비 중",
            "confidence": 0.0,
            "threshold": self.threshold,
            "confirm_hits": self.confirm_hits,
            "confirm_window": self.confirm_window,
            "model": self.model_path.name,
            "device_index": self.device_index,
            "display_name": self.display_name,
            "initial_display_name": self.display_name,
            "followup_display_name": self.followup_display_name,
            "initial_wake_completed": False,
            "last_error": None,
            "session_active": False,
            "session_timeout_sec": self.session_timeout_sec,
            "session_remaining_sec": 0,
        }

    @staticmethod
    def _optional_int(value: str | None) -> int | None:
        if value is None or not value.strip():
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(text or "").lower())

    @classmethod
    def _is_exit_command(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        phrases = (
            "대화종료",
            "음성종료",
            "대기모드",
            "음성대기모드",
            "그만들을게",
            "이제그만들어",
            "호출어대기로",
        )
        return any(phrase in normalized for phrase in phrases)

    @classmethod
    def _is_followup_wake(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        return normalized in {
            "케이오",
            "케이오야",
            "케이요",
            "ko",
            "kayo",
            "kaio",
        }

    def _current_display_name(self) -> str:
        return self.followup_display_name if self._initial_wake_completed else self.display_name

    def _mark_initial_wake_completed(self) -> None:
        with self._lock:
            self._initial_wake_completed = True
            self._status.update(
                initial_wake_completed=True,
                display_name=self.followup_display_name,
            )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ko-wakeword", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if not enabled:
            self.end_session("muted")
        with self._lock:
            self._enabled = enabled
            waiting_name = self._current_display_name()
            self._status.update(
                enabled=self._enabled,
                state="waiting_wakeword" if self._enabled else "muted",
                message=f"‘{waiting_name}’ 호출어 대기 중" if self._enabled else "음성 대기 꺼짐",
            )
        self.on_event("status", self.status())

    def suppress(self, duration_ms: int) -> None:
        duration_sec = max(0.0, min(30.0, duration_ms / 1000.0))
        with self._lock:
            self._suppressed_until = max(self._suppressed_until, time.monotonic() + duration_sec)

    def start_session(self, duration_sec: float | None = None, *, source: str = "wakeword") -> dict[str, Any]:
        duration = self._clamp_session_duration(duration_sec)
        with self._lock:
            self._session_active = True
            self._session_deadline = time.monotonic() + duration
            self._session_source = source
            self._status.update(
                session_active=True,
                session_timeout_sec=duration,
                state="session_waiting",
                message="명령 대기 중 · 명령을 하나 말씀하세요",
            )
        payload = self.status()
        payload["source"] = source
        self._emit("session_started", **payload)
        return payload

    def extend_session(self, duration_sec: float | None = None, *, source: str = "ui") -> dict[str, Any]:
        duration = self._clamp_session_duration(duration_sec)
        with self._lock:
            active = bool(self._session_active)
        if not active:
            return self.status()
        with self._lock:
            self._session_deadline = time.monotonic() + duration
            self._session_source = source or self._session_source
            self._status.update(
                session_active=True,
                session_timeout_sec=duration,
                state="session_waiting",
                message="명령 대기 중 · 명령을 하나 말씀하세요",
            )
        payload = self.status()
        payload["source"] = source
        self._emit("session_extended", **payload)
        return payload

    def set_session_duration(self, duration_sec: float | None = None, *, source: str = "ui") -> dict[str, Any]:
        """Set the remaining session window, starting a session only if already active."""
        return self.extend_session(duration_sec, source=source)

    def end_session(self, reason: str = "manual") -> dict[str, Any]:
        should_emit = False
        with self._lock:
            if self._session_active:
                should_emit = True
            self._session_active = False
            self._session_deadline = 0.0
            self._session_source = ""
            waiting_name = self._current_display_name()
            self._status.update(
                session_active=False,
                session_remaining_sec=0,
                state="waiting_wakeword" if self._enabled else "muted",
                message=f"‘{waiting_name}’ 호출어 대기 중" if self._enabled else "음성 대기 꺼짐",
            )
        payload = self.status()
        payload["reason"] = reason
        if should_emit:
            self._emit("session_ended", **payload)
        return payload

    def _clamp_session_duration(self, duration_sec: float | None) -> float:
        try:
            value = float(duration_sec if duration_sec is not None else self.session_timeout_sec)
        except (TypeError, ValueError):
            value = self.session_timeout_sec
        return max(5.0, min(self.session_max_sec, value))

    def _session_snapshot(self) -> tuple[bool, float]:
        with self._lock:
            active = bool(self._session_active)
            deadline = float(self._session_deadline)
        return active, deadline

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._status)
            active = bool(self._session_active)
            deadline = float(self._session_deadline)
        remaining = max(0.0, deadline - time.monotonic()) if active else 0.0
        if active and remaining <= 0:
            active = False
        result["session_active"] = active
        result["session_remaining_sec"] = int(round(remaining)) if active else 0
        return result

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)

    def _emit(self, event_type: str, **payload: Any) -> None:
        self.on_event(event_type, payload)

    def _run(self) -> None:
        audio = None
        stream = None
        try:
            import numpy as np
            import openwakeword
            import pyaudio
            from openwakeword.model import Model
            from scipy.signal import resample_poly

            if not self.model_path.is_file():
                raise FileNotFoundError(f"웨이크업 모델이 없습니다: {self.model_path}")
            if self.download_models:
                openwakeword.utils.download_models()

            model = Model(wakeword_models=[str(self.model_path)])
            model_key = self.model_path.stem
            audio = pyaudio.PyAudio()
            open_kwargs: dict[str, Any] = {
                "format": pyaudio.paInt16,
                "channels": 1,
                "rate": self.sample_rate,
                "input": True,
                "frames_per_buffer": self.frame_size,
            }
            if self.device_index is not None:
                open_kwargs["input_device_index"] = self.device_index
            stream = audio.open(**open_kwargs)

            device_name = "기본 입력 장치"
            try:
                index = self.device_index
                if index is None:
                    index = int(audio.get_default_input_device_info()["index"])
                device_name = str(audio.get_device_info_by_index(index).get("name", device_name))
                self.device_index = index
            except Exception:
                pass

            self._set_status(
                available=True,
                running=True,
                state="waiting_wakeword" if self._enabled else "muted",
                message=f"‘{self.display_name}’ 호출어 대기 중" if self._enabled else "음성 대기 꺼짐",
                device_index=self.device_index,
                device_name=device_name,
                last_error=None,
            )
            self._emit("ready", **self.status())

            frame_sec = self.frame_size / self.sample_rate
            pre_roll = deque(maxlen=max(3, int(1.0 / frame_sec)))
            confidence_window = deque(maxlen=self.confirm_window)
            noise_rms = self.rms_min / self.rms_multiplier
            last_trigger = 0.0

            while not self._stop_event.is_set():
                session_active, session_deadline = self._session_snapshot()
                if session_active:
                    if time.monotonic() >= session_deadline:
                        self.end_session("timeout")
                        pre_roll.clear()
                        confidence_window.clear()
                        continue

                    command_wav, noise_rms = self._wait_for_session_command(stream, np, noise_rms)
                    if command_wav is None:
                        active, deadline = self._session_snapshot()
                        if active and time.monotonic() >= deadline:
                            self.end_session("timeout")
                        continue

                    text = self._transcribe_command(command_wav, confidence=None, session_active=True)
                    if text is None:
                        continue
                    if self._is_exit_command(text):
                        self.end_session("voice_command")
                        continue
                    self.end_session("command_completed")
                    with self._lock:
                        self._suppressed_until = max(
                            self._suppressed_until,
                            time.monotonic() + self.session_followup_delay_sec,
                        )
                    continue

                if self._initial_wake_completed:
                    with self._lock:
                        followup_enabled = self._enabled
                    if not followup_enabled:
                        # Keep consuming the stream without sending muted audio
                        # to the transcription service.
                        stream.read(self.frame_size, exception_on_overflow=False)
                        continue
                    wake_wav, noise_rms = self._wait_for_followup_wake(
                        stream, np, noise_rms
                    )
                    if wake_wav is None:
                        continue
                    transcript = self._transcribe_followup_wake(wake_wav)
                    if transcript is None:
                        continue

                    self.start_session(self.session_timeout_sec, source="followup_wakeword")
                    self._set_status(
                        state="wake_detected",
                        message="‘케이오’ 감지 · 명령을 듣는 중",
                        last_transcript=transcript,
                    )
                    self._emit(
                        "wake_detected",
                        confidence=None,
                        session_active=True,
                        wakeword=self.followup_display_name,
                        source="followup_wakeword",
                    )
                    with self._lock:
                        self._suppressed_until = max(
                            self._suppressed_until,
                            time.monotonic() + self.session_followup_delay_sec,
                        )
                    continue

                raw = stream.read(self.frame_size, exception_on_overflow=False)
                samples_48k = np.frombuffer(raw, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples_48k.astype(np.float32) ** 2))) if samples_48k.size else 0.0
                pre_roll.append(raw)

                with self._lock:
                    enabled = self._enabled
                    suppressed_until = self._suppressed_until
                if not enabled or time.monotonic() < suppressed_until:
                    continue

                noise_rms = noise_rms * 0.995 + min(rms, self.rms_min * 2) * 0.005
                samples_16k = resample_poly(samples_48k, 1, 3).astype(np.int16)
                predictions = model.predict(samples_16k)
                confidence = float(predictions.get(model_key, max(predictions.values(), default=0.0)))
                self._set_status(confidence=round(confidence, 4), state="waiting_wakeword", message=f"‘{self.display_name}’ 호출어 대기 중")

                confidence_window.append(confidence)
                confirmed = sum(value >= self.threshold for value in confidence_window) >= self.confirm_hits
                now = time.monotonic()
                if not confirmed or now - last_trigger < self.cooldown_sec:
                    continue

                confidence_window.clear()
                last_trigger = now
                self._mark_initial_wake_completed()
                self.start_session(self.session_timeout_sec, source="wakeword")
                self._set_status(state="wake_detected", message="호출어 감지 · 명령을 듣는 중")
                self._emit("wake_detected", confidence=confidence, session_active=True)
                command_wav = self._record_command(stream, np, list(pre_roll), noise_rms)
                if hasattr(model, "reset"):
                    model.reset()
                pre_roll.clear()
                confidence_window.clear()

                text = self._transcribe_command(command_wav, confidence=confidence, session_active=True)
                if text is None:
                    continue
                if self._is_exit_command(text):
                    self.end_session("voice_command")
                    continue
                self.end_session("command_completed")
                with self._lock:
                    self._suppressed_until = max(
                        self._suppressed_until,
                        time.monotonic() + self.session_followup_delay_sec,
                    )

        except Exception as exc:
            message = f"음성 서비스 시작 실패: {type(exc).__name__}: {exc}"
            self._set_status(available=False, running=False, state="error", message=message, last_error=message)
            self._emit("error", message=message)
            print(message)
        finally:
            self.end_session("service_stopped")
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if audio is not None:
                try:
                    audio.terminate()
                except Exception:
                    pass
            self._set_status(running=False)

    def _transcribe_followup_wake(self, wake_wav: bytes) -> str | None:
        self._set_status(state="transcribing", message="‘케이오’ 호출을 확인하는 중")
        self._emit("transcribing", session_active=False, wakeword_check=True)
        try:
            result = transcribe_audio_bytes(
                wake_wav,
                "ko-followup-wakeword.wav",
                "audio/wav",
            )
        except TranscriptionError as exc:
            self._set_status(
                state="waiting_wakeword",
                message=f"‘{self.followup_display_name}’ 호출어 대기 중",
                last_error=str(exc),
            )
            self._emit("status", **self.status())
            return None

        text = str(result.get("text", "")).strip()
        if not self._is_followup_wake(text):
            self._set_status(
                state="waiting_wakeword",
                message=f"‘{self.followup_display_name}’ 호출어 대기 중",
                last_transcript=text,
                last_error=None,
            )
            self._emit("status", **self.status())
            return None
        return text

    def _transcribe_command(self, command_wav: bytes, *, confidence: float | None, session_active: bool) -> str | None:
        self._set_status(state="transcribing", message="음성 명령을 확인하는 중")
        self._emit("transcribing", session_active=session_active)
        try:
            result = transcribe_audio_bytes(command_wav, "ko-voice-command.wav", "audio/wav")
        except TranscriptionError as exc:
            message = str(exc)
            self._set_status(
                state="session_waiting" if session_active else "waiting_wakeword",
                message="명령을 다시 말씀해 주세요" if session_active else f"‘{self.display_name}’ 호출어 대기 중",
                last_error=message,
            )
            self._emit("command_error", message=message, session_active=session_active)
            return None

        text = str(result.get("text", "")).strip()
        if not text:
            self._set_status(
                state="session_waiting" if session_active else "waiting_wakeword",
                message="명령을 다시 말씀해 주세요" if session_active else f"‘{self.display_name}’ 호출어 대기 중",
            )
            return None

        self._set_status(
            state="session_waiting" if session_active else "waiting_wakeword",
            message="명령 처리 중 · 완료 후 호출 대기로 돌아갑니다" if session_active else f"‘{self.display_name}’ 호출어 대기 중",
            last_transcript=text,
            last_error=None,
        )
        payload: dict[str, Any] = {
            "text": text,
            "model": result.get("model"),
            "session_active": session_active,
        }
        if confidence is not None:
            payload["confidence"] = confidence
        self._emit("transcript", **payload)
        return text

    def _wait_for_session_command(self, stream: Any, np_module: Any, noise_rms: float) -> tuple[bytes | None, float]:
        frame_sec = self.frame_size / self.sample_rate
        pre_roll = deque(maxlen=max(3, int(0.8 / frame_sec)))
        voice_hits = 0

        self._set_status(state="session_waiting", message="명령 대기 중 · 명령을 하나 말씀하세요")

        while not self._stop_event.is_set():
            active, deadline = self._session_snapshot()
            if not active or time.monotonic() >= deadline:
                return None, noise_rms

            raw = stream.read(self.frame_size, exception_on_overflow=False)
            samples = np_module.frombuffer(raw, dtype=np_module.int16)
            rms = float(np_module.sqrt(np_module.mean(samples.astype(np_module.float32) ** 2))) if samples.size else 0.0

            with self._lock:
                enabled = self._enabled
                suppressed_until = self._suppressed_until
            if not enabled:
                return None, noise_rms
            if time.monotonic() < suppressed_until:
                pre_roll.clear()
                voice_hits = 0
                continue

            noise_rms = noise_rms * 0.995 + min(rms, self.rms_min * 2) * 0.005
            threshold = max(self.rms_min, noise_rms * self.rms_multiplier)
            pre_roll.append(raw)
            voice_hits = voice_hits + 1 if rms >= threshold else 0
            if voice_hits >= self.session_voice_hits:
                return self._record_command(stream, np_module, list(pre_roll), noise_rms), noise_rms

        return None, noise_rms

    def _wait_for_followup_wake(
        self,
        stream: Any,
        np_module: Any,
        noise_rms: float,
    ) -> tuple[bytes | None, float]:
        frame_sec = self.frame_size / self.sample_rate
        pre_roll = deque(maxlen=max(2, int(0.35 / frame_sec)))
        voice_hits = 0
        self._set_status(
            state="waiting_wakeword",
            message=f"‘{self.followup_display_name}’ 호출어 대기 중",
        )

        while not self._stop_event.is_set():
            raw = stream.read(self.frame_size, exception_on_overflow=False)
            samples = np_module.frombuffer(raw, dtype=np_module.int16)
            rms = float(
                np_module.sqrt(np_module.mean(samples.astype(np_module.float32) ** 2))
            ) if samples.size else 0.0

            with self._lock:
                enabled = self._enabled
                suppressed_until = self._suppressed_until
            if not enabled:
                return None, noise_rms
            if time.monotonic() < suppressed_until:
                pre_roll.clear()
                voice_hits = 0
                continue

            noise_rms = noise_rms * 0.995 + min(rms, self.rms_min * 2) * 0.005
            threshold = max(self.rms_min, noise_rms * self.rms_multiplier)
            pre_roll.append(raw)
            voice_hits = voice_hits + 1 if rms >= threshold else 0
            if voice_hits >= self.session_voice_hits:
                return (
                    self._record_followup_wake(
                        stream,
                        np_module,
                        list(pre_roll),
                        noise_rms,
                    ),
                    noise_rms,
                )
        return None, noise_rms

    def _record_followup_wake(
        self,
        stream: Any,
        np_module: Any,
        initial_frames: list[bytes],
        noise_rms: float,
    ) -> bytes:
        frames = list(initial_frames)
        started_at = time.monotonic()
        last_voice_at = started_at
        threshold = max(self.rms_min, noise_rms * self.rms_multiplier)

        while not self._stop_event.is_set():
            raw = stream.read(self.frame_size, exception_on_overflow=False)
            frames.append(raw)
            samples = np_module.frombuffer(raw, dtype=np_module.int16)
            rms = float(
                np_module.sqrt(np_module.mean(samples.astype(np_module.float32) ** 2))
            ) if samples.size else 0.0
            now = time.monotonic()
            elapsed = now - started_at
            if rms >= threshold:
                last_voice_at = now
            if elapsed >= self.followup_wake_max_sec:
                break
            if (
                elapsed >= self.followup_wake_min_sec
                and now - last_voice_at >= self.followup_wake_silence_sec
            ):
                break

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(frames))
        return wav_io.getvalue()

    def _record_command(self, stream: Any, np_module: Any, initial_frames: list[bytes], noise_rms: float) -> bytes:
        self._emit("listening", session_active=True)
        self._set_status(state="command_listening", message="명령을 듣는 중")
        frames = list(initial_frames)
        started_at = time.monotonic()
        last_voice_at = started_at
        threshold = max(self.rms_min, noise_rms * self.rms_multiplier)

        while not self._stop_event.is_set():
            raw = stream.read(self.frame_size, exception_on_overflow=False)
            frames.append(raw)
            samples = np_module.frombuffer(raw, dtype=np_module.int16)
            rms = float(np_module.sqrt(np_module.mean(samples.astype(np_module.float32) ** 2))) if samples.size else 0.0
            now = time.monotonic()
            elapsed = now - started_at
            if rms >= threshold:
                last_voice_at = now
            if elapsed >= self.command_max_sec:
                break
            if elapsed >= self.command_min_sec and now - last_voice_at >= self.command_silence_sec:
                break

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(frames))
        return wav_io.getvalue()

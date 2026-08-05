from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TranscriptionError(RuntimeError):
    pass


def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "ko-command.wav", media_type: str = "audio/wav") -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise TranscriptionError("OPENAI_API_KEY가 설정되지 않았습니다. configure_api_key.py를 먼저 실행해 주세요.")
    if len(audio_bytes) < 500:
        raise TranscriptionError("녹음이 너무 짧습니다. 다시 말해 주세요.")

    model = os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "whisper-1").strip() or "whisper-1"
    prompt = os.environ.get(
        "OPENAI_TRANSCRIPTION_PROMPT",
        "웨이크 업 케이오, 웨이크업 케이오, Wake up KO, 헤이 케이오, "
        "사용자 등록, 운동 시작, 훈련 시작, 운동 종료, 오른손, 왼손, "
        "잽, 스트레이트, 일시정지, 다시 시작, 현재 기록",
    ).strip()
    boundary = f"----KoBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in {"model": model, "language": "ko", "prompt": prompt}.items():
        chunks += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ]

    safe_filename = Path(filename).name.replace('"', "") or "ko-command.wav"
    chunks += [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'.encode(),
        f"Content-Type: {media_type}\r\n\r\n".encode(),
        audio_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = Request(
        f"{base_url}/audio/transcriptions",
        data=b"".join(chunks),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": "KO-Boxing-UI/1.0",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise TranscriptionError(
            f"OpenAI 음성 인식 요청이 거부되었습니다. HTTP {exc.code}: API 키와 결제 상태를 확인해 주세요."
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise TranscriptionError("OpenAI 서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.") from exc
    except Exception as exc:
        raise TranscriptionError("OpenAI 음성 인식 처리 중 오류가 발생했습니다.") from exc

    text = str(payload.get("text", "")).strip()
    if not text:
        raise TranscriptionError("음성을 인식하지 못했습니다. 다시 말해 주세요.")
    return {"text": text, "model": model}

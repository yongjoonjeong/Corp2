from __future__ import annotations

import getpass
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def main() -> None:
    print("KO OpenAI API 키 설정")
    print("입력한 키는 화면에 표시되지 않으며 이 PC의 .env 파일에만 저장됩니다.")
    key = getpass.getpass("새 OpenAI API 키: ").strip()
    if not key:
        raise SystemExit("키가 입력되지 않았습니다.")
    if len(key) < 20:
        raise SystemExit("API 키 형식이 너무 짧습니다.")

    preserved = []
    if ENV_FILE.is_file():
        preserved = [
            line for line in ENV_FILE.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("OPENAI_API_KEY=")
        ]
    content = "\n".join([f"OPENAI_API_KEY={key}", *preserved]).rstrip() + "\n"
    ENV_FILE.write_text(content, encoding="utf-8")
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        pass
    print(f"설정 완료: {ENV_FILE}")
    print("이 파일은 .gitignore에 포함되어 있습니다.")


if __name__ == "__main__":
    main()

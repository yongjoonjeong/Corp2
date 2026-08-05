#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

sudo apt-get update
sudo apt-get install -y \
  curl libglib2.0-0 libgl1 libportaudio2 portaudio19-dev \
  python3-dev python3-venv v4l-utils

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

echo
echo "설치 완료"
echo "터미널 1: roboton"
echo "터미널 2: ./run.sh"

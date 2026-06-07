#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/Users/ader/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
"$PYTHON_BIN" -m venv .venv-f5
source .venv-f5/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio
python -m pip install f5-tts

cat <<'MSG'
F5-TTS installed in .venv-f5.

Next:
  1. Put authorized reference wav files somewhere in this repo.
  2. Fill reference_audio/reference_text in tools/tts_eval_config.json.
  3. Run:
       source .venv-f5/bin/activate
       python3 tools/tts_eval.py --provider f5-cli
MSG

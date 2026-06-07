#!/usr/bin/env python3
"""Local TTS bridge for the iOS simulator.

This intentionally exposes a tiny stable HTTP surface:

POST /tts
{
  "text": "...",
  "presetId": "bow-detective",
  "presetName": "紅領結偵探",
  "modelHint": "..."
}

The primary provider is Sherpa-ONNX ZipVoice using the selected
`taiwan_mandarin_low_r` teacher reference. macOS `say` remains available as a
smoke-test fallback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parents[1]

ZIPVOICE_MODEL_DIR = (
    ROOT
    / "external"
    / "ZipVoice"
    / "egs"
    / "zipvoice"
    / "exp"
    / "zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10"
)
ZIPVOICE_VOCODER = ROOT / "models" / "sherpa" / "vocos_24khz_dynamic_qint8.onnx"
ZIPVOICE_DATA_DIR = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min"
ZIPVOICE_STEPS = 3
ZIPVOICE_SPEED = 1.0
TAIWAN_LOW_R_REFERENCE_AUDIO = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "teacher_qwen3_1p7b_distill_v1"
    / "audio"
    / "distill_0158.wav"
)
TAIWAN_LOW_R_REFERENCE_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"

VOICE_BY_PRESET = {
    "taiwan-mandarin-low-r": ("Meijia", 176),
    "bow-detective": ("Tingting", 205),
    "taiwan-variety": ("Meijia", 198),
    "news-anchor": ("Meijia", 172),
    "warm-narrator": ("Meijia", 160),
    "dramatic-reveal": ("Tingting", 142),
}

PROVIDER = "zipvoice"
_ZIPVOICE_TTS = None


def synthesize_with_say(text: str, preset_id: str) -> bytes:
    voice, rate = VOICE_BY_PRESET.get(preset_id, ("Meijia", 180))
    with tempfile.TemporaryDirectory(prefix="red-bow-tts-") as tmp:
        output = Path(tmp) / "speech.aiff"
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", str(output), text],
            check=True,
        )
        return output.read_bytes()


def _load_zipvoice_tts():
    global _ZIPVOICE_TTS
    if _ZIPVOICE_TTS is not None:
        return _ZIPVOICE_TTS

    import sherpa_onnx

    encoder = ZIPVOICE_MODEL_DIR / "text_encoder_int8.onnx"
    decoder = ZIPVOICE_MODEL_DIR / "fm_decoder_int8.onnx"
    if not encoder.exists():
        encoder = ZIPVOICE_MODEL_DIR / "encoder.int8.onnx"
    if not decoder.exists():
        decoder = ZIPVOICE_MODEL_DIR / "decoder.int8.onnx"
    tokens = ZIPVOICE_MODEL_DIR / "tokens.txt"
    if not tokens.exists():
        tokens = ZIPVOICE_DATA_DIR / "tokens.txt"

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                data_dir=str(ZIPVOICE_DATA_DIR / "espeak-ng-data"),
                lexicon=str(ZIPVOICE_DATA_DIR / "lexicon.txt"),
                vocoder=str(ZIPVOICE_VOCODER),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("Invalid ZipVoice config. Check models/sherpa assets.")
    _ZIPVOICE_TTS = sherpa_onnx.OfflineTts(config)
    return _ZIPVOICE_TTS


def synthesize_with_zipvoice(text: str) -> bytes:
    import librosa
    import sherpa_onnx
    import soundfile as sf

    if not TAIWAN_LOW_R_REFERENCE_AUDIO.exists():
        raise RuntimeError(f"Missing reference audio: {TAIWAN_LOW_R_REFERENCE_AUDIO}")

    tts = _load_zipvoice_tts()
    reference_audio, sample_rate = librosa.load(str(TAIWAN_LOW_R_REFERENCE_AUDIO), sr=None)
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.reference_text = TAIWAN_LOW_R_REFERENCE_TEXT
    gen_config.num_steps = ZIPVOICE_STEPS
    gen_config.speed = ZIPVOICE_SPEED
    gen_config.extra["min_char_in_sentence"] = "20"

    audio = tts.generate(text, gen_config)
    if len(audio.samples) == 0:
        raise RuntimeError("ZipVoice returned empty audio")

    with tempfile.TemporaryDirectory(prefix="red-bow-zipvoice-") as tmp:
        output = Path(tmp) / "speech.wav"
        sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
        return output.read_bytes()


def synthesize(text: str, preset_id: str) -> tuple[bytes, str]:
    if PROVIDER == "macos-say":
        return synthesize_with_say(text, preset_id), "audio/aiff"
    if PROVIDER == "zipvoice":
        return synthesize_with_zipvoice(text), "audio/wav"
    raise RuntimeError(f"unknown provider: {PROVIDER}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                {
                    "ok": True,
                    "provider": PROVIDER,
                    "voice": "taiwan_mandarin_low_r",
                    "reference_audio": str(TAIWAN_LOW_R_REFERENCE_AUDIO),
                    "model_dir": str(ZIPVOICE_MODEL_DIR) if PROVIDER == "zipvoice" else None,
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/tts":
            self.send_error(404)
            return

        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body)
            text = str(payload.get("text", "")).strip()
            preset_id = str(payload.get("presetId", ""))
            if not text:
                self._send_json({"error": "missing text"}, status=400)
                return

            audio, content_type = synthesize(text, preset_id)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        except subprocess.CalledProcessError as error:
            self._send_json({"error": f"TTS command failed: {error}"}, status=500)
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    global PROVIDER, HOST, PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["zipvoice", "macos-say"], default=PROVIDER)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--model-dir", type=Path, default=ZIPVOICE_MODEL_DIR)
    parser.add_argument("--vocoder", type=Path, default=ZIPVOICE_VOCODER)
    parser.add_argument("--steps", type=int, default=ZIPVOICE_STEPS)
    parser.add_argument("--speed", type=float, default=ZIPVOICE_SPEED)
    args = parser.parse_args()
    PROVIDER = args.provider
    HOST = args.host
    PORT = args.port
    globals()["ZIPVOICE_MODEL_DIR"] = args.model_dir
    globals()["ZIPVOICE_VOCODER"] = args.vocoder
    globals()["ZIPVOICE_STEPS"] = args.steps
    globals()["ZIPVOICE_SPEED"] = args.speed

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Red Bow local TTS server: http://{HOST}:{PORT}")
    print(f"Provider: {PROVIDER}")
    if PROVIDER == "zipvoice":
        print("Voice: taiwan_mandarin_low_r")
        print(f"Model: {ZIPVOICE_MODEL_DIR}")
        print(f"Vocoder: {ZIPVOICE_VOCODER}")
        print(f"Steps: {ZIPVOICE_STEPS}; speed: {ZIPVOICE_SPEED}")
        print(f"Reference: {TAIWAN_LOW_R_REFERENCE_AUDIO}")
    server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phone demo server for ZipVoice Taiwan gentle-female TTS."""

from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports"
MODEL_DIR = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min"
VOCODER = ROOT / "models" / "sherpa" / "vocos_24khz.onnx"
REFERENCE_AUDIO = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "reports"
    / "zipvoice_three_way"
    / "audio"
    / "teacher_qwen3_1p7b"
    / "seed_0001.wav"
)
REFERENCE_TEXT = "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"

_TTS = None
_REFERENCE_AUDIO_DATA: np.ndarray | None = None
_REFERENCE_SAMPLE_RATE: int | None = None


def load_reference() -> tuple[np.ndarray, int]:
    global _REFERENCE_AUDIO_DATA, _REFERENCE_SAMPLE_RATE
    if _REFERENCE_AUDIO_DATA is None or _REFERENCE_SAMPLE_RATE is None:
        audio, sample_rate = sf.read(str(REFERENCE_AUDIO), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        _REFERENCE_AUDIO_DATA = audio
        _REFERENCE_SAMPLE_RATE = sample_rate
    return _REFERENCE_AUDIO_DATA, _REFERENCE_SAMPLE_RATE


def load_tts():
    global _TTS
    if _TTS is not None:
        return _TTS

    import sherpa_onnx

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(MODEL_DIR / "tokens.txt"),
                encoder=str(MODEL_DIR / "encoder.int8.onnx"),
                decoder=str(MODEL_DIR / "decoder.int8.onnx"),
                data_dir=str(MODEL_DIR / "espeak-ng-data"),
                lexicon=str(MODEL_DIR / "lexicon.txt"),
                vocoder=str(VOCODER),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("Invalid ZipVoice config")
    _TTS = sherpa_onnx.OfflineTts(config)
    return _TTS


def synthesize(text: str) -> tuple[bytes, dict[str, float | int]]:
    import sherpa_onnx

    tts = load_tts()
    reference_audio, reference_sample_rate = load_reference()

    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = reference_sample_rate
    gen_config.reference_text = REFERENCE_TEXT
    gen_config.num_steps = 4
    gen_config.extra["min_char_in_sentence"] = "20"

    start = perf_counter()
    audio = tts.generate(text, gen_config)
    elapsed = perf_counter() - start
    if len(audio.samples) == 0:
        raise RuntimeError("ZipVoice returned empty audio")

    with tempfile.TemporaryDirectory(prefix="zipvoice-demo-") as tmp:
        output = Path(tmp) / "speech.wav"
        sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
        data = output.read_bytes()

    duration = len(audio.samples) / audio.sample_rate
    return data, {
        "generation_s": elapsed,
        "audio_duration_s": duration,
        "rtf": elapsed / duration if duration else 0.0,
        "sample_rate": audio.sample_rate,
        "bytes": len(data),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "model_dir": str(MODEL_DIR),
                    "vocoder": str(VOCODER),
                    "reference_audio": str(REFERENCE_AUDIO),
                    "reference_text": REFERENCE_TEXT,
                }
            )
            return

        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            path = "/zipvoice_demo/index.html"
        target = (REPORT_ROOT / path.lstrip("/")).resolve()
        if not str(target).startswith(str(REPORT_ROOT.resolve())):
            self.send_error(403)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/api/tts":
            self.send_error(404)
            return

        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body)
            text = str(payload.get("text", "")).strip()
            if not text:
                self.send_json({"error": "missing text"}, status=400)
                return
            if len(text) > 180:
                self.send_json({"error": "text too long; keep it under 180 characters"}, status=400)
                return

            audio, metrics = synthesize(text)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("X-Generation-Seconds", f"{metrics['generation_s']:.3f}")
            self.send_header("X-Audio-Duration-Seconds", f"{metrics['audio_duration_s']:.3f}")
            self.send_header("X-RTF", f"{metrics['rtf']:.3f}")
            self.end_headers()
            self.wfile.write(audio)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    if args.preload:
        load_reference()
        load_tts()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ZipVoice demo server: http://{args.host}:{args.port}")
    print(f"Demo: /zipvoice_demo/index.html")
    print(f"Report: /zipvoice_main_report/index.html")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

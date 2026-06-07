#!/usr/bin/env python3
"""Live demo server for the latest ZipVoice-Distill ONNX int8 voice."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
ZIPVOICE_ROOT = ROOT / "external" / "ZipVoice"
ZIPVOICE_EGS = ZIPVOICE_ROOT / "egs" / "zipvoice"
DEMO_ROOT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "zipvoice_live_demo"
MODEL_DIR = ZIPVOICE_EGS / "exp" / "zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10"
PROMPT_WAV = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_distill_v1" / "audio" / "distill_0158.wav"
PROMPT_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"

sys.path.insert(0, str(ZIPVOICE_ROOT))


class ZipVoiceEngine:
    def __init__(self, num_threads: int = 4) -> None:
        import torch
        from zipvoice.bin.infer_zipvoice import get_vocoder
        from zipvoice.bin.infer_zipvoice_onnx import OnnxModel, generate_sentence
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        from zipvoice.utils.feature import VocosFbank

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

        self.generate_sentence = generate_sentence
        self.model = OnnxModel(
            str(MODEL_DIR / "text_encoder_int8.onnx"),
            str(MODEL_DIR / "fm_decoder_int8.onnx"),
            num_thread=num_threads,
        )
        self.vocoder = get_vocoder(None)
        self.vocoder.eval()
        self.tokenizer = EmiliaTokenizer(token_file=str(MODEL_DIR / "tokens.txt"))
        self.feature_extractor = VocosFbank()
        self.lock = threading.Lock()

    def synthesize(self, text: str, num_step: int = 3) -> tuple[bytes, dict[str, float | int]]:
        if num_step not in {2, 3, 4}:
            raise ValueError("num_step must be 2, 3, or 4")
        with tempfile.TemporaryDirectory(prefix="zipvoice-live-demo-") as tmp:
            output = Path(tmp) / "speech.wav"
            with self.lock:
                started = perf_counter()
                metrics = self.generate_sentence(
                    save_path=str(output),
                    prompt_text=PROMPT_TEXT,
                    prompt_wav=str(PROMPT_WAV),
                    text=text,
                    model=self.model,
                    vocoder=self.vocoder,
                    tokenizer=self.tokenizer,
                    feature_extractor=self.feature_extractor,
                    num_step=num_step,
                    guidance_scale=3.0,
                    speed=1.0,
                    t_shift=0.5,
                    target_rms=0.1,
                    feat_scale=0.1,
                    sampling_rate=24000,
                    remove_long_sil=False,
                )
                wall_seconds = perf_counter() - started
            data = output.read_bytes()
        return data, {
            "wall_seconds": wall_seconds,
            "model_seconds": float(metrics["t"]),
            "audio_seconds": float(metrics["wav_seconds"]),
            "rtf": float(metrics["rtf"]),
            "sample_rate": 24000,
            "bytes": len(data),
            "num_step": num_step,
        }


ENGINE: ZipVoiceEngine | None = None


def get_engine(num_threads: int = 4) -> ZipVoiceEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = ZipVoiceEngine(num_threads=num_threads)
    return ENGINE


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "model": "ZipVoice-Distill few-step ONNX int8",
                    "num_steps": [2, 3, 4],
                    "default_step": 3,
                    "model_dir": str(MODEL_DIR),
                    "prompt_wav": str(PROMPT_WAV),
                    "qwen_teacher": "Static comparison samples are included in the page.",
                }
            )
            return

        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            path = "/index.html"
        target = (DEMO_ROOT / path.lstrip("/")).resolve()
        if not str(target).startswith(str(DEMO_ROOT.resolve())):
            self.send_error(403)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            self.send_error(404)
            return

        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
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
            payload = json.loads(body or b"{}")
            text = str(payload.get("text", "")).strip()
            if not text:
                self.send_json({"error": "missing text"}, status=400)
                return
            if len(text) > 140:
                self.send_json({"error": "請先控制在 140 字內，方便測單句速度。"}, status=400)
                return
            num_step = int(payload.get("num_step", 3))
            audio, metrics = get_engine().synthesize(text, num_step=num_step)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("X-Wall-Seconds", f"{metrics['wall_seconds']:.3f}")
            self.send_header("X-Model-Seconds", f"{metrics['model_seconds']:.3f}")
            self.send_header("X-Audio-Seconds", f"{metrics['audio_seconds']:.3f}")
            self.send_header("X-RTF", f"{metrics['rtf']:.3f}")
            self.send_header("X-Num-Step", str(metrics["num_step"]))
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if args.preload:
        get_engine()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ZipVoice-Distill live demo: http://{args.host}:{args.port}/")
    print("Latest voice: stage2 few-step ONNX int8 epoch-10, selectable 2/3/4-step")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

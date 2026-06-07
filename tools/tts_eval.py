#!/usr/bin/env python3
"""Generate a TTS audition run and an HTML scorecard."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "tools" / "tts_eval_config.json"
RUNS_DIR = ROOT / "tts_runs"

MACOS_VOICE_BY_TARGET = {
    "bow-detective": ("Tingting", 205),
    "taiwan-variety": ("Meijia", 198),
    "news-anchor": ("Meijia", 172),
    "warm-narrator": ("Meijia", 160),
    "dramatic-reveal": ("Tingting", 142),
}


@dataclass
class Result:
    target_id: str
    target_name: str
    goal: str
    reference_audio: Path | None
    test_index: int
    text: str
    output: Path | None
    seconds: float
    status: str
    error: str = ""


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def synthesize_macos_say(text: str, target_id: str, output: Path) -> None:
    voice, rate = MACOS_VOICE_BY_TARGET.get(target_id, ("Meijia", 180))
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(output), text], check=True)


def synthesize_f5_cli(text: str, target: dict, output: Path) -> None:
    ref_audio = target.get("reference_audio", "").strip()
    ref_text = target.get("reference_text", "").strip()
    if not ref_audio or not ref_text:
        raise RuntimeError("missing reference_audio/reference_text in tools/tts_eval_config.json")

    ref_path = Path(ref_audio)
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path
    if not ref_path.exists():
        raise RuntimeError(f"reference audio does not exist: {ref_path}")

    if shutil.which("f5-tts_infer-cli") is None:
        raise RuntimeError("f5-tts_infer-cli not found. Install with: bash tools/setup_f5_tts.sh")

    work_dir = output.parent / f"_f5_{output.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "f5-tts_infer-cli",
            "--model",
            "F5TTS_v1_Base",
            "--ref_audio",
            str(ref_path),
            "--ref_text",
            ref_text,
            "--gen_text",
            text,
            "--output_dir",
            str(work_dir),
            "--output_file",
            output.name,
            "--remove_silence",
        ],
        cwd=work_dir,
        check=True,
    )
    generated = work_dir / output.name
    if not generated.exists():
        raise RuntimeError(f"F5-TTS finished but did not create {generated}")
    shutil.copy2(generated, output)


def synthesize_sherpa_zipvoice(text: str, target: dict, output: Path) -> None:
    ref_audio = target.get("reference_audio", "").strip()
    ref_text = target.get("reference_text", "").strip()
    if not ref_audio or not ref_text:
        raise RuntimeError("missing reference_audio/reference_text in config")

    ref_path = Path(ref_audio)
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path
    if not ref_path.exists():
        raise RuntimeError(f"reference audio does not exist: {ref_path}")

    import librosa
    import sherpa_onnx
    import soundfile as sf

    model_dir = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
    vocoder = ROOT / "models" / "sherpa" / "vocos_24khz.onnx"
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(model_dir / "tokens.txt"),
                encoder=str(model_dir / "encoder.int8.onnx"),
                decoder=str(model_dir / "decoder.int8.onnx"),
                data_dir=str(model_dir / "espeak-ng-data"),
                lexicon=str(model_dir / "lexicon.txt"),
                vocoder=str(vocoder),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("invalid sherpa ZipVoice config; check models/sherpa")

    tts = sherpa_onnx.OfflineTts(config)
    reference_audio, sample_rate = librosa.load(str(ref_path), sr=None)
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.reference_text = ref_text
    gen_config.num_steps = int(target.get("zipvoice_num_steps", 4))
    gen_config.extra["min_char_in_sentence"] = str(target.get("zipvoice_min_char_in_sentence", 20))

    audio = tts.generate(text, gen_config)
    if len(audio.samples) == 0:
        raise RuntimeError("ZipVoice returned empty audio")
    sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")


def synthesize_sherpa_pocket(text: str, target: dict, output: Path) -> None:
    ref_audio = target.get("reference_audio", "").strip()
    if not ref_audio:
        raise RuntimeError("missing reference_audio in config")

    ref_path = Path(ref_audio)
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path
    if not ref_path.exists():
        raise RuntimeError(f"reference audio does not exist: {ref_path}")

    import librosa
    import sherpa_onnx
    import soundfile as sf

    model_dir = ROOT / "models" / "sherpa" / "sherpa-onnx-pocket-tts-int8-2026-01-26"
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                lm_flow=str(model_dir / "lm_flow.int8.onnx"),
                lm_main=str(model_dir / "lm_main.int8.onnx"),
                encoder=str(model_dir / "encoder.onnx"),
                decoder=str(model_dir / "decoder.int8.onnx"),
                text_conditioner=str(model_dir / "text_conditioner.onnx"),
                vocab_json=str(model_dir / "vocab.json"),
                token_scores_json=str(model_dir / "token_scores.json"),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("invalid sherpa PocketTTS config; check models/sherpa")

    tts = sherpa_onnx.OfflineTts(config)
    reference_audio, sample_rate = librosa.load(str(ref_path), sr=tts.sample_rate)
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.num_steps = int(target.get("pocket_num_steps", 5))

    audio = tts.generate(text, gen_config)
    if len(audio.samples) == 0:
        raise RuntimeError("PocketTTS returned empty audio")
    sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")


def synthesize_qwen3_mlx(text: str, target: dict, output: Path) -> None:
    ref_audio = target.get("reference_audio", "").strip()
    ref_text = target.get("reference_text", "").strip()
    if not ref_audio or not ref_text:
        raise RuntimeError("missing reference_audio/reference_text in config")

    ref_path = Path(ref_audio)
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path
    if not ref_path.exists():
        raise RuntimeError(f"reference audio does not exist: {ref_path}")

    prefix = output.with_suffix("")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_audio.tts.generate",
            "--model",
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
            "--text",
            text,
            "--ref_audio",
            str(ref_path),
            "--ref_text",
            ref_text,
            "--file_prefix",
            str(prefix),
            "--audio_format",
            "wav",
        ],
        check=True,
    )
    generated = prefix.with_name(f"{prefix.name}_000.wav")
    if not generated.exists():
        raise RuntimeError(f"Qwen3 MLX finished but did not create {generated}")
    shutil.copy2(generated, output)


def run(provider: str, limit: int | None = None, config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_config(config_path)
    run_name = config_path.stem.replace("tts_eval_", "")
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S") / f"{run_name}-{provider}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    generated_count = 0
    for target in config["targets"]:
        target_dir = run_dir / target["id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, text in enumerate(target["tests"], start=1):
            if limit is not None and generated_count >= limit:
                break
            suffix = ".aiff" if provider == "macos-say" else ".wav"
            output = target_dir / f"{index:02d}{suffix}"
            started = perf_counter()
            try:
                if provider == "macos-say":
                    synthesize_macos_say(text, target["id"], output)
                elif provider == "f5-cli":
                    synthesize_f5_cli(text, target, output)
                elif provider == "sherpa-zipvoice":
                    synthesize_sherpa_zipvoice(text, target, output)
                elif provider == "sherpa-pocket":
                    synthesize_sherpa_pocket(text, target, output)
                elif provider == "qwen3-mlx":
                    synthesize_qwen3_mlx(text, target, output)
                else:
                    raise RuntimeError(f"unknown provider: {provider}")
                status = "ok"
                error = ""
            except Exception as exc:
                status = "failed"
                error = str(exc)
                output = None
            results.append(
                Result(
                    target_id=target["id"],
                    target_name=target["name"],
                    goal=target["goal"],
                    reference_audio=(ROOT / target["reference_audio"]) if target.get("reference_audio") else None,
                    test_index=index,
                    text=text,
                    output=output,
                    seconds=perf_counter() - started,
                    status=status,
                    error=error,
                )
            )
            generated_count += 1
        if limit is not None and generated_count >= limit:
            break

    write_report(run_dir, provider, results)
    return run_dir / "index.html"


def write_report(run_dir: Path, provider: str, results: list[Result]) -> None:
    rows = []
    for result in results:
        reference = ""
        if result.reference_audio and result.reference_audio.exists():
            rel_ref = os.path.relpath(result.reference_audio, run_dir)
            reference = f'<audio controls src="{html.escape(rel_ref)}"></audio>'
        audio = ""
        if result.output:
            rel = result.output.relative_to(run_dir).as_posix()
            audio = f'<audio controls src="{html.escape(rel)}"></audio>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.target_name)}</td>"
            f"<td>{html.escape(result.goal)}</td>"
            f"<td>{reference}</td>"
            f"<td>{html.escape(result.text)}</td>"
            f"<td>{audio}</td>"
            f"<td>{result.seconds:.1f}s</td>"
            f"<td>{html.escape(result.status)}</td>"
            f"<td>{html.escape(result.error)}</td>"
            '<td contenteditable="true"></td>'
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>紅色蝴蝶結 TTS 試聽表</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", sans-serif; margin: 24px; background: #faf8f5; color: #171312; }}
    h1 {{ margin-bottom: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #ddd7d0; padding: 10px; vertical-align: top; }}
    th {{ background: #f2eee8; text-align: left; }}
    audio {{ width: 220px; }}
    .hint {{ color: #6f6560; margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>紅色蝴蝶結 TTS 試聽表</h1>
  <p class="hint">Provider: {html.escape(provider)}. 評分建議：相似感 / 中文清楚度 / 情緒 / 穩定度 / 是否值得繼續。</p>
  <table>
    <thead>
      <tr>
        <th>目標</th>
        <th>聲音目標</th>
        <th>Reference</th>
        <th>台詞</th>
        <th>音訊</th>
        <th>耗時</th>
        <th>狀態</th>
        <th>錯誤</th>
        <th>人工評語</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (run_dir / "index.html").write_text(report, encoding="utf-8")
    machine = [
        {
            "target_id": r.target_id,
            "target_name": r.target_name,
            "goal": r.goal,
            "reference_audio": str(r.reference_audio) if r.reference_audio else None,
            "test_index": r.test_index,
            "text": r.text,
            "output": str(r.output) if r.output else None,
            "seconds": r.seconds,
            "status": r.status,
            "error": r.error,
        }
        for r in results
    ]
    (run_dir / "results.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["macos-say", "f5-cli", "sherpa-zipvoice", "sherpa-pocket", "qwen3-mlx"],
        default="macos-say",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(run(args.provider, args.limit, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

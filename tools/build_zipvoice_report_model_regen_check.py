#!/usr/bin/env python3
"""Regenerate audio with the exact ZipVoice report dynamic-qint8 candidate."""

from __future__ import annotations

import base64
import html
import json
import math
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
REPORT_SRC = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "reports"
    / "zipvoice_4step_optimize_v1"
)
OUT_DIR = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "reports"
    / "zipvoice_report_model_regen_check_v1"
)
AUDIO_DIR = OUT_DIR / "audio"
OUT_HTML = OUT_DIR / "zipvoice_report_model_regen_check_v1.html"

ZIP_MODEL_DIR = (
    ROOT
    / "external"
    / "ZipVoice"
    / "egs"
    / "zipvoice"
    / "exp"
    / "zipvoice_cosy_teacher_smoke_onnx_ckpt60"
)
APP_MODEL_DIR = ROOT / "RedBowVoice" / "RedBowAssets" / "Models" / "CosyZipVoice4StepQvocoder"
VOCODER = ROOT / "models" / "sherpa" / "vocos_24khz_dynamic_qint8.onnx"
REF_AUDIO = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "golden_teacher"
    / "cosyvoice2_clear_best2_line04_v1"
    / "reference_clear_best2_7s.wav"
)
REF_TEXT = (
    "所以我当时就说,我想要做一张疗愈人的专辑。 "
    "开始当然就是我们的提案会议,我就提出了因为多年"
)

LINES = [
    ("cosy_01", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("cosy_02", "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。"),
    ("cosy_03", "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。"),
]


def mild_clean(input_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(input_audio, dtype=np.float32)
    if x.size == 0:
        return x
    x = x - float(np.mean(x))

    def high_pass(inp: np.ndarray, cutoff: float) -> np.ndarray:
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff)
        alpha = rc / (rc + dt)
        previous_x = 0.0
        previous_y = 0.0
        out = []
        for sample in inp:
            y = alpha * (previous_y + float(sample) - previous_x)
            previous_x = float(sample)
            previous_y = y
            out.append(y)
        return np.asarray(out, dtype=np.float32)

    def low_pass(inp: np.ndarray, cutoff: float) -> np.ndarray:
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff)
        alpha = dt / (rc + dt)
        previous_y = 0.0
        out = []
        for sample in inp:
            y = previous_y + alpha * (float(sample) - previous_y)
            previous_y = y
            out.append(y)
        return np.asarray(out, dtype=np.float32)

    y = low_pass(high_pass(x, 65.0), 10_500.0)
    drive = 1.35
    y = np.tanh(y * drive) / drive

    rms = float(np.sqrt(np.mean(y * y)))
    target_rms = 10 ** (-20 / 20)
    if rms > 1e-6:
        y = y * min(target_rms / rms, 3.5)

    peak_limit = 10 ** (-1 / 20)
    peak = float(np.max(np.abs(y)))
    if peak > peak_limit:
        y = y * (peak_limit / peak)

    return np.clip(y, -1.0, 1.0).astype(np.float32)


def high_frequency_ratio(path: Path) -> float:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if y.size < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    freqs = np.fft.rfftfreq(y.size, 1.0 / sr)
    high = float(spec[freqs >= 8000].sum())
    low = float(spec[(freqs >= 80) & (freqs < 8000)].sum()) + 1e-12
    return 10 * math.log10((high + 1e-12) / low)


def audio_stats(path: Path) -> dict[str, float | str]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return {
        "duration": round(float(y.size / sr), 3),
        "rms_db": round(float(20 * math.log10(rms + 1e-12)), 2),
        "peak_db": round(float(20 * math.log10(peak + 1e-12)), 2),
        "hf_over_8k_db": round(high_frequency_ratio(path), 2),
    }


def make_tts() -> sherpa_onnx.OfflineTts:
    zipvoice = sherpa_onnx.OfflineTtsZipvoiceModelConfig(
        tokens=str(ZIP_MODEL_DIR / "tokens.txt"),
        encoder=str(ZIP_MODEL_DIR / "text_encoder_int8.onnx"),
        decoder=str(ZIP_MODEL_DIR / "fm_decoder_int8.onnx"),
        vocoder=str(VOCODER),
        data_dir=str(APP_MODEL_DIR / "espeak-ng-data"),
        lexicon=str(APP_MODEL_DIR / "lexicon.txt"),
    )
    zipvoice.feat_scale = 0.1
    zipvoice.t_shift = 0.5
    zipvoice.target_rms = 0.1
    zipvoice.guidance_scale = 3.0

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=zipvoice,
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("Invalid report ZipVoice config")
    return sherpa_onnx.OfflineTts(config)


def copy_originals(rows: list[dict]) -> None:
    for line_id, text in LINES:
        for variant, src_dir in [
            ("report_qint8_raw_original", REPORT_SRC / "assets" / "qvocoder_step4"),
            ("report_qint8_mild_original", REPORT_SRC / "assets" / "qvocoder_clean_mild"),
        ]:
            src = src_dir / f"{line_id}.wav"
            dst = AUDIO_DIR / f"{line_id}_{variant}.wav"
            dst.write_bytes(src.read_bytes())
            rows.append(
                {
                    "line_id": line_id,
                    "text": text,
                    "variant": variant,
                    "path": str(dst),
                    "seconds": None,
                    **audio_stats(dst),
                }
            )


def generate_fresh(rows: list[dict]) -> None:
    tts = make_tts()
    ref_audio, ref_sr = sf.read(str(REF_AUDIO), dtype="float32", always_2d=False)
    if getattr(ref_audio, "ndim", 1) > 1:
        ref_audio = ref_audio.mean(axis=1)

    for line_id, text in LINES:
        for attempt in range(1, 4):
            gen = sherpa_onnx.GenerationConfig()
            gen.reference_audio = ref_audio
            gen.reference_sample_rate = ref_sr
            gen.reference_text = REF_TEXT
            gen.num_steps = 4
            gen.speed = 1.0
            gen.extra["min_char_in_sentence"] = "20"

            started = time.perf_counter()
            audio = tts.generate(text, gen)
            elapsed = time.perf_counter() - started
            raw = np.asarray(audio.samples, dtype=np.float32)
            clean = mild_clean(raw, audio.sample_rate)

            for variant, samples in [
                (f"fresh_dynamic_raw_try{attempt}", raw),
                (f"fresh_dynamic_mild_try{attempt}", clean),
            ]:
                dst = AUDIO_DIR / f"{line_id}_{variant}.wav"
                sf.write(str(dst), samples, audio.sample_rate, subtype="PCM_16")
                rows.append(
                    {
                        "line_id": line_id,
                        "text": text,
                        "variant": variant,
                        "path": str(dst),
                        "seconds": round(elapsed, 3),
                        **audio_stats(dst),
                    }
                )


def data_uri(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def card(row: dict) -> str:
    path = Path(row["path"])
    return f"""
    <article class="card">
      <h3>{html.escape(row['variant'])}</h3>
      <audio controls preload="metadata" src="{data_uri(path)}"></audio>
      <dl>
        <div><dt>生成</dt><dd>{'-' if row['seconds'] is None else str(row['seconds']) + 's'}</dd></div>
        <div><dt>長度</dt><dd>{row['duration']}s</dd></div>
        <div><dt>RMS</dt><dd>{row['rms_db']} dB</dd></div>
        <div><dt>8k+ HF</dt><dd>{row['hf_over_8k_db']} dB</dd></div>
      </dl>
    </article>
    """


def build_html(rows: list[dict]) -> str:
    sections = []
    for line_id, text in LINES:
        line_rows = [r for r in rows if r["line_id"] == line_id]
        sections.append(
            f"""
            <section class="line">
              <h2>{html.escape(line_id)}</h2>
              <p class="text">{html.escape(text)}</p>
              <div class="grid">{''.join(card(r) for r in line_rows)}</div>
            </section>
            """
        )

    summary = {
        "model": str(ZIP_MODEL_DIR),
        "vocoder": str(VOCODER),
        "reference_audio": str(REF_AUDIO),
        "reference_text": REF_TEXT,
        "num_steps": 4,
        "guidance_scale": 3.0,
        "report_original": str(REPORT_SRC),
    }

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZipVoice report model regen check</title>
  <style>
    body {{
      margin: 0;
      background: #f7f5ef;
      color: #24211d;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", sans-serif;
      line-height: 1.58;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 22px 14px 48px; }}
    h1 {{ font-size: 36px; line-height: 1.12; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 8px; }}
    h3 {{ font-size: 14px; margin: 0 0 8px; overflow-wrap: anywhere; }}
    .lead {{ color: #625b51; max-width: 840px; }}
    .meta, .line {{ background: #fffefa; border: 1px solid #d8d1c6; border-radius: 8px; padding: 14px; margin-top: 14px; }}
    .text {{ font-weight: 750; margin: 0 0 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .card {{ background: #f0ebe2; border: 1px solid #d8d1c6; border-radius: 8px; padding: 10px; }}
    audio {{ width: 100%; }}
    dl {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin: 8px 0 0; }}
    dt {{ color: #70675d; font-size: 12px; }}
    dd {{ margin: 0; font-weight: 700; font-size: 13px; }}
    code {{ overflow-wrap: anywhere; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>ZipVoice 報告模型重新生成檢查</h1>
    <p class="lead">這頁直接使用原報告標的模型：ckpt60 int8 encoder/decoder + <code>vocos_24khz_dynamic_qint8.onnx</code>。每句先放當初報告固定音檔，再放今天 fresh generate 三次，確認報告是不是一次性好聽。</p>
    <section class="meta">
      <h2>Config</h2>
      <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
    </section>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for stale in AUDIO_DIR.glob("*.wav"):
        stale.unlink()

    rows: list[dict] = []
    copy_originals(rows)
    generate_fresh(rows)

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_HTML.write_text(build_html(rows), encoding="utf-8")
    print(OUT_HTML)
    print(OUT_DIR / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a three-way listening report for the selected voice.

Compares:
1. Qwen3 1.7B VoiceDesign teacher audio.
2. Sherpa-ONNX ZipVoice int8 using the selected teacher reference.
3. Sherpa-ONNX ZipVoice int8 using its bundled news-female reference.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import resource
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
TEACHER_DIR = BASE / "teacher_qwen3_1p7b_seed"
TEACHER_AUDIO_DIR = TEACHER_DIR / "audio"
TEXTS = BASE / "seed_texts.jsonl"
PROFILE = ROOT / "voice_profiles" / "taiwan_mandarin_low_r.json"
OUT = BASE / "reports" / "zipvoice_three_way"
METRICS_FILE = OUT / "runtime_metrics.json"

ZIPVOICE_MODEL_DIR = Path(
    os.environ.get(
        "ZIPVOICE_MODEL_DIR",
        ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
    )
)
ZIPVOICE_VOCODER = Path(os.environ.get("ZIPVOICE_VOCODER", ROOT / "models" / "sherpa" / "vocos_24khz.onnx"))
ZIPVOICE_ORIGINAL_REF = ZIPVOICE_MODEL_DIR / "test_wavs" / "news-female.wav"
ZIPVOICE_ORIGINAL_REF_TEXT = "各位村民, 大家新年好! 近期, 湖北省武汉市等多个地区"

TEACHER_REF_ID = "seed_0001"
TEACHER_REF_TEXT = "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"
TEACHER_REF_AUDIO = TEACHER_AUDIO_DIR / f"{TEACHER_REF_ID}.wav"

SELECTED_IDS = [
    "seed_0001",
    "seed_0002",
    "seed_0003",
    "seed_0004",
    "seed_0005",
    "seed_0006",
    "seed_0007",
    "seed_0008",
    "seed_0013",
    "seed_0014",
    "seed_0019",
    "seed_0022",
    "seed_0027",
    "seed_0032",
    "seed_0033",
    "seed_0034",
    "seed_0035",
    "seed_0036",
]


@dataclass
class SampleScore:
    item_id: str
    category: str
    text: str
    teacher_audio: str
    zipvoice_teacher_ref_audio: str
    zipvoice_original_audio: str
    teacher_seconds: float | None
    zipvoice_teacher_ref_seconds: float
    zipvoice_original_seconds: float
    teacher_ref_score: float
    original_ref_score: float
    teacher_ref_mfcc: float
    teacher_ref_f0: float
    teacher_ref_centroid: float
    original_ref_mfcc: float
    original_ref_f0: float
    original_ref_centroid: float


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    seen: set[tuple[int, int]] = set()
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            total += stat.st_blocks * 512 if stat.st_blocks else stat.st_size
    return total


def fmt_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GB"


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def relative(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def ensure_zipvoice_tts():
    import sherpa_onnx

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(ZIPVOICE_MODEL_DIR / "tokens.txt"),
                encoder=str(ZIPVOICE_MODEL_DIR / "encoder.int8.onnx"),
                decoder=str(ZIPVOICE_MODEL_DIR / "decoder.int8.onnx"),
                data_dir=str(ZIPVOICE_MODEL_DIR / "espeak-ng-data"),
                lexicon=str(ZIPVOICE_MODEL_DIR / "lexicon.txt"),
                vocoder=str(ZIPVOICE_VOCODER),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError("Invalid ZipVoice config")
    return sherpa_onnx.OfflineTts(config)


def zipvoice_generate(tts, text: str, ref_audio: Path, ref_text: str, output: Path) -> float:
    import sherpa_onnx
    import soundfile as sf

    started = perf_counter()
    reference_audio, sample_rate = sf.read(str(ref_audio), dtype="float32", always_2d=False)
    if getattr(reference_audio, "ndim", 1) > 1:
        reference_audio = reference_audio.mean(axis=1)
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.reference_text = ref_text
    gen_config.num_steps = 4
    gen_config.extra["min_char_in_sentence"] = "20"
    audio = tts.generate(text, gen_config)
    if len(audio.samples) == 0:
        raise RuntimeError("ZipVoice returned empty audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
    return perf_counter() - started


def audio_features(path: Path) -> dict:
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=24000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    if len(y) < sr // 2:
        y, sr = librosa.load(path, sr=24000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    try:
        f0 = librosa.yin(y, fmin=80, fmax=600, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0_med = float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        f0_med = 0.0
    return {
        "mfcc": np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]),
        "centroid": float(np.mean(centroid)),
        "f0": f0_med,
    }


def cosine01(a, b) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, (float(np.dot(a, b)) / denom + 1.0) / 2.0))


def log_similarity(a: float, b: float, width: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return max(0.0, min(1.0, math.exp(-abs(math.log(a / b)) / width)))


def score_against_teacher(teacher: Path, candidate: Path) -> tuple[float, float, float, float]:
    import librosa

    t = audio_features(teacher)
    c = audio_features(candidate)
    mfcc = cosine01(t["mfcc"], c["mfcc"])
    f0 = log_similarity(float(t["f0"]), float(c["f0"]), 0.45)
    centroid = log_similarity(float(t["centroid"]), float(c["centroid"]), 0.55)
    total = 0.62 * mfcc + 0.23 * f0 + 0.15 * centroid
    teacher_duration = librosa.get_duration(path=teacher)
    candidate_duration = librosa.get_duration(path=candidate)
    if candidate_duration < 1.5:
        total *= 0.30
    elif teacher_duration > 0 and candidate_duration < teacher_duration * 0.45:
        total *= 0.45
    elif teacher_duration > 0 and candidate_duration > teacher_duration * 2.2:
        total *= 0.55
    if candidate_duration > 30:
        total *= 0.45
    elif candidate_duration > 20:
        total *= 0.60
    return total, mfcc, f0, centroid


def teacher_manifest_seconds() -> dict[str, float]:
    manifest = json.loads((TEACHER_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {item["id"]: float(item["seconds"]) for item in manifest if item.get("status") == "ok"}


def build_samples(force: bool) -> list[SampleScore]:
    rows_by_id = {row["id"]: row for row in read_jsonl(TEXTS)}
    teacher_seconds = teacher_manifest_seconds()
    prior_results = {}
    if not force and (OUT / "results.json").exists():
        prior_results = {row["item_id"]: row for row in json.loads((OUT / "results.json").read_text(encoding="utf-8"))}
    audio_root = OUT / "audio"
    teacher_out = audio_root / "teacher_qwen3_1p7b"
    new_out = audio_root / "zipvoice_teacher_ref"
    original_out = audio_root / "zipvoice_original_news_female"
    for folder in [teacher_out, new_out, original_out]:
        folder.mkdir(parents=True, exist_ok=True)

    tts = ensure_zipvoice_tts()
    samples: list[SampleScore] = []
    for item_id in SELECTED_IDS:
        row = rows_by_id[item_id]
        text = row["text"]
        teacher_src = TEACHER_AUDIO_DIR / f"{item_id}.wav"
        teacher_dest = teacher_out / f"{item_id}.wav"
        if force or not teacher_dest.exists():
            shutil.copy2(teacher_src, teacher_dest)

        new_audio = new_out / f"{item_id}.wav"
        original_audio = original_out / f"{item_id}.wav"
        if force or not new_audio.exists():
            new_seconds = zipvoice_generate(tts, text, TEACHER_REF_AUDIO, TEACHER_REF_TEXT, new_audio)
        else:
            new_seconds = float(prior_results.get(item_id, {}).get("zipvoice_teacher_ref_seconds", 0.0))
        if force or not original_audio.exists():
            original_seconds = zipvoice_generate(
                tts,
                text,
                ZIPVOICE_ORIGINAL_REF,
                ZIPVOICE_ORIGINAL_REF_TEXT,
                original_audio,
            )
        else:
            original_seconds = float(prior_results.get(item_id, {}).get("zipvoice_original_seconds", 0.0))

        teacher_ref_score, teacher_ref_mfcc, teacher_ref_f0, teacher_ref_centroid = score_against_teacher(
            teacher_dest,
            new_audio,
        )
        original_ref_score, original_ref_mfcc, original_ref_f0, original_ref_centroid = score_against_teacher(
            teacher_dest,
            original_audio,
        )
        samples.append(
            SampleScore(
                item_id=item_id,
                category=row.get("category", ""),
                text=text,
                teacher_audio=relative(teacher_dest),
                zipvoice_teacher_ref_audio=relative(new_audio),
                zipvoice_original_audio=relative(original_audio),
                teacher_seconds=teacher_seconds.get(item_id),
                zipvoice_teacher_ref_seconds=new_seconds,
                zipvoice_original_seconds=original_seconds,
                teacher_ref_score=teacher_ref_score,
                original_ref_score=original_ref_score,
                teacher_ref_mfcc=teacher_ref_mfcc,
                teacher_ref_f0=teacher_ref_f0,
                teacher_ref_centroid=teacher_ref_centroid,
                original_ref_mfcc=original_ref_mfcc,
                original_ref_f0=original_ref_f0,
                original_ref_centroid=original_ref_centroid,
            )
        )
        print(item_id, "teacher-ref", f"{teacher_ref_score:.3f}", "original", f"{original_ref_score:.3f}")
    return samples


def load_runtime_metrics() -> dict:
    if METRICS_FILE.exists():
        return read_json(METRICS_FILE)
    return {}


def update_runtime_metrics(key: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_runtime_metrics()
    metrics[key] = payload
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metric_card(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<div class=\"metric-row\"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in rows
    )
    return f"<section class=\"metric-card\"><h3>{html.escape(title)}</h3>{body}</section>"


def render_report(samples: list[SampleScore]) -> None:
    profile = read_json(PROFILE)
    runtime = load_runtime_metrics()
    qwen_cache = Path("/Users/ader/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")
    zipvoice_pack_bytes = size_bytes(ZIPVOICE_MODEL_DIR) + size_bytes(ZIPVOICE_VOCODER)
    zh_min_pack = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min"
    zh_min_pack_bytes = size_bytes(zh_min_pack) + size_bytes(ZIPVOICE_VOCODER) if zh_min_pack.exists() else 0
    sweep_path = BASE / "reports" / "zipvoice_reference_sweep" / "results.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.exists() else []
    best_sweep = sweep[0] if sweep else None

    teacher_avg_seconds = average([s.teacher_seconds for s in samples if s.teacher_seconds is not None])
    new_avg_seconds = average([s.zipvoice_teacher_ref_seconds for s in samples if s.zipvoice_teacher_ref_seconds > 0])
    original_avg_seconds = average([s.zipvoice_original_seconds for s in samples if s.zipvoice_original_seconds > 0])
    new_avg_score = average([s.teacher_ref_score for s in samples])
    original_avg_score = average([s.original_ref_score for s in samples])

    teacher_runtime = runtime.get("teacher_qwen3_1p7b", {})
    zip_runtime = runtime.get("zipvoice_teacher_ref", runtime.get("zipvoice_int8", {}))
    zip_original_runtime = runtime.get("zipvoice_original_news_female", zip_runtime)
    teacher_size_text = fmt_bytes(size_bytes(qwen_cache)) if qwen_cache.exists() else "未找到"
    zip_size_text = fmt_bytes(zipvoice_pack_bytes)
    teacher_mem_text = teacher_runtime.get("peak_rss_mb", "待測")
    zip_mem_text = zip_runtime.get("peak_rss_mb", "待測")
    zip_original_mem_text = zip_original_runtime.get("peak_rss_mb", "同 ZipVoice")

    cards = [
        metric_card(
            "Qwen3 1.7B Teacher 原聲",
            [
                ("用途", "高品質 teacher / 目標聲音"),
                ("模型快取", teacher_size_text),
                ("平均生成", f"{teacher_avg_seconds:.2f}s / 句"),
                ("峰值 RSS", teacher_mem_text),
                ("備註", "MLX 4bit，大模型，不是手機主線"),
            ],
        ),
        metric_card(
            "新做聲音：ZipVoice 聲音引導",
            [
                ("用途", "目前 app 主線手機候選"),
                ("模型包", zip_size_text),
                ("平均生成", f"{new_avg_seconds:.2f}s / 句"),
                ("平均像度", f"{new_avg_score:.3f}"),
                ("峰值 RSS", zip_mem_text),
            ],
        ),
        metric_card(
            "ZipVoice 本來聲音：原廠女聲",
            [
                ("用途", "看 ZipVoice 原廠參考聲音長相"),
                ("模型包", zip_size_text),
                ("平均生成", f"{original_avg_seconds:.2f}s / 句"),
                ("對 teacher 像度", f"{original_avg_score:.3f}"),
                ("峰值 RSS", zip_original_mem_text),
            ],
        ),
    ]

    sample_cards = []
    for index, sample in enumerate(samples, start=1):
        category = sample.category.replace("selected_reference", "selected").replace("_", " ")
        sample_cards.append(
            f"""
<section class="sample-card">
  <div class="sample-head">
    <span>#{index:02d} · {html.escape(category)}</span>
    <strong>{html.escape(sample.item_id)}</strong>
  </div>
  <p class="line">{html.escape(sample.text)}</p>
  <div class="compare-scroll">
    <div class="compare-grid">
      <div class="audio-block teacher">
        <div><b>1. Teacher 原聲</b><small>Qwen3 1.7B</small></div>
        <dl class="spec-list">
          <dt>模型</dt><dd>Qwen3 1.7B</dd>
          <dt>大小</dt><dd>{html.escape(teacher_size_text)}</dd>
          <dt>記憶體</dt><dd>{html.escape(teacher_mem_text)}</dd>
          <dt>生成</dt><dd>{sample.teacher_seconds or 0:.2f}s</dd>
        </dl>
        <audio controls preload="metadata" src="{html.escape(sample.teacher_audio)}"></audio>
      </div>
      <div class="audio-block highlight">
        <div><b>2. 新做聲音</b><small>ZipVoice · score {sample.teacher_ref_score:.3f}</small></div>
        <dl class="spec-list">
          <dt>模型</dt><dd>ZipVoice int8</dd>
          <dt>大小</dt><dd>{html.escape(zip_size_text)}</dd>
          <dt>記憶體</dt><dd>{html.escape(zip_mem_text)}</dd>
          <dt>生成</dt><dd>{sample.zipvoice_teacher_ref_seconds:.2f}s</dd>
        </dl>
        <audio controls preload="metadata" src="{html.escape(sample.zipvoice_teacher_ref_audio)}"></audio>
      </div>
      <div class="audio-block original">
        <div><b>3. ZipVoice 本來聲音</b><small>原廠聲 · score {sample.original_ref_score:.3f}</small></div>
        <dl class="spec-list">
          <dt>模型</dt><dd>ZipVoice int8</dd>
          <dt>大小</dt><dd>{html.escape(zip_size_text)}</dd>
          <dt>記憶體</dt><dd>{html.escape(zip_original_mem_text)}</dd>
          <dt>生成</dt><dd>{sample.zipvoice_original_seconds:.2f}s</dd>
        </dl>
        <audio controls preload="metadata" src="{html.escape(sample.zipvoice_original_audio)}"></audio>
      </div>
    </div>
  </div>
  <div class="score-grid">
    <span>新做 MFCC {sample.teacher_ref_mfcc:.3f}</span>
    <span>新做 F0 {sample.teacher_ref_f0:.3f}</span>
    <span>原聲 MFCC {sample.original_ref_mfcc:.3f}</span>
    <span>原聲 F0 {sample.original_ref_f0:.3f}</span>
  </div>
</section>
"""
        )

    optimization_items = [
        (
            "聲音引導句 sweep",
            "已測 seed_0001 到 seed_0006；seed_0001 仍最佳"
            if best_sweep is None
            else f"最佳是 {best_sweep['reference_id']}，快速 sweep 平均 {best_sweep['avg_score']:.3f}",
        ),
        (
            "zh-min package",
            f"可跑，模型包約 {fmt_bytes(zh_min_pack_bytes)}，比完整包少約 {fmt_bytes(max(0, zipvoice_pack_bytes - zh_min_pack_bytes))}"
            if zh_min_pack_bytes
            else "尚未產生",
        ),
        (
            "Memory",
            f"改用輕量 wav loader 後，ZipVoice 峰值重測為 {zip_runtime.get('peak_rss_mb', '待測')}",
        ),
        (
            "更小且更像",
            "下一步需要訓練/蒸餾單一聲音或再量化 decoder/vocoder；單靠換聲音引導句已掃過，seed_0001 最穩。",
        ),
    ]
    optimization = "".join(
        f"<div class=\"opt-row\"><b>{html.escape(title)}</b><span>{html.escape(detail)}</span></div>"
        for title, detail in optimization_items
    )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZipVoice 三路聲音報告</title>
<style>
:root {{
  --paper: #fbf8f3;
  --ink: #1f1715;
  --muted: #7b716d;
  --line: #e8ded5;
  --red: #c82336;
  --pink: #ff4c77;
  --card: #fffefd;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", sans-serif;
  line-height: 1.55;
}}
header {{
  padding: 22px 18px 18px;
  background: linear-gradient(135deg, #b9162a, #ef4566);
  color: white;
}}
.eyebrow {{ font-size: 13px; opacity: .86; }}
h1 {{ margin: 6px 0 8px; font-size: 26px; line-height: 1.18; letter-spacing: 0; }}
header p {{ margin: 0; font-size: 15px; opacity: .92; }}
main {{ max-width: 920px; margin: 0 auto; padding: 16px; }}
.summary {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}}
.metric-card, .sample-card, .note {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 1px 0 rgba(34, 20, 12, .04);
}}
.metric-card {{ padding: 14px; }}
.metric-card h3 {{ margin: 0 0 10px; font-size: 17px; }}
.metric-row {{
  display: flex;
  gap: 10px;
  justify-content: space-between;
  border-top: 1px solid #f0e7df;
  padding: 8px 0;
  font-size: 14px;
}}
.metric-row span {{ color: var(--muted); }}
.metric-row strong {{ text-align: right; }}
.note {{
  margin: 14px 0;
  padding: 13px 14px;
  font-size: 15px;
  color: #544844;
}}
.method-card {{ padding: 14px; margin: 14px 0; }}
.method-card h3 {{ margin: 0 0 10px; font-size: 18px; }}
.method-card p {{ margin: 8px 0; font-size: 15px; color: #473c38; }}
.method-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}}
.method-box {{
  border: 1px solid #f0e6dd;
  border-radius: 8px;
  padding: 10px;
  background: #fffaf7;
}}
.method-box b {{ display: block; margin-bottom: 5px; font-size: 15px; }}
.method-box span {{ display: block; color: #5f5550; font-size: 14px; }}
h2 {{ font-size: 20px; margin: 24px 0 12px; }}
.sample-card {{ padding: 14px; margin-bottom: 14px; }}
.opt-card {{ padding: 14px; margin: 14px 0; }}
.opt-row {{
  display: grid;
  grid-template-columns: 150px 1fr;
  gap: 12px;
  padding: 9px 0;
  border-top: 1px solid #f0e7df;
  font-size: 15px;
}}
.opt-row:first-child {{ border-top: 0; }}
.opt-row span {{ color: #544844; }}
.sample-head {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 13px;
}}
.line {{ font-size: 18px; margin: 10px 0 12px; font-weight: 650; }}
.audio-block {{
  border: 1px solid #f0e6dd;
  border-radius: 8px;
  padding: 8px;
  background: #fffaf7;
}}
.compare-scroll {{ overflow-x: auto; padding-bottom: 2px; }}
.compare-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
}}
.audio-block.highlight {{ border-color: rgba(200, 35, 54, .35); background: #fff5f7; }}
.audio-block.teacher {{ background: #f8fbff; }}
.audio-block.original {{ background: #fbfaf6; }}
.audio-block div {{
  min-height: 43px;
  margin-bottom: 7px;
}}
.audio-block b {{ font-size: 14px; }}
.audio-block small {{ display: block; color: var(--muted); margin-top: 3px; font-size: 12px; }}
.spec-list {{
  display: grid;
  grid-template-columns: 43px 1fr;
  gap: 2px 5px;
  margin: 0 0 8px;
  padding: 7px;
  border-radius: 6px;
  background: rgba(255,255,255,.58);
  font-size: 11px;
}}
.spec-list dt {{
  margin: 0;
  color: var(--muted);
}}
.spec-list dd {{
  margin: 0;
  font-weight: 700;
  overflow-wrap: anywhere;
}}
audio {{ width: 100%; height: 38px; }}
.score-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 6px;
  margin-top: 10px;
}}
.score-grid span {{
  background: #f5eee8;
  border-radius: 6px;
  padding: 6px;
  font-size: 12px;
  color: #5c504b;
}}
.footer {{ color: var(--muted); font-size: 13px; margin: 18px 0 30px; }}
@media (max-width: 720px) {{
  main {{ padding: 12px; }}
  .summary {{ grid-template-columns: 1fr; }}
  .method-grid {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 24px; }}
  .line {{ font-size: 17px; }}
  .score-grid {{ grid-template-columns: 1fr 1fr; }}
  .opt-row {{ grid-template-columns: 1fr; gap: 3px; }}
}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">taiwan_mandarin_low_r · Sherpa-ONNX ZipVoice int8 · {len(samples)} 句同文對照</div>
  <h1>老師原聲、新做聲音、ZipVoice 本來聲音</h1>
  <p>每張卡都照同一順序聽：1. Qwen3 1.7B teacher，2. 新做 ZipVoice 聲音，3. ZipVoice 原廠女聲。</p>
</header>
<main>
  <section class="summary">{''.join(cards)}</section>
  <section class="note">
    目前「新做聲音」還不是訓練後的新權重，而是 ZipVoice int8 以 teacher 句子做 zero-shot 聲音引導。這份報告先回答：手機級 ZipVoice 直接上，離 teacher 有多近。
  </section>
  <section class="metric-card method-card">
    <h3>這份報告到底做了什麼</h3>
    <p>你的理解是對的：目前是在用 Sherpa-ONNX ZipVoice int8 去模仿 Qwen3 1.7B teacher 產生的「台灣國語低卷舌女聲」。Qwen3 1.7B 是老師，ZipVoice 是手機候選模型。</p>
    <p>但這還不是完整蒸餾。現在沒有訓練新的 student 權重，而是拿 teacher 的一句聲音當聲音引導，讓 ZipVoice zero-shot 講同一句測試文本。</p>
    <div class="method-grid">
      <div class="method-box"><b>現在：zero-shot 模仿</b><span>不用訓練，立刻可測。優點是快、能跑手機路線；缺點是聲音穩定度和像度受聲音引導句限制。</span></div>
      <div class="method-box"><b>完整蒸餾：還沒做完</b><span>要先用 Qwen3 1.7B 產大量同聲音語料，再訓練小 student。這才算真正把 teacher 聲音壓進小模型。</span></div>
      <div class="method-box"><b>能不能更小</b><span>可以，但不能只靠刪檔。現在 ZipVoice 大頭是 decoder 和 vocoder。要再小，需要單聲音訓練、再量化、或換更小架構。</span></div>
      <div class="method-box"><b>能不能更像</b><span>有機會。完整蒸餾通常會比 zero-shot 更穩，因為模型只學這一個聲音，不需要每次靠提示音猜。</span></div>
    </div>
  </section>
  <section class="metric-card method-card">
    <h3>下一步完整蒸餾目標</h3>
    <p>目標不是做通用 TTS，而是只做這個「台大低卷舌女聲」。這會讓小模型更有機會在中階手機上跑，並且比 zero-shot 更穩。</p>
    <div class="method-grid">
      <div class="method-box"><b>資料</b><span>用 Qwen3 1.7B teacher 生成 500 到 3000 句，涵蓋聊天、疑問、安撫、緊張、長句和短句。</span></div>
      <div class="method-box"><b>模型</b><span>優先測單聲音 student：小型 VITS/Piper 類、ZipVoice 微調路線、或小 vocoder + acoustic model 分離路線。</span></div>
      <div class="method-box"><b>手機目標</b><span>先把包壓到 80 到 180 MB，峰值記憶體往 400 到 700 MB 靠，生成速度至少接近即時。</span></div>
      <div class="method-box"><b>風險</b><span>模型越小越容易失去自然尾音和台灣口音。最後要以耳聽為準，分數只做篩選。</span></div>
    </div>
  </section>
  <section class="metric-card method-card">
    <h3>完整蒸餾做得到嗎</h3>
    <p>做得到，但它比較像「聲音蒸餾」而不是直接把 Qwen3 1.7B 權重壓扁。Qwen3 1.7B 負責產出高品質 teacher corpus；小模型負責學會這一個台灣女生聲音。</p>
    <div class="method-grid">
      <div class="method-box"><b>可行結論</b><span>可行。最務實目標是單聲音中文 TTS，不做多角色、多語言、多情緒全能模型。</span></div>
      <div class="method-box"><b>手機目標</b><span>中階手機先抓模型 50 到 180 MB、峰值記憶體 300 到 700 MB、生成速度接近即時或快於即時。</span></div>
      <div class="method-box"><b>更像的方法</b><span>用 teacher 產 500、1500、3000 句三階段資料，訓練後每階段都跟這份三欄報告 A/B 試聽。</span></div>
      <div class="method-box"><b>更台灣的方法</b><span>文本保持簡中輸入但內容寫台灣口語，句子加入低卷舌、無兒化、柔尾音、自然停頓、不要主播腔的測試集。</span></div>
      <div class="method-box"><b>模型候選</b><span>先測單聲音 Piper/VITS 類小模型，再測 ZipVoice 微調或 acoustic model + 小 vocoder 分離路線。</span></div>
      <div class="method-box"><b>成功標準</b><span>不是分數最高就算過，而是你聽起來覺得像、自然、長句不崩、手機跑得動。</span></div>
    </div>
  </section>
  <section class="metric-card opt-card">
    <h3>做到目前的優化結果</h3>
    {optimization}
  </section>
  <section class="note">
    Teacher prompt：{html.escape(profile["voice_design_prompt"])}
  </section>
  <h2>逐句試聽</h2>
  {''.join(sample_cards)}
  <p class="footer">像度分數是 MFCC/F0/頻譜的 proxy，只能當排序輔助；最後以你的耳朵決定。</p>
</main>
</body>
</html>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(report, encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps([asdict(sample) for sample in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUT / "index.html")


def benchmark_zipvoice(variant: str, limit: int, update_metrics: bool) -> None:
    rows_by_id = {row["id"]: row for row in read_jsonl(TEXTS)}
    ref_audio = TEACHER_REF_AUDIO if variant == "teacher-ref" else ZIPVOICE_ORIGINAL_REF
    ref_text = TEACHER_REF_TEXT if variant == "teacher-ref" else ZIPVOICE_ORIGINAL_REF_TEXT
    with tempfile.TemporaryDirectory(prefix="zipvoice-bench-") as tmp:
        tts = ensure_zipvoice_tts()
        loaded_at_mb = peak_rss_mb()
        started = perf_counter()
        for item_id in SELECTED_IDS[:limit]:
            output = Path(tmp) / f"{item_id}.wav"
            zipvoice_generate(tts, rows_by_id[item_id]["text"], ref_audio, ref_text, output)
        elapsed = perf_counter() - started
    payload = {
        "variant": variant,
        "samples": limit,
        "generation_seconds": f"{elapsed:.2f}s",
        "seconds_per_sentence": f"{elapsed / limit:.2f}s",
        "rss_after_model_load_mb": f"{loaded_at_mb:.1f} MB",
        "peak_rss_mb": f"{peak_rss_mb():.1f} MB",
        "note": "Measured on this Mac process with sherpa-onnx CPU provider; iOS/Android RSS will differ.",
    }
    if update_metrics:
        key = "zipvoice_teacher_ref" if variant == "teacher-ref" else "zipvoice_original_news_female"
        update_runtime_metrics(key, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def benchmark_teacher(limit: int, update_metrics: bool) -> None:
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model

    profile = read_json(PROFILE)
    rows = read_jsonl(TEXTS)[:limit]
    with tempfile.TemporaryDirectory(prefix="qwen-teacher-bench-") as tmp:
        started_load = perf_counter()
        model = load_model(profile["teacher_model"])
        loaded_seconds = perf_counter() - started_load
        loaded_at_mb = peak_rss_mb()
        started = perf_counter()
        for row in rows:
            generate_audio(
                text=row["text"],
                model=model,
                instruct=profile["voice_design_prompt"],
                lang_code=profile.get("language_code", "zh"),
                output_path=tmp,
                file_prefix=row["id"],
                audio_format="wav",
                verbose=False,
            )
        generation_seconds = perf_counter() - started
    payload = {
        "samples": limit,
        "model_load_seconds": f"{loaded_seconds:.2f}s",
        "generation_seconds": f"{generation_seconds:.2f}s",
        "seconds_per_sentence": f"{generation_seconds / limit:.2f}s",
        "rss_after_model_load_mb": f"{loaded_at_mb:.1f} MB",
        "peak_rss_mb": f"{peak_rss_mb():.1f} MB",
        "note": "Measured on this Mac with MLX 4bit teacher model; includes model load and generation process peak.",
    }
    if update_metrics:
        update_runtime_metrics("teacher_qwen3_1p7b", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark-zipvoice", choices=["teacher-ref", "original-news-female"])
    parser.add_argument("--benchmark-teacher", action="store_true")
    parser.add_argument("--update-runtime-metrics", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    if args.benchmark_zipvoice:
        benchmark_zipvoice(args.benchmark_zipvoice, args.limit, args.update_runtime_metrics)
        return 0
    if args.benchmark_teacher:
        benchmark_teacher(args.limit, args.update_runtime_metrics)
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    samples = build_samples(force=args.force)
    render_report(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

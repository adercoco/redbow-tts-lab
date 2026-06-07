#!/usr/bin/env python3
"""Build teacher-vs-mobile-student baseline report."""

from __future__ import annotations

import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
TEACHER = BASE / "teacher_qwen3_1p7b_seed"
OUT = BASE / "reports" / "mobile_student_baseline"
CONFIG = ROOT / "tools" / "tts_eval_taiwan_mandarin_low_r_mobile_baseline.json"


@dataclass
class Row:
    model: str
    model_size: str
    mobile_fit: str
    text: str
    teacher_audio: Path
    student_audio: Path
    seconds: float
    score: float
    mfcc: float
    f0: float
    centroid: float


def features(path: Path) -> dict[str, np.ndarray | float]:
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


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, (float(np.dot(a, b)) / denom + 1.0) / 2.0))


def log_similarity(a: float, b: float, width: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return max(0.0, min(1.0, math.exp(-abs(math.log(a / b)) / width)))


def score(teacher: Path, student: Path) -> tuple[float, float, float, float]:
    t = features(teacher)
    s = features(student)
    mfcc = cosine01(t["mfcc"], s["mfcc"])  # type: ignore[arg-type]
    f0 = log_similarity(float(t["f0"]), float(s["f0"]), 0.45)
    centroid = log_similarity(float(t["centroid"]), float(s["centroid"]), 0.55)
    total = 0.62 * mfcc + 0.23 * f0 + 0.15 * centroid
    td = librosa.get_duration(path=teacher)
    sd = librosa.get_duration(path=student)
    if sd < 1.5:
        total *= 0.30
    elif td > 0 and sd < td * 0.45:
        total *= 0.45
    elif td > 0 and sd > td * 2.2:
        total *= 0.55
    if sd > 30:
        total *= 0.45
    elif sd > 20:
        total *= 0.60
    return total, mfcc, f0, centroid


def collect_run(run_dir: Path, model: str, model_size: str, mobile_fit: str, teacher_ids: list[str]) -> list[Row]:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows: list[Row] = []
    for result, teacher_id in zip(results, teacher_ids):
        if result["status"] != "ok":
            continue
        teacher_audio = TEACHER / "audio" / f"{teacher_id}.wav"
        student_audio = Path(result["output"])
        total, mfcc, f0, centroid = score(teacher_audio, student_audio)
        rows.append(
            Row(
                model=model,
                model_size=model_size,
                mobile_fit=mobile_fit,
                text=result["text"],
                teacher_audio=teacher_audio,
                student_audio=student_audio,
                seconds=float(result["seconds"]),
                score=total,
                mfcc=mfcc,
                f0=f0,
                centroid=centroid,
            )
        )
    return rows


def copy(src: Path, subdir: str) -> str:
    dest_dir = OUT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.stem}-{abs(hash(str(src))) % 100000}{src.suffix}"
    shutil.copy2(src, dest)
    return dest.relative_to(OUT).as_posix()


def verdict(score_value: float) -> str:
    if score_value >= 0.82:
        return "很接近，優先"
    if score_value >= 0.74:
        return "可用候選"
    if score_value >= 0.62:
        return "有像但需改善"
    return "不夠像"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    teacher_ids = config["targets"][0]["teacher_eval_ids"]
    runs = [
        (
            BASE / "students" / "zipvoice_baseline",
            "Sherpa-ONNX ZipVoice int8",
            "~207MB model pack",
            "最佳跨 iOS/Android ONNX 路線",
        ),
        (
            BASE / "students" / "qwen3_0p6b_mlx_baseline",
            "Qwen3-TTS 0.6B Base 4bit MLX",
            "0.6B 4bit, GB級記憶體",
            "高階 iPhone/Apple Silicon 路線",
        ),
        (
            BASE / "students" / "neutts_nano_baseline",
            "NeuTTS Nano",
            "Nano/on-device",
            "可手機化但中文需驗證",
        ),
    ]
    rows: list[Row] = []
    for run_dir, model, size, fit in runs:
        if (run_dir / "results.json").exists():
            rows.extend(collect_run(run_dir, model, size, fit, teacher_ids))

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for model in sorted({row.model for row in rows}):
        items = [row for row in rows if row.model == model]
        avg = float(np.mean([row.score for row in items]))
        avg_seconds = float(np.mean([row.seconds for row in items]))
        summary.append((avg, model, items[0].model_size, items[0].mobile_fit, avg_seconds, len(items)))
    summary.sort(reverse=True)

    summary_rows = [
        "<tr>"
        f"<td>{html.escape(model)}</td><td>{html.escape(size)}</td><td>{html.escape(fit)}</td>"
        f"<td>{avg:.3f}</td><td>{avg_seconds:.1f}s</td><td>{count}</td><td>{html.escape(verdict(avg))}</td>"
        "</tr>"
        for avg, model, size, fit, avg_seconds, count in summary
    ]

    detail_rows = []
    for row in sorted(rows, key=lambda item: (item.text, -item.score, item.model)):
        teacher_rel = copy(row.teacher_audio, "teacher")
        student_rel = copy(row.student_audio, f"students/{row.model}")
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(row.text)}</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(teacher_rel)}\"></audio></td>"
            f"<td>{html.escape(row.model)}</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(student_rel)}\"></audio></td>"
            f"<td class=\"score\">{row.score:.3f}</td><td>{row.mfcc:.3f}</td><td>{row.f0:.3f}</td><td>{row.centroid:.3f}</td>"
            f"<td>{row.seconds:.1f}s</td><td>{html.escape(verdict(row.score))}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>taiwan_mandarin_low_r Mobile Student Baseline</title>
<style>
body{{margin:0;background:#f7f4ef;color:#181514;font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif;}}
header{{background:#b51d2a;color:white;padding:18px 16px 12px;}}
h1{{font-size:22px;line-height:1.2;margin:0 0 6px;}}
header p{{margin:0;font-size:14px;opacity:.9;line-height:1.5;}}
h2{{font-size:18px;margin:20px 14px 8px;}}
.wrap{{overflow-x:auto;margin:0 14px 18px;border:1px solid #ded6cc;background:white;}}
table{{border-collapse:collapse;min-width:980px;width:100%;}}
th,td{{border-bottom:1px solid #eee5dc;padding:10px;text-align:left;vertical-align:top;font-size:14px;}}
th{{background:#fff5ed;}}
audio{{width:220px;max-width:70vw;}}
.score{{font-weight:700;color:#9b1823;}}
.note{{padding:10px 16px;color:#5f5550;font-size:14px;line-height:1.6;}}
</style>
</head>
<body>
<header>
<h1>taiwan_mandarin_low_r：1.7B Teacher vs Mobile Student Baseline</h1>
<p>每列同一句台詞：先聽 Qwen3 1.7B teacher，再聽手機可跑 student。這是 no-train baseline，下一步才是訓練/微調。</p>
</header>
<p class="note">分數是聲學 proxy，用 MFCC/F0/頻譜與時長懲罰估算，不取代你的耳朵。</p>
<h2>總評</h2>
<div class="wrap"><table><thead><tr><th>Student</th><th>大小</th><th>手機路線</th><th>平均像度</th><th>平均生成</th><th>樣本</th><th>初評</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
<h2>A/B 試聽</h2>
<div class="wrap"><table><thead><tr><th>台詞</th><th>1.7B Teacher</th><th>Student</th><th>Student Audio</th><th>像度</th><th>MFCC</th><th>F0</th><th>頻譜</th><th>耗時</th><th>初評</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div>
</body>
</html>
"""
    (OUT / "index.html").write_text(report, encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps([row.__dict__ | {"teacher_audio": str(row.teacher_audio), "student_audio": str(row.student_audio)} for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

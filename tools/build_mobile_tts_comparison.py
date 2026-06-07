#!/usr/bin/env python3
"""Build a phone-friendly TTS clone comparison report."""

from __future__ import annotations

import html
import json
import math
import os
import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools" / "tts_eval_haibara_authorized_clone.json"
REPORTS_DIR = ROOT / "tts_mobile_reports"


@dataclass
class Sample:
    model: str
    mobile_fit: str
    target_name: str
    goal: str
    reference_audio: Path
    text: str
    output: Path
    seconds: float
    score: float
    mfcc_score: float
    f0_score: float
    centroid_score: float


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(path: Path) -> dict[str, np.ndarray | float]:
    y, sr = librosa.load(path, sr=24000, mono=True)
    if len(y) == 0:
        raise RuntimeError(f"empty audio: {path}")
    y, _ = librosa.effects.trim(y, top_db=35)
    if len(y) < sr // 2:
        y, sr = librosa.load(path, sr=24000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_vec = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))

    try:
        f0 = librosa.yin(y, fmin=80, fmax=600, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0_median = float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        f0_median = 0.0

    return {
        "mfcc": mfcc_vec,
        "centroid": centroid_mean,
        "f0": f0_median,
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


def similarity(reference: Path, output: Path) -> tuple[float, float, float, float]:
    ref = load_features(reference)
    out = load_features(output)
    mfcc = cosine01(ref["mfcc"], out["mfcc"])  # type: ignore[arg-type]
    f0 = log_similarity(float(ref["f0"]), float(out["f0"]), 0.45)
    centroid = log_similarity(float(ref["centroid"]), float(out["centroid"]), 0.55)
    total = 0.62 * mfcc + 0.23 * f0 + 0.15 * centroid
    reference_duration = librosa.get_duration(path=reference)
    output_duration = librosa.get_duration(path=output)
    if output_duration < 1.5:
        total *= 0.30
    elif reference_duration > 0 and output_duration < reference_duration * 0.45:
        total *= 0.45
    elif reference_duration > 0 and output_duration > reference_duration * 2.2:
        total *= 0.55
    if output_duration > 30:
        total *= 0.45
    elif output_duration > 20:
        total *= 0.60
    return total, mfcc, f0, centroid


def collect(run_dir: Path, model: str, mobile_fit: str, config_path: Path) -> list[Sample]:
    config = load_json(config_path)
    by_id = {target["id"]: target for target in config["targets"]}  # type: ignore[index]
    results = load_json(run_dir / "results.json")
    samples: list[Sample] = []
    for row in results:  # type: ignore[union-attr]
        if row["status"] != "ok":
            continue
        target = by_id[row["target_id"]]
        reference = Path(target["reference_audio"])
        output = Path(row["output"])
        total, mfcc, f0, centroid = similarity(reference, output)
        samples.append(
            Sample(
                model=model,
                mobile_fit=mobile_fit,
                target_name=row["target_name"],
                goal=row["goal"],
                reference_audio=reference,
                text=row["text"],
                output=output,
                seconds=float(row["seconds"]),
                score=total,
                mfcc_score=mfcc,
                f0_score=f0,
                centroid_score=centroid,
            )
        )
    return samples


def verdict(score: float) -> str:
    if score >= 0.78:
        return "像度強，值得進 app 試"
    if score >= 0.70:
        return "有像，值得保留比較"
    if score >= 0.62:
        return "部分像，需聽感確認"
    return "像度偏弱"


def copy_audio(src: Path, report_dir: Path, subdir: str) -> str:
    dest_dir = report_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.stem}-{abs(hash(str(src))) % 100000}{src.suffix}"
    shutil.copy2(src, dest)
    return dest.relative_to(report_dir).as_posix()


def summarize(samples: list[Sample]) -> list[dict[str, object]]:
    rows = []
    for model in sorted({s.model for s in samples}):
        model_samples = [s for s in samples if s.model == model]
        rows.append(
            {
                "model": model,
                "mobile_fit": model_samples[0].mobile_fit,
                "avg_score": float(np.mean([s.score for s in model_samples])),
                "avg_seconds": float(np.mean([s.seconds for s in model_samples])),
                "count": len(model_samples),
            }
        )
    return sorted(rows, key=lambda x: float(x["avg_score"]), reverse=True)


def write_report(samples: list[Sample], report_dir: Path) -> None:
    summary_rows = []
    for row in summarize(samples):
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['mobile_fit']))}</td>"
            f"<td>{float(row['avg_score']):.3f}</td>"
            f"<td>{float(row['avg_seconds']):.1f}s</td>"
            f"<td>{row['count']}</td>"
            f"<td>{html.escape(verdict(float(row['avg_score'])))}</td>"
            "</tr>"
        )

    detail_rows = []
    for sample in sorted(samples, key=lambda s: (s.target_name, -s.score, s.model, s.text)):
        ref_rel = copy_audio(sample.reference_audio, report_dir, "references")
        out_rel = copy_audio(sample.output, report_dir, f"outputs/{sample.model}")
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(sample.target_name)}</td>"
            f"<td>{html.escape(sample.model)}</td>"
            f"<td>{html.escape(sample.mobile_fit)}</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(ref_rel)}\"></audio></td>"
            f"<td>{html.escape(sample.text)}</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(out_rel)}\"></audio></td>"
            f"<td class=\"score\">{sample.score:.3f}</td>"
            f"<td>{sample.mfcc_score:.3f}</td>"
            f"<td>{sample.f0_score:.3f}</td>"
            f"<td>{sample.centroid_score:.3f}</td>"
            f"<td>{sample.seconds:.1f}s</td>"
            f"<td>{html.escape(verdict(sample.score))}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>灰原 Clone TTS 手機試聽比較</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", sans-serif; background: #f7f4ef; color: #181514; }}
    header {{ padding: 18px 16px 8px; background: #b51d2a; color: white; }}
    h1 {{ font-size: 22px; margin: 0 0 6px; line-height: 1.2; }}
    h2 {{ font-size: 18px; margin: 24px 16px 10px; }}
    p {{ margin: 0; line-height: 1.45; }}
    .hint {{ opacity: .88; font-size: 14px; }}
    .table-wrap {{ overflow-x: auto; margin: 0 12px 22px; border: 1px solid #ddd4c9; background: white; }}
    table {{ border-collapse: collapse; min-width: 960px; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ece4dc; padding: 10px; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #fff5ed; position: sticky; top: 0; z-index: 1; }}
    audio {{ width: 220px; max-width: 70vw; }}
    .score {{ font-weight: 700; color: #9b1823; }}
    .note {{ margin: 0 16px 18px; font-size: 14px; color: #5f5550; line-height: 1.6; }}
    @media (max-width: 700px) {{
      header {{ padding-top: 14px; }}
      h1 {{ font-size: 20px; }}
      th, td {{ font-size: 13px; padding: 8px; }}
      audio {{ width: 190px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>灰原 Clone TTS 手機試聽比較</h1>
    <p class="hint">授權 reference 音檔測試。分數是聲學 proxy，不取代你的耳朵；它用 MFCC / F0 / 頻譜中心粗估音色接近度。</p>
  </header>

  <h2>模型總評</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>模型</th><th>手機離線可行性</th><th>平均像度</th><th>平均生成耗時</th><th>樣本</th><th>初評</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
  </div>

  <p class="note">閱讀方式：先聽每列 Reference，再聽輸出。A 是 10 秒乾淨片段，B 是 8 秒短片段，C 是完整 30 秒片段。C 對 F5 會被裁短，對 ZipVoice 則可能生成較慢且節奏飄。</p>

  <h2>逐句試聽</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Reference 組</th><th>模型</th><th>手機可行性</th><th>Reference</th><th>台詞</th><th>輸出</th><th>像度</th><th>MFCC</th><th>F0</th><th>頻譜</th><th>耗時</th><th>我的初評</th></tr>
      </thead>
      <tbody>{''.join(detail_rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    (report_dir / "index.html").write_text(report, encoding="utf-8")
    data = [
        {
            "model": s.model,
            "mobile_fit": s.mobile_fit,
            "target_name": s.target_name,
            "text": s.text,
            "output": str(s.output),
            "reference_audio": str(s.reference_audio),
            "seconds": s.seconds,
            "score": s.score,
            "mfcc_score": s.mfcc_score,
            "f0_score": s.f0_score,
            "centroid_score": s.centroid_score,
            "verdict": verdict(s.score),
        }
        for s in samples
    ]
    (report_dir / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label", default="haibara_clone")
    parser.add_argument("--include-f5", action="store_true")
    parser.add_argument("--f5-run", type=Path, default=ROOT / "tts_runs" / "20260525-093801" / "haibara_authorized_clone-f5-cli")
    parser.add_argument("--zipvoice-run", type=Path, default=ROOT / "tts_runs" / "20260525-094640" / "haibara_authorized_clone-sherpa-zipvoice")
    parser.add_argument("--pocket-run", type=Path, default=ROOT / "tts_runs" / "20260525-094946" / "haibara_authorized_clone-sherpa-pocket")
    parser.add_argument("--qwen3-run", type=Path, default=None)
    parser.add_argument("--neutts-run", type=Path, default=None)
    args = parser.parse_args()

    samples = []
    if args.include_f5:
        samples.extend(collect(args.f5_run, "F5-TTS v1 Base", "桌機品質候選；未列入手機-only 主評", args.config))
    samples.extend(collect(args.zipvoice_run, "Sherpa-ONNX ZipVoice int8", "手機主候選；ONNX/int8，跨 iOS/Android 路線清楚", args.config))
    samples.extend(collect(args.pocket_run, "Sherpa-ONNX PocketTTS int8", "手機技術候選；中文同句輸出不穩，保留做負例", args.config))
    if args.qwen3_run is not None:
        samples.extend(collect(args.qwen3_run, "Qwen3-TTS 0.6B Base 4bit MLX", "高品質手機候選；iOS/Apple Silicon MLX 路線，記憶體需求較高", args.config))
    if args.neutts_run is not None:
        samples.extend(collect(args.neutts_run, "NeuTTS Nano", "手機/嵌入式候選；官方 on-device/GGUF，中文不是主力語言", args.config))

    report_dir = REPORTS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_report(samples, report_dir)
    print(report_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

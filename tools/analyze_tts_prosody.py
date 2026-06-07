#!/usr/bin/env python3
"""Analyze TTS prosody flatness for Taiwan Mandarin voice work."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


BASE = Path("/Users/ader/Documents/App/distillation/taiwan_mandarin_low_r")
REPORT = BASE / "reports" / "prosody_flatness_v1"
ZIPVOICE_THREE_WAY = BASE / "reports" / "zipvoice_three_way" / "audio"


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    pattern: Path


GROUPS = [
    Group("teacher", "Qwen3 teacher", ZIPVOICE_THREE_WAY / "teacher_qwen3_1p7b"),
    Group("zipvoice_teacher_ref", "ZipVoice student, teacher reference", ZIPVOICE_THREE_WAY / "zipvoice_teacher_ref"),
    Group("zipvoice_original", "ZipVoice original female", ZIPVOICE_THREE_WAY / "zipvoice_original_news_female"),
]


def hz_to_semitone(values: np.ndarray) -> np.ndarray:
    values = values[values > 0]
    if len(values) == 0:
        return values
    median = np.nanmedian(values)
    return 12 * np.log2(values / median)


def longest_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for item in mask:
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def analyze_file(path: Path) -> dict[str, float | str]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        sr = 24000

    hop_length = 256
    frame_length = 1024
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) if np.max(rms) > 0 else np.zeros_like(rms)
    voiced_energy = rms_db > -35
    silence_ratio = float(np.mean(~voiced_energy)) if len(voiced_energy) else 0.0
    longest_silence_s = longest_run(~voiced_energy) * hop_length / sr

    try:
        f0 = librosa.yin(
            audio,
            fmin=90,
            fmax=420,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )
    except Exception:
        f0 = np.array([])

    if len(f0):
        f0 = f0[np.isfinite(f0)]
    f0 = f0[(f0 >= 90) & (f0 <= 420)]
    semi = hz_to_semitone(f0)
    if len(semi):
        f0_median = float(np.median(f0))
        f0_std_st = float(np.std(semi))
        f0_range_st = float(np.percentile(semi, 95) - np.percentile(semi, 5))
        f0_iqr_st = float(np.percentile(semi, 75) - np.percentile(semi, 25))
        f0_movement_st = float(np.mean(np.abs(np.diff(semi)))) if len(semi) > 1 else 0.0
    else:
        f0_median = 0.0
        f0_std_st = 0.0
        f0_range_st = 0.0
        f0_iqr_st = 0.0
        f0_movement_st = 0.0

    energy_std_db = float(np.std(rms_db)) if len(rms_db) else 0.0
    energy_range_db = (
        float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)) if len(rms_db) else 0.0
    )

    # Lower means flatter. This intentionally weights F0 range more than energy.
    flatness_index = (
        0.42 * f0_range_st
        + 0.22 * f0_std_st
        + 0.18 * f0_movement_st
        + 0.10 * (energy_range_db / 6)
        + 0.08 * (longest_silence_s * 4)
    )

    return {
        "file": str(path),
        "name": path.stem,
        "duration_s": duration,
        "f0_median_hz": f0_median,
        "f0_std_st": f0_std_st,
        "f0_range_st": f0_range_st,
        "f0_iqr_st": f0_iqr_st,
        "f0_movement_st": f0_movement_st,
        "energy_std_db": energy_std_db,
        "energy_range_db": energy_range_db,
        "silence_ratio": silence_ratio,
        "longest_silence_s": longest_silence_s,
        "flatness_index": flatness_index,
    }


def mean_metric(rows: list[dict[str, float | str]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.mean(values) if values else 0.0


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def bar(value: float, max_value: float) -> str:
    width = 100 * value / max_value if max_value else 0
    return f"<span class='bar'><span style='width:{width:.1f}%'></span></span>"


def build_html(summary: list[dict[str, object]], rows_by_group: dict[str, list[dict[str, object]]]) -> str:
    max_flat = max(float(item["flatness_index"]) for item in summary) if summary else 1
    max_range = max(float(item["f0_range_st"]) for item in summary) if summary else 1

    cards = []
    for item in summary:
        key = str(item["key"])
        label = str(item["label"])
        flatness = float(item["flatness_index"])
        f0_range = float(item["f0_range_st"])
        cards.append(
            f"""
            <article class="card">
              <h3>{label}</h3>
              <div class="score">{fmt(flatness)} <small>flatness index</small></div>
              {bar(flatness, max_flat)}
              <div class="spec"><span>F0 5-95% 範圍</span><b>{fmt(f0_range)} st</b></div>
              <div class="spec"><span>F0 frame movement</span><b>{fmt(float(item["f0_movement_st"]), 3)} st</b></div>
              <div class="spec"><span>能量範圍</span><b>{fmt(float(item["energy_range_db"]))} dB</b></div>
              <div class="spec"><span>停頓比例</span><b>{fmt(float(item["silence_ratio"]) * 100, 1)}%</b></div>
              <div class="spec"><span>樣本數</span><b>{len(rows_by_group[key])}</b></div>
            </article>
            """
        )

    range_rows = []
    for item in summary:
        range_rows.append(
            f"""
            <tr>
              <td>{item["label"]}</td>
              <td>{fmt(float(item["f0_range_st"]))} st<br>{bar(float(item["f0_range_st"]), max_range)}</td>
              <td>{fmt(float(item["flatness_index"]))}</td>
              <td>{fmt(float(item["energy_range_db"]))} dB</td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台灣聲線 Prosody 平坦度分析</title>
  <style>
    body {{ margin:0; background:#f7f3ed; color:#171514; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; line-height:1.58; }}
    main {{ width:min(100%,430px); margin:0 auto; padding:18px 14px 48px; }}
    .eyebrow {{ color:#a52232; font-size:13px; font-weight:800; margin-bottom:6px; }}
    h1 {{ margin:0; font-size:28px; line-height:1.18; letter-spacing:0; }}
    h2 {{ margin:24px 0 10px; font-size:20px; letter-spacing:0; }}
    h3 {{ margin:0 0 8px; font-size:19px; letter-spacing:0; }}
    p {{ margin:8px 0 12px; color:#665e57; font-size:15px; }}
    .hero {{ background:#211816; color:white; border-radius:8px; padding:16px; margin:14px 0; }}
    .hero p {{ color:rgba(255,255,255,.76); margin:0; }}
    .big {{ display:block; font-size:29px; line-height:1.1; font-weight:850; margin:6px 0; letter-spacing:0; }}
    .card {{ background:#fffdfa; border:1px solid #ded7ce; border-radius:8px; padding:14px; margin:10px 0; }}
    .score {{ color:#a52232; font-size:28px; font-weight:850; line-height:1.1; }}
    .score small {{ display:block; color:#665e57; font-size:12px; font-weight:650; }}
    .bar {{ display:block; width:100%; height:9px; background:#eee8df; border-radius:999px; overflow:hidden; margin:10px 0; }}
    .bar span {{ display:block; height:100%; background:#a52232; border-radius:999px; }}
    .spec {{ display:grid; grid-template-columns:1fr auto; gap:8px; border-top:1px solid #eee8df; padding-top:7px; margin-top:7px; font-size:14px; }}
    .spec span {{ color:#665e57; }}
    .note {{ background:#efe7dd; border:1px solid #ded7ce; border-radius:8px; padding:12px; color:#4f4740; font-size:14px; }}
    table {{ width:100%; border-collapse:collapse; background:#fffdfa; border:1px solid #ded7ce; border-radius:8px; overflow:hidden; }}
    th, td {{ padding:10px; border-bottom:1px solid #eee8df; text-align:left; vertical-align:top; font-size:14px; }}
    th {{ background:#f1e9df; }}
    code {{ background:#eee8df; padding:1px 5px; border-radius:5px; font-size:13px; }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Prosody 分析 · F0 / energy / pause</div>
    <h1>台灣女生聲線是不是要更平？</h1>
    <p>這份用現有 18 句三方音檔分析：老師、ZipVoice teacher-ref 學生、ZipVoice 原本女聲。</p>

    <section class="hero">
      <p>初步答案</p>
      <span class="big">可以分析，而且「更平」大概率是方向之一</span>
      <p>但不是完全無起伏，而是減少誇張 F0 跳動、少一點舞台腔，保留句尾自然下收和輕微停頓。</p>
    </section>

    <section class="note">
      指標說明：flatness index 越低代表越平。它綜合 F0 音高範圍、逐幀音高移動、能量起伏和長停頓。這不是聽感最終判決，但很適合拿來做蒸餾資料篩選。
    </section>

    <h2>平均結果</h2>
    {''.join(cards)}

    <h2>音高起伏</h2>
    <table>
      <thead><tr><th>聲音</th><th>F0 範圍</th><th>平坦指標</th><th>能量範圍</th></tr></thead>
      <tbody>{''.join(range_rows)}</tbody>
    </table>

    <h2>蒸餾建議</h2>
    <section class="note">
      早期訓練可以刻意偏平：優先挑 F0 範圍較窄、能量起伏較小、停頓自然的 teacher 句子。不要把所有情緒都壓平，否則會像機器人；台灣感比較像「平穩、尾音自然、不過度卷舌、不主播腔」。這批資料裡 ZipVoice teacher-ref 比 ZipVoice 原聲更平，也更像 teacher；但 teacher 本身仍有細膩起伏，所以後期要加回自然句尾和情緒。
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPORT)
    args = parser.parse_args()

    rows_by_group: dict[str, list[dict[str, object]]] = {}
    summary: list[dict[str, object]] = []
    for group in GROUPS:
        files = sorted(group.pattern.glob("*.wav"))
        rows = [analyze_file(path) for path in files]
        rows_by_group[group.key] = rows
        item: dict[str, object] = {"key": group.key, "label": group.label}
        for metric in [
            "duration_s",
            "f0_median_hz",
            "f0_std_st",
            "f0_range_st",
            "f0_iqr_st",
            "f0_movement_st",
            "energy_std_db",
            "energy_range_db",
            "silence_ratio",
            "longest_silence_s",
            "flatness_index",
        ]:
            item[metric] = mean_metric(rows, metric)
        summary.append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prosody_metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows_by_group}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(
        build_html(summary, rows_by_group),
        encoding="utf-8",
    )
    print(args.out_dir / "index.html")
    for item in summary:
        print(
            item["label"],
            "flatness=",
            fmt(float(item["flatness_index"])),
            "f0_range_st=",
            fmt(float(item["f0_range_st"])),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

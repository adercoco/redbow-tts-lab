#!/usr/bin/env python3
"""Build prosody-filtered teacher manifests for Taiwan Mandarin TTS distillation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from analyze_tts_prosody import analyze_file


BASE = Path("/Users/ader/Documents/App/distillation/taiwan_mandarin_low_r")
TEACHER_MANIFEST = BASE / "teacher_qwen3_1p7b_distill_v1" / "manifest.json"
OUTPUT_DIR = BASE / "datasets" / "prosody_filtered_teacher_v1"
REPORT_DIR = BASE / "reports" / "prosody_flatness_v1"
AUDIO_PREVIEW_DIR = REPORT_DIR / "phase1_audio"


def read_teacher_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    usable: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        audio = Path(str(row["audio"]))
        if not audio.is_absolute():
            audio = Path("/Users/ader/Documents/App") / audio
        if audio.exists():
            item = dict(row)
            item["audio"] = str(audio)
            usable.append(item)
    return usable


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distill_score(row: dict[str, Any]) -> float:
    duration = float(row["duration_s"])
    silence_ratio = float(row["silence_ratio"])
    longest_silence = float(row["longest_silence_s"])
    f0_median = float(row["f0_median_hz"])

    duration_penalty = 0.0
    if duration < 1.35:
        duration_penalty += (1.35 - duration) * 4.0
    if duration > 7.5:
        duration_penalty += (duration - 7.5) * 0.8

    f0_penalty = 0.0
    if f0_median < 145:
        f0_penalty += (145 - f0_median) / 40
    if f0_median > 330:
        f0_penalty += (f0_median - 330) / 40

    return (
        float(row["flatness_index"])
        + 4.0 * silence_ratio
        + 1.8 * longest_silence
        + duration_penalty
        + f0_penalty
    )


def is_clean(row: dict[str, Any]) -> bool:
    return (
        1.0 <= float(row["duration_s"]) <= 15.0
        and float(row["silence_ratio"]) <= 0.65
        and float(row["longest_silence_s"]) <= 2.0
        and 90 <= float(row["f0_median_hz"]) <= 420
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "id",
        "phase",
        "category",
        "text",
        "audio",
        "duration_s",
        "flatness_index",
        "distill_score",
        "f0_median_hz",
        "f0_range_st",
        "f0_movement_st",
        "energy_range_db",
        "silence_ratio",
        "longest_silence_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({field: row.get(field, "") for field in fields} | {"rank": index})


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.mean(values) if values else 0.0


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def copy_previews(rows: list[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    AUDIO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    previews = []
    for row in rows[:count]:
        src = Path(str(row["audio"]))
        dst = AUDIO_PREVIEW_DIR / src.name
        shutil.copy2(src, dst)
        preview = dict(row)
        preview["preview_audio"] = f"phase1_audio/{dst.name}"
        previews.append(preview)
    return previews


def build_html(
    all_rows: list[dict[str, Any]],
    phase1: list[dict[str, Any]],
    phase2: list[dict[str, Any]],
    previews: list[dict[str, Any]],
) -> str:
    phase1_flat = mean(phase1, "flatness_index")
    phase2_flat = mean(phase2, "flatness_index")
    all_flat = mean(all_rows, "flatness_index")
    p20 = percentile([float(row["flatness_index"]) for row in all_rows], 0.20)
    p50 = percentile([float(row["flatness_index"]) for row in all_rows], 0.50)

    preview_cards = []
    for row in previews:
        preview_cards.append(
            f"""
            <article class="sample">
              <div class="tag">{row["id"]} · {row.get("category", "")}</div>
              <p>{row["text"]}</p>
              <audio controls preload="none" src="{row["preview_audio"]}"></audio>
              <dl>
                <div><dt>flatness</dt><dd>{fmt(float(row["flatness_index"]))}</dd></div>
                <div><dt>F0 range</dt><dd>{fmt(float(row["f0_range_st"]))} st</dd></div>
                <div><dt>pause</dt><dd>{fmt(float(row["silence_ratio"]) * 100, 1)}%</dd></div>
              </dl>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>完整蒸餾 Phase 1：平穩核心資料</title>
  <style>
    body {{ margin:0; background:#f7f3ed; color:#181514; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; line-height:1.58; }}
    main {{ width:min(100%,430px); margin:0 auto; padding:18px 14px 52px; box-sizing:border-box; }}
    .eyebrow {{ color:#9a2233; font-size:13px; font-weight:800; margin-bottom:6px; }}
    h1 {{ margin:0; font-size:28px; line-height:1.16; letter-spacing:0; }}
    h2 {{ margin:24px 0 10px; font-size:20px; letter-spacing:0; }}
    p {{ margin:8px 0 12px; color:#5f5750; font-size:15px; }}
    .hero {{ background:#201817; color:white; border-radius:8px; padding:16px; margin:14px 0; }}
    .hero p {{ color:rgba(255,255,255,.76); margin:0; }}
    .big {{ display:block; font-size:30px; line-height:1.08; font-weight:850; margin:7px 0; letter-spacing:0; }}
    .grid {{ display:grid; grid-template-columns:1fr; gap:10px; margin:12px 0; }}
    .metric, .sample, .note {{ background:#fffdfa; border:1px solid #ded7ce; border-radius:8px; padding:14px; }}
    .metric b {{ display:block; color:#9a2233; font-size:27px; line-height:1.1; }}
    .metric span {{ color:#665e57; font-size:13px; font-weight:700; }}
    .note {{ background:#efe7dd; color:#4f4740; font-size:14px; }}
    .sample p {{ color:#241f1c; font-size:16px; }}
    .tag {{ color:#9a2233; font-size:12px; font-weight:800; }}
    audio {{ width:100%; margin:4px 0 8px; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin:0; }}
    dl div {{ background:#f3ede5; border-radius:8px; padding:8px; min-width:0; }}
    dt {{ color:#6d6259; font-size:11px; font-weight:750; }}
    dd {{ margin:1px 0 0; color:#181514; font-size:14px; font-weight:820; }}
    code {{ background:#eee8df; padding:1px 5px; border-radius:5px; font-size:13px; }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Full distillation · data curriculum</div>
    <h1>Phase 1 先學平穩台灣感，不先學誇張抑揚</h1>
    <p>從 1000 句 Qwen3 teacher 語料，用 F0、能量、停頓挑出早期訓練核心。這一步是完整蒸餾的資料課表，不是最終模型。</p>

    <section class="hero">
      <p>Phase 1 core</p>
      <span class="big">{len(phase1)} 句平穩核心</span>
      <p>先讓學生學到穩、柔、少舞台腔的底色，再用 Phase 2 加回自然句尾和一點情緒。</p>
    </section>

    <section class="grid">
      <div class="metric"><span>全老師語料平均 flatness</span><b>{fmt(all_flat)}</b></div>
      <div class="metric"><span>Phase 1 平穩核心平均 flatness</span><b>{fmt(phase1_flat)}</b></div>
      <div class="metric"><span>Phase 2 自然尾音平均 flatness</span><b>{fmt(phase2_flat)}</b></div>
    </section>

    <section class="note">
      flatness index 越低越平。這次 P20={fmt(p20)}、P50={fmt(p50)}；Phase 1 不是挑最低到死，而是同時避開太短、長停頓、F0 異常的句子。
    </section>

    <h2>已產出的訓練檔</h2>
    <section class="note">
      <code>datasets/prosody_filtered_teacher_v1/phase1_flat_core.jsonl</code><br>
      <code>datasets/prosody_filtered_teacher_v1/phase2_natural_tail.jsonl</code><br>
      <code>datasets/prosody_filtered_teacher_v1/manifest.json</code>
    </section>

    <h2>可聽樣本</h2>
    {''.join(preview_cards)}

    <h2>訓練策略</h2>
    <section class="note">
      完整蒸餾建議先用 Phase 1 跑 warmup，讓學生權重吸收穩定音色與較平的台灣日常語調；接著混入 Phase 2，避免聲音變成無表情。最後再用少量人工聽感評分校準「溫柔、聰明、真實」。
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-manifest", type=Path, default=TEACHER_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--phase1-count", type=int, default=300)
    parser.add_argument("--phase2-count", type=int, default=300)
    args = parser.parse_args()

    teacher_rows = read_teacher_manifest(args.teacher_manifest)
    analyzed: list[dict[str, Any]] = []
    for index, item in enumerate(teacher_rows, start=1):
        metrics = analyze_file(Path(str(item["audio"])))
        row = item | metrics
        row["distill_score"] = distill_score(row)
        row["clean_for_training"] = is_clean(row)
        analyzed.append(row)
        if index % 100 == 0:
            print(f"analyzed {index}/{len(teacher_rows)}")

    clean_rows = [row for row in analyzed if row["clean_for_training"]]
    ranked = sorted(clean_rows, key=lambda row: (float(row["distill_score"]), float(row["flatness_index"])))
    phase1 = [dict(row, phase="phase1_flat_core") for row in ranked[: args.phase1_count]]
    phase1_ids = {row["id"] for row in phase1}

    remaining = [row for row in clean_rows if row["id"] not in phase1_ids]
    median_flatness = percentile([float(row["flatness_index"]) for row in clean_rows], 0.50)
    phase2_ranked = sorted(
        remaining,
        key=lambda row: (
            abs(float(row["flatness_index"]) - median_flatness),
            float(row["longest_silence_s"]),
            float(row["distill_score"]),
        ),
    )
    phase2 = [dict(row, phase="phase2_natural_tail") for row in phase2_ranked[: args.phase2_count]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "phase1_flat_core.jsonl", phase1)
    write_csv(args.output_dir / "phase1_flat_core.csv", phase1)
    write_jsonl(args.output_dir / "phase2_natural_tail.jsonl", phase2)
    write_csv(args.output_dir / "phase2_natural_tail.csv", phase2)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_manifest": str(args.teacher_manifest),
                "total_teacher_rows": len(teacher_rows),
                "analyzed_rows": len(analyzed),
                "clean_training_rows": len(clean_rows),
                "phase1_count": len(phase1),
                "phase2_count": len(phase2),
                "selection": {
                    "phase1": "lowest distill_score among clean rows; score weights low flatness, short natural pauses, valid duration, plausible female F0",
                    "phase2": "medium-flat clean rows near median flatness to restore natural phrase movement",
                },
                "metrics": {
                    "all_flatness_mean": mean(analyzed, "flatness_index"),
                    "clean_flatness_mean": mean(clean_rows, "flatness_index"),
                    "phase1_flatness_mean": mean(phase1, "flatness_index"),
                    "phase2_flatness_mean": mean(phase2, "flatness_index"),
                    "phase1_f0_range_mean_st": mean(phase1, "f0_range_st"),
                    "phase2_f0_range_mean_st": mean(phase2, "f0_range_st"),
                    "phase1_silence_ratio_mean": mean(phase1, "silence_ratio"),
                    "phase2_silence_ratio_mean": mean(phase2, "silence_ratio"),
                },
                "files": {
                    "phase1_jsonl": str(args.output_dir / "phase1_flat_core.jsonl"),
                    "phase1_csv": str(args.output_dir / "phase1_flat_core.csv"),
                    "phase2_jsonl": str(args.output_dir / "phase2_natural_tail.jsonl"),
                    "phase2_csv": str(args.output_dir / "phase2_natural_tail.csv"),
                    "report": str(args.report_dir / "teacher_phase1.html"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    previews = copy_previews(phase1)
    (args.report_dir / "teacher_phase1.html").write_text(
        build_html(analyzed, phase1, phase2, previews),
        encoding="utf-8",
    )

    print(args.output_dir / "manifest.json")
    print(args.report_dir / "teacher_phase1.html")
    print(
        "phase1",
        len(phase1),
        "flatness",
        fmt(mean(phase1, "flatness_index")),
        "f0_range",
        fmt(mean(phase1, "f0_range_st")),
    )
    print(
        "phase2",
        len(phase2),
        "flatness",
        fmt(mean(phase2, "flatness_index")),
        "f0_range",
        fmt(mean(phase2, "f0_range_st")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

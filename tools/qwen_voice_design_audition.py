#!/usr/bin/env python3
"""Generate Qwen3 1.7B VoiceDesign teacher voice candidates."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qwen_teacher_voice_select" / "1.7b_voicedesign_taiwan_cute_girl"
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    instruct: str


CANDIDATES = [
    Candidate(
        "ntu_cute_natural",
        "台大女大学生 可爱自然",
        "台湾台北长大的女大学生，二十一岁，声音可爱但自然，不要娃娃音，普通话带一点台湾腔，语速中等，语气聪明、亲切、轻松。",
    ),
    Candidate(
        "ntu_soft_smart",
        "台大女大学生 温柔聪明",
        "年轻台湾女生，像台大文学院女大学生，声音柔和清亮，有书卷气，讲话温柔聪明，普通话有自然台湾口音，情绪克制但可爱。",
    ),
    Candidate(
        "taipei_daily_chat",
        "台北女生 日常聊天",
        "二十岁出头的台北女生，日常聊天口吻，声音甜美、清楚、自然，带轻微台湾国语腔，不夸张，不做作，像朋友传语音讯息。",
    ),
    Candidate(
        "cute_anchor_clean",
        "可爱女声 清楚播报",
        "年轻可爱的台湾女生，发音清楚，声音明亮干净，像校园广播主持人，语气友善，有一点撒娇感但保持自然。",
    ),
    Candidate(
        "quiet_senpai",
        "安静学姐 冷静可爱",
        "台湾女大学生学姐，声音偏轻、冷静、聪明，语气有一点害羞和可爱，讲话不急，普通话带台湾腔，适合温柔提醒别人。",
    ),
    Candidate(
        "sweet_but_real",
        "甜但真人感",
        "真实的台湾年轻女生，甜美但不是动画声，不要过度卖萌，音色干净，尾音自然上扬，像二十岁大学生在咖啡厅轻声说话。",
    ),
    Candidate(
        "anime_taiwan_girl",
        "动画感 台湾少女",
        "台湾少女感女声，年龄二十岁左右，声音可爱、轻快、明亮，带一点动画角色感，但仍然像真人，普通话带台湾腔。",
    ),
    Candidate(
        "mature_cute_college",
        "成熟一点 可爱大学生",
        "台湾大学女生，声音比少女成熟一点，清甜、稳定、有礼貌，像认真又可爱的助教，普通话自然、有台湾语感。",
    ),
    Candidate(
        "soft_whispery",
        "轻声温柔 可爱",
        "年轻台湾女生，声音轻柔、亲密、可爱，像小声讲话的语音讯息，语速慢一点，普通话有台湾腔，情绪温暖。",
    ),
    Candidate(
        "bright_energetic",
        "明亮活泼 校园感",
        "活泼的台湾女大学生，声音明亮、有元气、可爱但不尖，像社团活动主持人，讲话自然流畅，带台湾国语语感。",
    ),
]


TEXTS = [
    (
        "daily",
        "欸，你先不要紧张啦。我们一步一步来，把事情讲清楚就好了。",
    ),
    (
        "assistant",
        "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
    ),
    (
        "cute",
        "这个声音可以吗？我想要听起来像台湾女生，可爱一点，但是不要太夸张。",
    ),
]


def generate() -> list[dict[str, object]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID}...")
    model = load_model(MODEL_ID)
    rows: list[dict[str, object]] = []

    for candidate in CANDIDATES:
        candidate_dir = OUT_DIR / candidate.id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for text_id, text in TEXTS:
            started = perf_counter()
            output = candidate_dir / f"{text_id}.wav"
            try:
                generate_audio(
                    text=text,
                    model=model,
                    instruct=candidate.instruct,
                    lang_code="zh",
                    output_path=str(candidate_dir),
                    file_prefix=text_id,
                    audio_format="wav",
                    verbose=False,
                )
                generated = candidate_dir / f"{text_id}_000.wav"
                if not generated.exists():
                    matches = sorted(candidate_dir.glob(f"{text_id}*.wav"))
                    if not matches:
                        raise RuntimeError("no wav generated")
                    generated = matches[-1]
                if generated != output:
                    shutil.copy2(generated, output)
                status = "ok"
                error = ""
                output_text: str | None = str(output)
            except Exception as exc:
                status = "failed"
                error = str(exc)
                output_text = None
            seconds = perf_counter() - started
            print(candidate.id, text_id, status, f"{seconds:.1f}s", output_text or error)
            rows.append(
                {
                    "candidate_id": candidate.id,
                    "label": candidate.label,
                    "instruct": candidate.instruct,
                    "text_id": text_id,
                    "text": text,
                    "output": output_text,
                    "seconds": seconds,
                    "status": status,
                    "error": error,
                }
            )
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    by_candidate: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)

    sections = []
    for candidate_id, items in by_candidate.items():
        first = items[0]
        sections.append(
            f"<section class=\"candidate\"><h2>{html.escape(str(first['label']))}</h2>"
            f"<p class=\"id\">{html.escape(candidate_id)}</p>"
            f"<p class=\"prompt\">{html.escape(str(first['instruct']))}</p>"
        )
        for row in items:
            sections.append("<div class=\"sample\">")
            sections.append(
                f"<div class=\"meta\"><strong>{html.escape(str(row['text_id']))}</strong>"
                f"<span>{float(row['seconds']):.1f}s</span></div>"
            )
            sections.append(f"<p class=\"text\">{html.escape(str(row['text']))}</p>")
            if row["output"]:
                rel = Path(str(row["output"])).relative_to(OUT_DIR).as_posix()
                sections.append(f'<audio controls preload="metadata" src="{html.escape(rel)}"></audio>')
            else:
                sections.append(f"<p class=\"err\">{html.escape(str(row['error']))}</p>")
            sections.append("</div>")
        sections.append("</section>")

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3 1.7B 台湾可爱女生 VoiceDesign</title>
<style>
body{{margin:0;background:#f7f4ef;color:#181514;font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif;}}
header{{background:#b51d2a;color:white;padding:18px 16px 12px;}}
h1{{font-size:22px;line-height:1.2;margin:0 0 6px;}}
header p{{margin:0;font-size:14px;opacity:.9;line-height:1.5;}}
main{{padding:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;}}
.candidate{{background:white;border:1px solid #ded6cc;padding:12px;}}
h2{{font-size:18px;margin:0 0 2px;}}
.id{{font-size:12px;color:#9b1823;margin:0 0 8px;}}
.prompt{{font-size:13px;color:#5f5550;line-height:1.55;margin:0 0 10px;}}
.sample{{border-top:1px solid #eee5dc;padding-top:10px;margin-top:10px;}}
.meta{{display:flex;justify-content:space-between;gap:8px;color:#8a1c25;font-size:13px;}}
.text{{font-size:14px;line-height:1.45;margin:8px 0;}}
audio{{width:100%;}}
.err{{color:#a00;}}
.note{{padding:10px 16px;color:#5f5550;font-size:14px;line-height:1.6;}}
</style>
</head>
<body>
<header>
<h1>Qwen3 1.7B VoiceDesign：台湾可爱女生候选</h1>
<p>用简中台词测试台湾女生感。请挑 1 到 3 个最像「台灣人中文 + 可愛女生」的 teacher 声音。</p>
</header>
<p class="note">每组 3 句：日常安抚、冷静提醒、直接说明想要的声音。下一步会用你选的 teacher 产训练资料，再蒸馏小模型。</p>
<main>{''.join(sections)}</main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(report, encoding="utf-8")
    (OUT_DIR / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR / "index.html")


def main() -> int:
    rows = generate()
    write_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

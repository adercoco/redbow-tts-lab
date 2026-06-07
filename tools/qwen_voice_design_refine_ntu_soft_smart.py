#!/usr/bin/env python3
"""Refine the selected Qwen3 VoiceDesign teacher prompt around ntu_soft_smart."""

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
OUT_DIR = ROOT / "qwen_teacher_voice_select" / "1.7b_voicedesign_ntu_soft_smart_refine"
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"


@dataclass(frozen=True)
class PromptVariant:
    id: str
    label: str
    instruct: str


PROMPTS = [
    PromptVariant(
        "base_selected",
        "原版冠军",
        "年轻台湾女生，像台大文学院女大学生，声音柔和清亮，有书卷气，讲话温柔聪明，普通话有自然台湾口音，情绪克制但可爱。",
    ),
    PromptVariant(
        "taiwan_tail_no_erhua",
        "更台湾尾音 少卷舌",
        "年轻台湾女生，像台大文学院女大学生，声音柔和清亮，有书卷气，讲话温柔聪明。普通话带自然台湾国语口音，尾音轻轻上扬，少卷舌，不要儿化音，不要大陆播音腔，情绪克制但可爱。",
    ),
    PromptVariant(
        "taipei_campus_soft",
        "台北校园自然",
        "二十一岁台北女大学生，像台大文学院学生，声音柔和清亮、自然亲切。说普通话时有台湾国语语感，平翘舌不要太明显，尾音自然、有一点可爱，不要像新闻主播。",
    ),
    PromptVariant(
        "taiwan_mandarin_low_r",
        "台湾国语低卷舌",
        "台湾年轻女生，中文普通话带台湾国语口音，卷舌很轻，儿化音几乎没有，语尾自然柔软。声音清亮温柔，有书卷气，像聪明的台大女生，克制、可爱、真实。",
    ),
    PromptVariant(
        "cute_but_not_mainland",
        "可爱但非大陆腔",
        "年轻台湾女大学生，声音温柔、聪明、清亮、可爱但不装。请避免大陆普通话播音腔，避免明显卷舌和儿化音，用自然台湾人说中文的感觉，尾音轻柔上扬。",
    ),
    PromptVariant(
        "literature_senpai",
        "文学院学姐感",
        "台大文学院女大学生学姐，声音柔和清亮，讲话聪明温柔，有一点害羞的可爱。普通话带台北台湾腔，语速中等偏慢，尾音自然，不要卷舌，不要过度甜腻。",
    ),
    PromptVariant(
        "soft_clear_taiwan",
        "柔和清楚台湾感",
        "台湾台北长大的年轻女生，声音柔和清楚，像大学生语音讯息。普通话有自然台湾口音，平翘舌比较轻，尾音有台湾女生的柔软感，语气温柔聪明。",
    ),
    PromptVariant(
        "less_cute_more_real",
        "更真人 少卖萌",
        "真实的台湾女大学生，像台大文学院学生，声音清亮温柔、聪明自然。可爱感只要一点点，不要卖萌，不要动画腔，不要大陆卷舌，尾音自然像台湾人聊天。",
    ),
]


TEXTS = [
    ("assistant", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("daily", "欸，你先不要紧张啦。我们一步一步来，把事情讲清楚就好了。"),
    ("taiwan_check", "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。"),
    ("soft_warning", "我不是要阻止你，只是现在继续往前走，可能会变得更危险。"),
    ("cute_request", "你可以再说一次吗？刚刚那句话我有听到，可是我想确认一下。"),
    ("long", "如果你真的想把这件事情做好，就不要急着证明自己，先把资料整理清楚。"),
]


def generate() -> list[dict[str, object]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID}...")
    model = load_model(MODEL_ID)
    rows: list[dict[str, object]] = []
    for variant in PROMPTS:
        variant_dir = OUT_DIR / variant.id
        variant_dir.mkdir(parents=True, exist_ok=True)
        for text_id, text in TEXTS:
            started = perf_counter()
            output = variant_dir / f"{text_id}.wav"
            try:
                generate_audio(
                    text=text,
                    model=model,
                    instruct=variant.instruct,
                    lang_code="zh",
                    output_path=str(variant_dir),
                    file_prefix=text_id,
                    audio_format="wav",
                    verbose=False,
                )
                generated = variant_dir / f"{text_id}_000.wav"
                if not generated.exists():
                    matches = sorted(variant_dir.glob(f"{text_id}*.wav"))
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
            print(variant.id, text_id, status, f"{seconds:.1f}s", output_text or error)
            rows.append(
                {
                    "variant_id": variant.id,
                    "label": variant.label,
                    "instruct": variant.instruct,
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
    by_variant: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_id"]), []).append(row)

    sections = []
    for variant_id, items in by_variant.items():
        first = items[0]
        sections.append(
            f"<section class=\"variant\"><h2>{html.escape(str(first['label']))}</h2>"
            f"<p class=\"id\">{html.escape(variant_id)}</p>"
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
<title>Qwen3 1.7B ntu_soft_smart prompt refine</title>
<style>
body{{margin:0;background:#f7f4ef;color:#181514;font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif;}}
header{{background:#b51d2a;color:white;padding:18px 16px 12px;}}
h1{{font-size:22px;line-height:1.2;margin:0 0 6px;}}
header p{{margin:0;font-size:14px;opacity:.9;line-height:1.5;}}
main{{padding:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;}}
.variant{{background:white;border:1px solid #ded6cc;padding:12px;}}
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
<h1>ntu_soft_smart 微调：更台湾、少卷舌、尾音自然</h1>
<p>围绕你选的冠军声音，微调 prompt。台词使用简中，测试台湾国语感、尾音、可爱但克制。</p>
</header>
<p class="note">建议先听每组的 assistant / taiwan_check / long。挑 1 到 2 个最稳的版本，下一步就用它当 teacher 产蒸馏资料。</p>
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

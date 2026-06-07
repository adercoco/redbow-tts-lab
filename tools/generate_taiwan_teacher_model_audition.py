#!/usr/bin/env python3
"""Generate teacher-voice auditions for Taiwan Mandarin female TTS."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v2"
QWEN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
F5_CLI = ROOT / ".venv-f5" / "bin" / "f5-tts_infer-cli"
REF_WAV = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_distill_v1" / "audio" / "distill_0158.wav"
REF_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"

TESTS = [
    (
        "calm",
        "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
    ),
    (
        "soft",
        "我知道你现在有点紧张，慢慢说就好，我会听完。",
    ),
]


@dataclass(frozen=True)
class QwenPrompt:
    id: str
    label: str
    prompt: str


QWEN_PROMPTS = [
    QwenPrompt(
        "low_r_clean",
        "低卷舌清亮台灣女生",
        "台湾年轻女生，中文普通话带台湾国语口音，卷舌很轻，儿化音几乎没有，语尾自然柔软。声音清亮温柔，有书卷气，像聪明的台大女生，克制、可爱、真实。",
    ),
    QwenPrompt(
        "taipei_soft_daily",
        "台北日常溫柔女生",
        "二十岁出头的台北女生，日常聊天口吻，声音清亮温柔，讲话像朋友传语音讯息。普通话有自然台湾腔，低卷舌，没有儿化音，尾音柔软自然，不要主播腔。",
    ),
    QwenPrompt(
        "ntu_literature_warm",
        "台大文學院書卷氣",
        "年轻台湾女生，像台大文学院女大学生，声音柔和清亮，有书卷气，讲话温柔聪明。台湾国语口音自然，卷舌很轻，语气克制但可爱，真实、不装、不娃娃音。",
    ),
    QwenPrompt(
        "cute_real_low_r",
        "可愛但真人低卷舌",
        "真实的台湾年轻女生，声音甜美清楚但不是动画声，不要过度撒娇。普通话带台湾国语口音，低卷舌，尾音轻轻上扬，像大学生在咖啡厅轻声聊天。",
    ),
    QwenPrompt(
        "bookstore_senpai",
        "書店學姊溫柔提醒",
        "台湾书店打工的年轻女生，像温柔学姐，小声但清楚，声音清亮有亲和力。普通话带台湾国语腔，低卷舌，完全没有儿化音，语尾自然柔软，不要播音腔。",
    ),
    QwenPrompt(
        "clinic_nurse_soft",
        "診所護理師親切聲",
        "台湾诊所年轻护理师，声音温柔、清楚、让人安心，讲话亲切但专业。普通话有自然台湾腔，卷舌很轻，语速中等偏慢，尾音柔和，不要机械感。",
    ),
    QwenPrompt(
        "customer_service_taipei",
        "台北客服乾淨親切",
        "台北年轻女生客服，声音干净清亮，礼貌、温柔、好听，但不要像机器人。普通话带低卷舌台湾国语，句尾自然，不要儿化音，不要大陆主播腔。",
    ),
    QwenPrompt(
        "campus_radio_sister",
        "校園廣播學姊",
        "台湾大学校园广播学姐，声音明亮清楚、温柔亲切，有一点笑意但不夸张。普通话自然台湾腔，卷舌轻，尾音软，像真的大学生在广播室讲话。",
    ),
    QwenPrompt(
        "coffee_shop_friend",
        "咖啡店朋友語音",
        "台湾年轻女生在咖啡店传语音讯息，声音轻柔好听，亲近、真实、带一点可爱。低卷舌台湾国语，没有儿化音，语气像朋友，不要主播感。",
    ),
    QwenPrompt(
        "soft_detective_assistant",
        "溫柔推理助手",
        "聪明温柔的台湾女生助手，讲话冷静、细心、声音清亮，有一点推理感但不严肃。普通话低卷舌、台湾腔自然，语尾柔软，适合说安抚和提醒的话。",
    ),
    QwenPrompt(
        "story_reader_taiwan",
        "台灣故事朗讀女生",
        "台湾年轻女生朗读故事，声音清亮温柔，有画面感但不夸张，像睡前轻声说话。普通话带台湾国语口音，卷舌轻，尾音自然，不要舞台腔。",
    ),
    QwenPrompt(
        "warm_app_voice",
        "App 內建溫柔女聲",
        "适合手机 app 的台湾年轻女生语音，声音温柔、清楚、轻快、耐听。低卷舌台湾国语，没有儿化音，语速自然，像真人助理，不要广告配音感。",
    ),
]


def esc(value: object) -> str:
    return html.escape(str(value))


def run_qwen() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model = load_model(QWEN_MODEL)
    for prompt in QWEN_PROMPTS:
        model_dir = OUT / "audio" / "qwen3_voicedesign" / prompt.id
        model_dir.mkdir(parents=True, exist_ok=True)
        for text_id, text in TESTS:
            started = perf_counter()
            final = model_dir / f"{text_id}.wav"
            try:
                generate_audio(
                    text=text,
                    model=model,
                    instruct=prompt.prompt,
                    lang_code="zh",
                    output_path=str(model_dir),
                    file_prefix=text_id,
                    audio_format="wav",
                    verbose=False,
                )
                generated = model_dir / f"{text_id}_000.wav"
                if not generated.exists():
                    matches = sorted(model_dir.glob(f"{text_id}*.wav"))
                    if not matches:
                        raise RuntimeError("Qwen generated no wav")
                    generated = matches[-1]
                shutil.copy2(generated, final)
                status = "ok"
                error = ""
            except Exception as exc:
                status = "failed"
                error = str(exc)
                final = None
            rows.append(
                {
                    "family": "Qwen3-TTS VoiceDesign",
                    "candidate_id": prompt.id,
                    "label": prompt.label,
                    "prompt": prompt.prompt,
                    "text_id": text_id,
                    "text": text,
                    "output": str(final) if final else None,
                    "seconds": perf_counter() - started,
                    "status": status,
                    "error": error,
                }
            )
    return rows


def run_f5() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model_dir = OUT / "audio" / "f5_tts_qwen_ref"
    model_dir.mkdir(parents=True, exist_ok=True)
    for text_id, text in TESTS:
        started = perf_counter()
        final = model_dir / f"{text_id}.wav"
        work = model_dir / f"_work_{text_id}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    str(F5_CLI),
                    "--model",
                    "F5TTS_v1_Base",
                    "--ref_audio",
                    str(REF_WAV),
                    "--ref_text",
                    REF_TEXT,
                    "--gen_text",
                    text,
                    "--output_dir",
                    str(work),
                    "--output_file",
                    final.name,
                    "--remove_silence",
                    "--nfe_step",
                    "32",
                ],
                cwd=work,
                check=True,
            )
            generated = work / final.name
            if not generated.exists():
                raise RuntimeError(f"F5 generated no wav at {generated}")
            shutil.copy2(generated, final)
            status = "ok"
            error = ""
        except Exception as exc:
            status = "failed"
            error = str(exc)
            final = None
        rows.append(
            {
                "family": "F5-TTS",
                "candidate_id": "f5_qwen_ref",
                "label": "F5-TTS clone Qwen teacher reference",
                "prompt": f"ref={REF_WAV.name}; ref_text={REF_TEXT}",
                "text_id": text_id,
                "text": text,
                "output": str(final) if final else None,
                "seconds": perf_counter() - started,
                "status": status,
                "error": error,
            }
        )
    return rows


def audio_rel(path_text: str) -> str:
    return Path(path_text).relative_to(OUT).as_posix()


def render_report(rows: list[dict[str, object]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["candidate_id"])), []).append(row)

    sections = []
    for (_family, _candidate_id), items in grouped.items():
        first = items[0]
        sections.append(
            f"""
            <article class="candidate">
              <div class="family">{esc(first['family'])}</div>
              <h2>{esc(first['label'])}</h2>
              <p class="prompt">{esc(first['prompt'])}</p>
            """
        )
        for row in items:
            sections.append(
                f"""
                <section class="sample">
                  <div class="sample-head"><b>{esc(row['text_id'])}</b><span>{float(row['seconds']):.1f}s</span></div>
                  <p>{esc(row['text'])}</p>
                """
            )
            if row["output"]:
                sections.append(f'<audio controls preload="metadata" src="{esc(audio_rel(str(row["output"])))}"></audio>')
            else:
                sections.append(f'<p class="error">{esc(row["error"])}</p>')
            sections.append("</section>")
        sections.append("</article>")

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台灣女生老師聲音候選</title>
  <style>
    :root {{
      --bg:#f7f8fb;
      --panel:#fff;
      --ink:#17181c;
      --muted:#5f6977;
      --line:#d9dee8;
      --accent:#b0182b;
      --soft:#f2f5f8;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      background:var(--bg);
      color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;
      line-height:1.58;
    }}
    header, main {{ max-width:1120px; margin:0 auto; padding:16px 12px; }}
    h1 {{ margin:0 0 8px; font-size:32px; line-height:1.15; letter-spacing:0; }}
    h2 {{ margin:4px 0 8px; font-size:20px; letter-spacing:0; }}
    p {{ margin:0; }}
    .lead {{ color:var(--muted); font-size:16px; max-width:880px; }}
    .notice, .candidate {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
      padding:14px;
    }}
    .notice {{ margin-top:12px; }}
    .grid {{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:10px;
      margin-top:12px;
    }}
    .family {{ color:var(--accent); font-weight:800; font-size:13px; }}
    .prompt {{ color:var(--muted); font-size:14px; }}
    .sample {{
      margin-top:10px;
      padding-top:10px;
      border-top:1px solid var(--line);
    }}
    .sample-head {{ display:flex; justify-content:space-between; gap:8px; color:var(--accent); font-size:13px; }}
    .sample p {{ margin:8px 0; font-weight:650; }}
    audio {{ width:100%; height:38px; }}
    .error {{ color:#9b1c31; }}
    @media (max-width:760px) {{
      h1 {{ font-size:28px; }}
      .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>台灣女生老師聲音候選</h1>
    <p class="lead">目標：道地台灣國語、低卷舌、清亮溫柔、年輕女生。這頁先跑本機可用模型：Qwen3 VoiceDesign 與 F5-TTS。CosyVoice / GPT-SoVITS 尚未裝本機權重，且 GPT-SoVITS 社群權重需逐一驗證授權與聲音來源。</p>
    <section class="notice">
      <p><b>判斷標準：</b>先聽是否像台灣人中文，其次聽女聲是否清亮溫柔，再看是否有奇怪捲舌、兒化音、主播腔或過度娃娃音。勝出的聲音才拿去產老師語料，避免把不好的口音蒸餾進 ZipVoice。</p>
    </section>
  </header>
  <main class="grid">{''.join(sections)}</main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(run_qwen())
    rows.extend(run_f5())
    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "index.html").write_text(render_report(rows), encoding="utf-8")
    print(OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

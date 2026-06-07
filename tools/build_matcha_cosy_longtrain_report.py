#!/usr/bin/env python3
"""Build a local listening report for Cosy teacher vs Matcha long-train variants."""

from __future__ import annotations

import html
import json
import shutil
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
REPORT = BASE / "reports" / "matcha_cosy_longtrain_v1"
AUDIO = REPORT / "audio"
TEACHER = BASE / "teacher_cosy_eval_matcha_four_v1" / "manifest.json"
ASR = REPORT / "asr_results.json"

SAMPLES = [
    ("01", "matcha_long_1", "你先不要急，我们慢慢来，把事情一件一件处理好。"),
    ("02", "matcha_long_2", "我刚刚看了一下，应该不是你的问题，你不用太担心。"),
    ("03", "matcha_long_3", "没关系啦，你先讲，我在这边听，真的不用紧张。"),
    ("04", "matcha_long_4", "如果你愿意的话，我们等一下再一起确认一次。"),
]

VARIANTS = [
    {
        "id": "matcha_15k_s16",
        "asr_key": "15k_s16",
        "label": "Matcha-Cosy 15k / 16-step",
        "dir": BASE / "students" / "matcha_cosy_golden_pinyin_v1_step15000" / "eval_s16_t050_r085_punct",
        "note": "目前 app 路線的前一版長訓基準。",
    },
    {
        "id": "matcha_30k_s8",
        "asr_key": "30k_s8",
        "label": "Matcha-Cosy 30k / 8-step",
        "dir": BASE / "students" / "matcha_cosy_golden_pinyin_v1_step30000" / "eval_s8_t050_r082_punct",
        "note": "手機速度候選，先看能不能不掉字、不糊。",
    },
    {
        "id": "matcha_30k_s16",
        "asr_key": "30k_s16",
        "label": "Matcha-Cosy 30k / 16-step",
        "dir": BASE / "students" / "matcha_cosy_golden_pinyin_v1_step30000" / "eval_s16_t050_r085_punct",
        "note": "品質優先候選，用來判斷長訓是否真的比 15k 自然。",
    },
    {
        "id": "matcha_30k_s24",
        "asr_key": "30k_s24",
        "label": "Matcha-Cosy 30k / 24-step",
        "dir": BASE / "students" / "matcha_cosy_golden_pinyin_v1_step30000" / "eval_s24_t050_r085_punct",
        "note": "品質上限測試；若沒有明顯更好，就不值得放手機。",
    },
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except Exception:
        return None


def copy_audio(src: Path, name: str) -> str:
    AUDIO.mkdir(parents=True, exist_ok=True)
    dst = AUDIO / name
    shutil.copy2(src, dst)
    return esc(dst.relative_to(REPORT))


def load_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return {"load_seconds": None, "samples": []}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def avg(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt_sec(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


def teacher_by_text() -> dict[str, dict]:
    if not TEACHER.exists():
        return {}
    rows = json.loads(TEACHER.read_text(encoding="utf-8"))
    return {row["text"]: row for row in rows if row.get("status") == "ok" and row.get("audio")}


def load_asr() -> dict[tuple[str, str], dict]:
    if not ASR.exists():
        return {}
    rows = json.loads(ASR.read_text(encoding="utf-8"))
    return {(row.get("variant", ""), row.get("sample", "")): row for row in rows}


def summarize_asr(asr: dict[tuple[str, str], dict], key: str) -> str:
    rows = [row for (variant, _), row in asr.items() if variant == key]
    if not rows:
        return "-"
    passed = sum(1 for row in rows if row.get("asr_pass"))
    score = avg([row.get("asr_similarity") for row in rows])
    return f"{passed}/{len(rows)} · {score:.3f}" if score is not None else f"{passed}/{len(rows)}"


def variant_audio(variant: dict, sample_id: str) -> Path:
    return variant["dir"] / f"{sample_id}.wav"


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    teachers = teacher_by_text()
    asr = load_asr()
    metrics = {variant["id"]: load_metrics(variant["dir"]) for variant in VARIANTS}

    rows = []
    for variant in VARIANTS:
        m = metrics[variant["id"]]
        gen = [sample.get("gen_seconds") for sample in m.get("samples", [])]
        rtf = [sample.get("rtf") for sample in m.get("samples", [])]
        rss = [sample.get("peak_rss_mb") for sample in m.get("samples", [])]
        model_dir = variant["dir"].parents[0]
        ckpt = model_dir / "checkpoints" / "last.ckpt"
        ckpt_mb = ckpt.stat().st_size / 1024 / 1024 if ckpt.exists() else None
        rows.append(
            f"""
            <tr>
              <td>{esc(variant["label"])}</td>
              <td>{fmt_sec(m.get("load_seconds"))}</td>
              <td>{fmt_sec(avg(gen))}</td>
              <td>{"-" if avg(rtf) is None else f"{avg(rtf):.3f}"}</td>
              <td>{"-" if avg(rss) is None else f"{avg(rss):.0f}MB"}</td>
              <td>{esc(summarize_asr(asr, variant["asr_key"]))}</td>
              <td>{"-" if ckpt_mb is None else f"{ckpt_mb:.1f}MB ckpt"}</td>
              <td>{esc(variant["note"])}</td>
            </tr>
            """
        )

    sample_blocks = []
    for no, sample_id, text in SAMPLES:
        cells = []
        teacher = teachers.get(text)
        if teacher:
            src = copy_audio(Path(teacher["audio"]), f"teacher_{no}.wav")
            asr_row = asr.get(("teacher", no), {})
            cells.append(
                f"""
                <article class="voice teacher">
                  <h4>CosyVoice2 teacher</h4>
                  <audio controls preload="metadata" src="{src}"></audio>
                  <p>{fmt_sec(teacher.get("seconds"))} 生成 · {fmt_sec(wav_seconds(Path(teacher["audio"])))} 音檔</p>
                  <p>ASR {esc(asr_row.get("asr_similarity", "-"))} · {esc(asr_row.get("asr", ""))}</p>
                </article>
                """
            )
        else:
            cells.append(
                """
                <article class="voice teacher missing">
                  <h4>CosyVoice2 teacher</h4>
                  <p>這句尚未生成同句 teacher，後續補齊後本格會自動出現播放器。</p>
                </article>
                """
            )

        for variant in VARIANTS:
            src_path = variant_audio(variant, sample_id)
            if not src_path.exists():
                cells.append(
                    f"""
                    <article class="voice missing">
                      <h4>{esc(variant["label"])}</h4>
                      <p>尚未產生。</p>
                    </article>
                    """
                )
                continue
            src = copy_audio(src_path, f"{variant['id']}_{no}.wav")
            metric = next((s for s in metrics[variant["id"]].get("samples", []) if s.get("id") == sample_id), {})
            asr_row = asr.get((variant["asr_key"], no), {})
            cells.append(
                f"""
                <article class="voice">
                  <h4>{esc(variant["label"])}</h4>
                  <audio controls preload="metadata" src="{src}"></audio>
                  <p>{fmt_sec(metric.get("gen_seconds"))} 生成 · {fmt_sec(metric.get("audio_seconds"))} 音檔 · RTF {"-" if metric.get("rtf") is None else f"{metric.get('rtf'):.3f}"}</p>
                  <p>ASR {esc(asr_row.get("asr_similarity", "-"))} · {esc(asr_row.get("asr", ""))}</p>
                </article>
                """
            )

        sample_blocks.append(
            f"""
            <section class="sample">
              <div class="kicker">Sample {no}</div>
              <h3>{esc(text)}</h3>
              <div class="grid">{''.join(cells)}</div>
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosy → Matcha 長訓試聽 v1</title>
  <style>
    :root {{ --paper:#f7f3ea; --ink:#25231f; --muted:#6f675d; --line:#d8cbbb; --panel:#fffaf2; --red:#a73530; --green:#52664f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans TC","PingFang TC",sans-serif; line-height:1.68; letter-spacing:0; }}
    main {{ width:min(1320px, calc(100vw - 56px)); margin:0 auto; padding:42px 0 72px; }}
    header {{ border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:28px; }}
    .eyebrow,.kicker {{ color:var(--red); font-size:13px; font-weight:760; }}
    h1 {{ margin:8px 0 12px; font-size:42px; line-height:1.12; letter-spacing:0; }}
    h2 {{ margin:36px 0 12px; font-size:26px; letter-spacing:0; }}
    h3 {{ margin:4px 0 14px; font-size:19px; letter-spacing:0; }}
    h4 {{ margin:0 0 8px; font-size:15px; letter-spacing:0; }}
    .lead {{ margin:0; max-width:920px; color:var(--muted); font-size:18px; }}
    .note,.sample,table {{ background:rgba(255,250,242,.8); border:1px solid var(--line); border-radius:8px; }}
    .note {{ padding:16px 18px; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; }}
    th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:14px; }}
    th {{ color:var(--muted); background:rgba(224,214,198,.35); font-size:12px; }}
    tr:last-child td {{ border-bottom:0; }}
    .sample {{ margin:14px 0; padding:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }}
    .voice {{ min-width:0; border-top:3px solid var(--green); padding-top:10px; }}
    .voice.teacher {{ border-color:var(--red); }}
    .voice.missing {{ color:var(--muted); }}
    .voice p {{ margin:7px 0 0; color:var(--muted); font-size:13px; }}
    audio {{ width:100%; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    @media (max-width:1100px) {{ main {{ width:min(100vw - 28px,1320px); }} .grid {{ grid-template-columns:1fr; }} h1 {{ font-size:34px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Cosy teacher → Matcha student</div>
    <h1>Cosy → Matcha 長訓試聽 v1</h1>
    <p class="lead">這頁是用來判斷 Matcha 再長訓後，有沒有更接近 CosyVoice2 golden teacher：語速是否放慢、字距是否自然、聲音是否更清楚、尾音是否比較像台灣溫柔女生。</p>
  </header>

  <section class="note">
    <p><strong>怎麼聽：</strong>先聽 Cosy teacher，再聽 15k baseline，最後聽 30k 的 8 / 16 / 24-step。8-step 是手機速度候選；16-step 是品質候選；24-step 只檢查上限。</p>
    <p><strong>ASR gate：</strong>Whisper 檢查 20/20 通過。這代表內容大致能辨識，不代表聲音一定自然；自然度仍要聽。</p>
    <p><strong>技術上這算什麼：</strong>這是 teacher-student imitation fine-tune，不是 few-step flow distillation。Matcha 學的是同一個 Cosy 聲線語料的文字到 mel 分佈；steps 是推論時 ODE 解算步數。</p>
  </section>

  <h2>模型與速度</h2>
  <table>
    <thead><tr><th>版本</th><th>載入</th><th>平均一句生成</th><th>RTF</th><th>Peak RSS</th><th>ASR</th><th>checkpoint</th><th>定位</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>逐句試聽</h2>
  {''.join(sample_blocks)}
</main>
</body>
</html>
"""
    (REPORT / "matcha_cosy_longtrain_v1.html").write_text(doc, encoding="utf-8")
    print(REPORT / "matcha_cosy_longtrain_v1.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

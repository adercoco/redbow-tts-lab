#!/usr/bin/env python3
"""Build an HTML inventory of clone/distillation data usage."""

from __future__ import annotations

import collections
import datetime as dt
import html
import json
from pathlib import Path

import pandas as pd
import soundfile as sf


ROOT = Path("/Users/ader/Documents/App")
OUT_DIR = (
    ROOT
    / "distillation/conan_authorized_voice_refs_v1/reports/clone_data_inventory_v1"
)


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def load_json(path: str | Path, default=None):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def wav_duration(path: str | Path) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return None
    try:
        info = sf.info(str(p))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


def summarize_manifest_audio(items: list[dict], audio_key: str = "audio") -> dict:
    ok = [x for x in items if x.get("status", "ok") == "ok"]
    durations = []
    for row in ok:
        dur = wav_duration(row.get(audio_key))
        if dur is not None:
            durations.append(dur)
    return {
        "items": len(items),
        "ok": len(ok),
        "audio_seconds": round(sum(durations), 2) if durations else None,
        "audio_hours": round(sum(durations) / 3600, 3) if durations else None,
        "avg_audio_seconds": round(sum(durations) / len(durations), 2)
        if durations
        else None,
    }


def fmt_seconds(seconds) -> str:
    if seconds is None:
        return "未量到"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} 小時"
    if seconds >= 60:
        return f"{seconds / 60:.1f} 分鐘"
    return f"{seconds:.2f} 秒"


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        tds = "".join(f"<td>{row.get(key, '')}</td>" for key, _ in columns)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Current app references.
    haibara_ref = (
        ROOT
        / "distillation/conan_authorized_voice_refs_v1/clips_24k/haibara_002_haibara_ep751_000112000_000116500.wav"
    )
    taiwan_ref = (
        ROOT
        / "distillation/taiwan_mandarin_low_r/golden_teacher/cosyvoice2_clear_best2_line04_v1/reference_clear_best2_7s.wav"
    )
    current_rows = [
        {
            "用途": "目前 app：灰原",
            "模型": "CosyVoice2-0.5B zero-shot clone",
            "實際 reference": "1 句授權灰原短片段",
            "reference 秒數": fmt_seconds(wav_duration(haibara_ref)),
            "來源 pool": "灰原 6 句 / 25.00 秒",
            "備註": "目前 preset 用 haibara_002，避免 best3 裡「給打開」污染。",
        },
        {
            "用途": "目前 app：台灣溫柔女聲",
            "模型": "CosyVoice2-0.5B zero-shot clone",
            "實際 reference": "clear_best2_7s，2 段女性聲音拼接",
            "reference 秒數": fmt_seconds(wav_duration(taiwan_ref)),
            "來源 pool": "嚴格乾淨片段 12 句；候選 top30",
            "備註": "這是風格聲音，不標示真人；不是 fine-tune。",
        },
    ]

    # Authorized character clips.
    conan_manifest = load_json(
        "distillation/conan_authorized_voice_refs_v1/clip_manifest.json", []
    )
    role_stats = collections.defaultdict(lambda: {"count": 0, "seconds": 0.0})
    for row in conan_manifest:
        role_stats[row["role"]]["count"] += 1
        role_stats[row["role"]]["seconds"] += float(row.get("duration") or 0)
    role_rows = [
        {
            "角色": role,
            "可用片段": stat["count"],
            "總秒數": fmt_seconds(stat["seconds"]),
            "用途": "reference 選擇 / Cosy zero-shot audition / 灰原 SFT 前處理",
        }
        for role, stat in sorted(role_stats.items())
    ]

    role_packs = load_json(
        "distillation/conan_authorized_voice_refs_v1/reference_packs/manifest.json", []
    )
    pack_rows = [
        {
            "pack": x["pack_id"],
            "角色": x["role"],
            "reference 片段數": len(x.get("ref_files", [])),
            "reference 秒數": fmt_seconds(float(x.get("duration") or 0)),
            "用途": "CosyVoice2 zero-shot audition",
        }
        for x in role_packs
    ]

    # Taiwan female source and packs.
    clean_csv = (
        ROOT
        / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/clean_strict_24k/metadata.csv"
    )
    top30_csv = (
        ROOT
        / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/tts_ready_24k_top30/metadata.csv"
    )
    source_rows = []
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        source_rows.append(
            {
                "資料集": "clean_strict_24k",
                "句數": len(df),
                "總秒數": fmt_seconds(float(df["duration"].sum())),
                "用途": "人工/規則篩後的乾淨女聲 reference pool",
            }
        )
    if top30_csv.exists():
        df = pd.read_csv(top30_csv)
        source_rows.append(
            {
                "資料集": "tts_ready_24k_top30",
                "句數": len(df),
                "總秒數": fmt_seconds(float(df["duration"].sum())),
                "用途": "較寬鬆的 reference 候選 pool",
            }
        )
    ref_packs = load_json(
        "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/manifest.json",
        [],
    )
    taiwan_pack_rows = [
        {
            "pack": x["pack_id"],
            "片段數": len(x.get("ref_files", [])),
            "reference 秒數": fmt_seconds(float(x.get("total_duration") or 0)),
            "用途": x.get("note", ""),
        }
        for x in ref_packs
    ]
    golden = load_json(
        "distillation/taiwan_mandarin_low_r/golden_teacher/cosyvoice2_clear_best2_line04_v1/manifest.json",
        {},
    )
    taiwan_pack_rows.append(
        {
            "pack": "clear_best2_7s / golden teacher reference",
            "片段數": len(golden.get("reference_source_files", [])),
            "reference 秒數": fmt_seconds(wav_duration(taiwan_ref)),
            "用途": "目前 app 台灣溫柔女聲 reference；也用來產 Cosy teacher corpus",
        }
    )

    # Clone audition rows grouped by model/ref.
    audition = load_json(
        "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/clone_audition_v4_original_models/all_results_asr_scored.json",
        [],
    )
    grouped = {}
    for row in audition:
        key = (row.get("family"), row.get("candidate_id"), row.get("ref_id"))
        g = grouped.setdefault(
            key,
            {
                "family": row.get("family"),
                "candidate_id": row.get("candidate_id"),
                "ref_id": row.get("ref_id"),
                "outputs": 0,
                "ref_audio": row.get("ref_audio"),
                "ref_files": row.get("ref_files") or [],
                "avg_asr": [],
                "avg_spk": [],
            },
        )
        g["outputs"] += 1
        if row.get("asr_similarity") is not None:
            g["avg_asr"].append(row["asr_similarity"])
        if row.get("speaker_cosine") is not None:
            g["avg_spk"].append(row["speaker_cosine"])
    audition_rows = []
    for g in grouped.values():
        audition_rows.append(
            {
                "模型/家族": esc(g["family"]),
                "candidate": esc(g["candidate_id"]),
                "ref": esc(g["ref_id"]),
                "reference 秒數": fmt_seconds(wav_duration(g["ref_audio"])),
                "reference 片段數": len(g["ref_files"]) if g["ref_files"] else "",
                "輸出句數": g["outputs"],
                "平均 ASR": round(sum(g["avg_asr"]) / len(g["avg_asr"]), 3)
                if g["avg_asr"]
                else "",
                "平均聲紋": round(sum(g["avg_spk"]) / len(g["avg_spk"]), 3)
                if g["avg_spk"]
                else "",
            }
        )
    audition_rows.sort(key=lambda x: (str(x["模型/家族"]), str(x["candidate"])))

    # Teacher corpora and student datasets.
    teacher_specs = [
        (
            "Qwen3 1.7B seed",
            "distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_seed/manifest.json",
            "prompt 產聲，無人聲 reference；用於早期選聲音",
        ),
        (
            "Qwen3 1.7B distill 500",
            "distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_distill_v1/manifest.json",
            "prompt 產聲；給 ZipVoice/Piper 早期學生",
        ),
        (
            "Cosy raw best2 64",
            "distillation/taiwan_mandarin_low_r/teacher_cosy_raw_best2_distill_v1/manifest.json",
            "CosyVoice2 + raw_best2_7s reference",
        ),
        (
            "Cosy clear best2 golden daily 500",
            "distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1/manifest.json",
            "CosyVoice2 + clear_best2_7s reference；目前 Matcha/ZipVoice 主要老師",
        ),
        (
            "IndexTTS2 large 500",
            "distillation/taiwan_mandarin_low_r/teacher_indextts2_distill_large_v1/manifest.json",
            "IndexTTS2 pack reference；給 Matcha/Piper 對照",
        ),
    ]
    teacher_rows = []
    for name, path, note in teacher_specs:
        items = load_json(path, [])
        stat = summarize_manifest_audio(items)
        first = items[0] if items else {}
        teacher_rows.append(
            {
                "teacher corpus": name,
                "句數": stat["items"],
                "成功": stat["ok"],
                "音訊總長": fmt_seconds(stat["audio_seconds"]),
                "平均每句": fmt_seconds(stat["avg_audio_seconds"]),
                "reference / 來源": esc(
                    first.get("teacher_ref_audio")
                    or first.get("reference_audio")
                    or first.get("teacher_voice_profile")
                    or "無人聲 reference"
                ),
                "備註": esc(note),
            }
        )

    dataset_specs = [
        (
            "ZipVoice <- Qwen prosody-filtered",
            "distillation/taiwan_mandarin_low_r/datasets/zipvoice_custom_finetune_v1/manifest.json",
        ),
        (
            "ZipVoice <- Cosy 64 smoke",
            "distillation/taiwan_mandarin_low_r/datasets/zipvoice_cosy_teacher_smoke_v1/manifest.json",
        ),
        (
            "ZipVoice <- Cosy golden 500",
            "distillation/taiwan_mandarin_low_r/datasets/zipvoice_cosy_golden_daily_500_v1/manifest.json",
        ),
        (
            "Matcha <- Cosy golden 500",
            "distillation/taiwan_mandarin_low_r/datasets/matcha_pinyin_cosy_golden_500_v1/dataset_manifest.json",
        ),
        (
            "Matcha <- IndexTTS2 500",
            "distillation/taiwan_mandarin_low_r/datasets/matcha_pinyin_indextts2_500_v1/dataset_manifest.json",
        ),
        (
            "Piper <- Qwen 500",
            "distillation/taiwan_mandarin_low_r/datasets/piper_ljspeech_distill_v1/dataset_manifest.json",
        ),
        (
            "Piper <- IndexTTS2 500",
            "distillation/taiwan_mandarin_low_r/datasets/piper_ljspeech_indextts2_500_v1/dataset_manifest.json",
        ),
    ]
    student_rows = []
    for name, path in dataset_specs:
        d = load_json(path, {})
        if "train_count" in d:
            items = int(d.get("train_count", 0)) + int(d.get("dev_count", 0))
            split = f"{d.get('train_count')} / {d.get('dev_count')}"
            source = d.get("corpus") or d.get("source_phase_dir")
            seconds = None
        else:
            items = d.get("items", "")
            split = f"{d.get('train_items', '')} / {d.get('valid_items', '')}".strip(" /")
            source = d.get("source_manifest") or d.get("source")
            seconds = d.get("total_seconds")
            if seconds is None and d.get("rows"):
                seconds = sum(float(r.get("seconds") or 0) for r in d["rows"])
        student_rows.append(
            {
                "學生資料集": esc(name),
                "items": items,
                "train/dev": split,
                "訓練音訊總長": fmt_seconds(seconds) if seconds else "由 source manifest 決定",
                "來源": f"<code>{esc(source)}</code>",
            }
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS Clone Data Inventory v1</title>
<style>
:root {{ --bg:#f7f4ee; --panel:#fffdf8; --ink:#2d2a26; --muted:#766f66; --line:#ddd5c9; --accent:#9f3b31; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Noto Sans TC','PingFang TC',sans-serif; line-height:1.58; }}
main {{ max-width:1240px; margin:0 auto; padding:32px 24px 72px; }}
h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
h2 {{ margin:34px 0 12px; padding-top:22px; border-top:1px solid var(--line); font-size:21px; }}
p {{ margin:10px 0; }}
.meta {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.label {{ color:var(--muted); font-size:13px; }}
.big {{ font-size:25px; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); margin:10px 0 18px; }}
th, td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; font-size:14px; }}
th {{ background:#ece5da; font-weight:700; }}
tr:last-child td {{ border-bottom:0; }}
code {{ font-family:'SFMono-Regular',Consolas,monospace; font-size:12px; overflow-wrap:anywhere; }}
.note {{ background:#fff8e8; border:1px solid #e7d2a4; border-radius:8px; padding:12px 14px; }}
@media (max-width:900px) {{ .cards {{ grid-template-columns:1fr; }} main {{ padding:20px 14px 48px; }} table {{ display:block; overflow-x:auto; }} }}
</style>
</head>
<body><main>
<h1>TTS Clone Data Inventory v1</h1>
<div class="meta">產生時間：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 目的：回答目前 clone / teacher / student 各用了多少資料。</div>

<div class="cards">
  <div class="card"><div class="label">目前 app 灰原 clone</div><div class="big">4.50 秒</div><div>1 句授權 reference，Cosy zero-shot。</div></div>
  <div class="card"><div class="label">目前 app 台灣女聲 clone</div><div class="big">{esc(fmt_seconds(wav_duration(taiwan_ref)))}</div><div>2 段清理後 reference，Cosy zero-shot。</div></div>
  <div class="card"><div class="label">最大 teacher corpus</div><div class="big">500 句</div><div>Qwen / Cosy / IndexTTS2 都有 500 句版本。</div></div>
  <div class="card"><div class="label">灰原 fine-tune 可用資料</div><div class="big">25.00 秒</div><div>已前處理，但太小，尚未真正 SFT。</div></div>
</div>

<div class="note">
<b>讀法：</b>zero-shot clone 是「幾秒 reference 決定聲音」；teacher corpus 是「用老師模型產很多句，再拿那些 wav 訓練小模型」；fine-tune 則是「模型權重真的更新」。這三種 data 量不能混著看。
</div>

<h2>現在 app 實際用多少 reference</h2>
{table(current_rows, [('用途','用途'),('模型','模型'),('實際 reference','實際 reference'),('reference 秒數','reference 秒數'),('來源 pool','來源 pool'),('備註','備註')])}

<h2>授權角色聲音資料池</h2>
{table(role_rows, [('角色','角色'),('可用片段','可用片段'),('總秒數','總秒數'),('用途','用途')])}
{table(pack_rows, [('pack','reference pack'),('角色','角色'),('reference 片段數','reference 片段數'),('reference 秒數','reference 秒數'),('用途','用途')])}

<h2>台灣女聲 reference 資料池</h2>
{table(source_rows, [('資料集','資料集'),('句數','句數'),('總秒數','總秒數'),('用途','用途')])}
{table(taiwan_pack_rows, [('pack','reference pack'),('片段數','片段數'),('reference 秒數','reference 秒數'),('用途','用途')])}

<h2>原始 clone audition 用的資料</h2>
<p>這裡是 F5 / GPT-SoVITS / CosyVoice2 / IndexTTS2 / Qwen / ZipVoice 等原始 clone 測試。多數模型都不是用 500 句訓練，而是拿同一組 7-11 秒 reference 做 zero-shot 或 pack clone。</p>
{table(audition_rows, [('模型/家族','模型/家族'),('candidate','candidate'),('ref','ref'),('reference 秒數','reference 秒數'),('reference 片段數','reference 片段數'),('輸出句數','輸出句數'),('平均 ASR','平均 ASR'),('平均聲紋','平均聲紋')])}

<h2>Teacher corpus：拿老師模型生出來的訓練語料</h2>
{table(teacher_rows, [('teacher corpus','teacher corpus'),('句數','句數'),('成功','成功'),('音訊總長','音訊總長'),('平均每句','平均每句'),('reference / 來源','reference / 來源'),('備註','備註')])}

<h2>學生模型訓練資料</h2>
<p>這一區才是 Piper / Matcha / ZipVoice 等學生模型真正吃進去的 dataset。它們不是直接 clone 原始人聲，而是學 Qwen、Cosy 或 IndexTTS2 產出的 teacher wav。</p>
{table(student_rows, [('學生資料集','學生資料集'),('items','items'),('train/dev','train/dev'),('訓練音訊總長','訓練音訊總長'),('來源','來源')])}

<h2>工程結論</h2>
<ul>
<li><b>現在灰原 app 聲音：</b>其實只靠 4.5 秒 reference zero-shot，不是 fine-tune。</li>
<li><b>現在台灣女聲 app 聲音：</b>靠 7 秒左右清理 reference zero-shot，背後不是真人名義，是聲音風格。</li>
<li><b>Matcha / ZipVoice 這些學生：</b>主要用 500 句 teacher corpus；資料量比 zero-shot 大很多，但學的是老師輸出的聲音，不是原始人聲本身。</li>
<li><b>灰原 Cosy fine-tune：</b>目前只有 25 秒授權資料，已做訓練前處理；要真的穩，至少需要更多同角色乾淨片段和 CUDA GPU。</li>
</ul>

</main></body></html>
"""

    (OUT_DIR / "clone_data_inventory_v1.html").write_text(report, encoding="utf-8")
    (OUT_DIR / "index.html").write_text(report, encoding="utf-8")
    print(OUT_DIR / "clone_data_inventory_v1.html")


if __name__ == "__main__":
    main()

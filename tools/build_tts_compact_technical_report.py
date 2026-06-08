#!/usr/bin/env python3
"""Build a compact technical entry report for the TTS experiments."""

from __future__ import annotations

import datetime as dt
import html
import json
import shutil
import statistics
from pathlib import Path


ROOT = Path("/Users/ader/Documents/App")
OUT_DIR = ROOT / "distillation/taiwan_mandarin_low_r/reports/tts_compact_technical_v1"
ASSETS = OUT_DIR / "assets"
OUT = OUT_DIR / "index.html"


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def load_json(path: str | Path, default=None):
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return default
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def copy_asset(src: str | Path, name: str) -> str:
    p = ROOT / src if not Path(src).is_absolute() else Path(src)
    dst = ASSETS / name
    if p.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        return f"assets/{name}"
    return ""


def file_size(path: str | Path) -> str:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return "未量到"
    total = p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024 or unit == "TB":
            return f"{total:.1f}{unit}" if unit != "B" else f"{total}B"
        total /= 1024
    return f"{total:.1f}TB"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt_sec(value: float | None) -> str:
    return "未量到" if value is None else f"{value:.2f}s"


def audio_tag(src: str) -> str:
    return f'<audio controls preload="metadata" src="{esc(src)}"></audio>' if src else "音檔未找到"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def summarize_matcha_metrics(path: str) -> dict:
    data = load_json(path, {})
    samples = data.get("samples", [])
    gen = [float(x.get("gen_seconds") or x.get("total_seconds")) for x in samples if x.get("gen_seconds") or x.get("total_seconds")]
    rtf = [float(x["rtf"]) for x in samples if x.get("rtf") is not None]
    rss = [float(x["peak_rss_mb"]) for x in samples if x.get("peak_rss_mb") is not None]
    return {
        "load": data.get("load_seconds"),
        "gen_avg": mean(gen),
        "rtf_avg": mean(rtf),
        "peak_rss": max(rss) if rss else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Audio examples copied into this report.
    examples = {
        "cosy_teacher": copy_asset(
            "distillation/taiwan_mandarin_low_r/teacher_cosy_eval_matcha_four_v1/audio/matcha_eval_01.wav",
            "cosy_teacher_matcha_eval_01.wav",
        ),
        "qwen_teacher": copy_asset(
            "distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/audio/teacher_qwen3_1p7b/seed_0001.wav",
            "qwen3_1p7b_seed_0001.wav",
        ),
        "index_teacher": copy_asset(
            "distillation/taiwan_mandarin_low_r/teacher_indextts2_distill_large_v1/audio/indextts2_tw_0001.wav",
            "indextts2_teacher_0001.wav",
        ),
        "zipvoice_clean": copy_asset(
            "distillation/taiwan_mandarin_low_r/reports/zipvoice_4step_optimize_v1/assets/qvocoder_clean_mild/cosy_01.wav",
            "zipvoice_qvocoder_clean_mild_cosy_01.wav",
        ),
        "zipvoice_step3": copy_asset(
            "distillation/taiwan_mandarin_low_r/reports/qwen_zipvoice_speedup_v1/audio/sherpa_step3_s01.wav",
            "zipvoice_sherpa_step3_s01.wav",
        ),
        "matcha_cosy": copy_asset(
            "distillation/taiwan_mandarin_low_r/students/matcha_cosy_golden_pinyin_v1_step30000/eval_s16_t050_r085_punct/matcha_long_1.wav",
            "matcha_cosy_30k_s16_01.wav",
        ),
        "matcha_cosy_onnx": copy_asset(
            "distillation/taiwan_mandarin_low_r/students/matcha_cosy_golden_pinyin_v1_step30000/onnx_mobile/samples_steps16_compact/matcha_onnx_1.wav",
            "matcha_cosy_30k_onnx_01.wav",
        ),
        "piper": copy_asset(
            "distillation/taiwan_mandarin_low_r/reports/mobile_model_summary/audio/piper_step1000_matmul_qint8/fast_1.wav",
            "piper_step1000_qint8_fast_1.wav",
        ),
        "f5": copy_asset(
            "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/clone_audition_v4_original_models/audio/f5_tts_v1_base_pack_best2/line_01.wav",
            "f5_tts_v1_base_line_01.wav",
        ),
        "gpt_sovits": copy_asset(
            "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/clone_audition_v4_original_models/audio/gpt_sovits_v2_single_ref/line_01.wav",
            "gpt_sovits_v2_line_01.wav",
        ),
    }

    runtime = load_json("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/runtime_metrics.json", {})
    qwen_summary = load_json("distillation/taiwan_mandarin_low_r/reports/zipvoice_three_way/zipvoice_distill_v1_summary.json", {})
    speedup = load_json("distillation/taiwan_mandarin_low_r/reports/qwen_zipvoice_speedup_v1/summary.json", {})
    piper_q = load_json("distillation/taiwan_mandarin_low_r/reports/mobile_model_summary/piper_step1000_matmul_qint8_metrics.json", {})
    piper_samples = piper_q.get("samples", [])
    piper_gen = [float(x["total_s"]) for x in piper_samples if x.get("total_s") is not None]
    piper_rtf = [float(x["rtf"]) for x in piper_samples if x.get("rtf") is not None]
    matcha_pt = summarize_matcha_metrics(
        "distillation/taiwan_mandarin_low_r/students/matcha_cosy_golden_pinyin_v1_step30000/eval_s16_t050_r085_punct/metrics.json"
    )
    matcha_onnx = summarize_matcha_metrics(
        "distillation/taiwan_mandarin_low_r/students/matcha_cosy_golden_pinyin_v1_step30000/onnx_mobile/samples_steps16_compact/metrics.json"
    )
    clone_rows = load_json(
        "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/clone_audition_v4_original_models/all_results_asr_scored.json",
        [],
    )

    def clone_stats(family: str) -> dict:
        rows = [r for r in clone_rows if r.get("family") == family and r.get("status", "ok") == "ok"]
        secs = [float(r["seconds"]) for r in rows if isinstance(r.get("seconds"), (int, float))]
        asr = [float(r["asr_similarity"]) for r in rows if isinstance(r.get("asr_similarity"), (int, float))]
        spk = [float(r["speaker_cosine"]) for r in rows if isinstance(r.get("speaker_cosine"), (int, float))]
        return {
            "n": len(rows),
            "sec_med": statistics.median(secs) if secs else None,
            "asr": mean(asr),
            "spk": mean(spk),
        }

    cosy_clone = clone_stats("CosyVoice2-0.5B original")
    index_clone = clone_stats("IndexTTS2 original")
    f5_clone = clone_stats("F5-TTS v1 Base original")
    gpt_clone = clone_stats("GPT-SoVITS v2 original single-ref")

    model_rows = [
        [
            "<b>CosyVoice2-0.5B</b><br><span class='muted'>目前角色變聲器主力；zero-shot clone / 未 fine-tune</span>",
            "4.5GB model dir",
            "Cosy teacher 報告：5.73GB peak RSS；目前 server 常駐吃數 GB",
            "台灣女聲 sweep median 6.23s/句；角色 audition median 12.22s/句",
            "音質/聲線上限最好，適合 server；不適合手機離線。",
            audio_tag(examples["cosy_teacher"]),
        ],
        [
            "<b>IndexTTS2 pack</b><br><span class='muted'>你聽感最喜歡的 teacher 候選之一</span>",
            "repo 9.5GB；checkpoints 8.3GB",
            "未做完整 peak RSS；本機 MPS 熱機後約 8-12s/句，首句曾約 47.9s",
            f"clone audition median {fmt_sec(index_clone['sec_med'])}; ASR {index_clone['asr']:.3f}; spk {index_clone['spk']:.3f}",
            "很適合當 teacher；直接手機跑機率低，授權也要另外確認。",
            audio_tag(examples["index_teacher"]),
        ],
        [
            "<b>Qwen3 1.7B VoiceDesign</b><br><span class='muted'>早期台灣女生 prompt 老師</span>",
            "約 2.2GB cache / MLX 4bit",
            runtime.get("teacher_qwen3_1p7b", {}).get("peak_rss_mb", "2.2GB 級"),
            runtime.get("teacher_qwen3_1p7b", {}).get("seconds_per_sentence", "1.7-2.1s/句"),
            "VoiceDesign 可塑性強；純 clone 分數低，不適合當角色 clone 主線。",
            audio_tag(examples["qwen_teacher"]),
        ],
        [
            "<b>ZipVoice / Sherpa ONNX int8</b><br><span class='muted'>手機離線候選，但目前有電子/乾淨度問題</span>",
            "核心 175.8MB；pack 約 181MB；int4 try 124MB",
            "Sherpa speed trial peak 959MB；Cosy step ladder peak 約 1.55GB",
            "Qwen route step3 avg 1.11s/句；Cosy 4-step int8 三句 9.78s / RTF 0.45",
            "速度和大小有希望；音質需 few-step distillation + vocoder/denoise 改善。",
            audio_tag(examples["zipvoice_clean"]) + "<br><span class='muted'>qVocoder + mild clean</span>",
        ],
        [
            "<b>Matcha-Cosy 30k</b><br><span class='muted'>目前手機離線最像樣的工程主線</span>",
            "ONNX acoustic 73MB；student dir 295MB；另需 vocoder",
            f"PyTorch eval peak {matcha_pt['peak_rss']:.0f}MB；ONNX RSS 尚未完整量測",
            f"ONNX avg {fmt_sec(matcha_onnx['gen_avg'])}/句；RTF {matcha_onnx['rtf_avg']:.3f}",
            "比 Piper 自然很多，速度也好；仍有字距/自然度問題，需更多資料與 vocoder tuning。",
            audio_tag(examples["matcha_cosy_onnx"]) + "<br><span class='muted'>30k ONNX sample</span>",
        ],
        [
            "<b>Piper / VITS</b><br><span class='muted'>最小最快，但你已判斷機械音太重</span>",
            f"{piper_q.get('model_size_mb', 60):.1f}MB ONNX",
            f"peak {piper_q.get('peak_rss_mb', 520):.0f}MB",
            f"avg {fmt_sec(mean(piper_gen))}/句；RTF {mean(piper_rtf):.3f}",
            "反應非常快；自然度/情緒/像 teacher 的能力明顯不足，不做主線。",
            audio_tag(examples["piper"]),
        ],
    ]

    side_rows = [
        [
            "F5-TTS v1 Base",
            "zero-shot clone",
            f"median {fmt_sec(f5_clone['sec_med'])}; ASR {f5_clone['asr']:.3f}; spk {f5_clone['spk']:.3f}",
            "clone 還可以，但速度慢、手機部署不如 Matcha/ZipVoice 直觀。",
            audio_tag(examples["f5"]),
        ],
        [
            "GPT-SoVITS v2",
            "single-ref zero-shot",
            f"median {fmt_sec(gpt_clone['sec_med'])}; ASR {gpt_clone['asr']:.3f}; spk {gpt_clone['spk']:.3f}",
            "本輪內容可懂但聲紋分數低；先保留，不當主線。",
            audio_tag(examples["gpt_sovits"]),
        ],
    ]

    data_rows = [
        ["目前灰原授權資料", "6 clips / 25s", "已完成 wav.scp/text/utt2spk/spk2utt、embedding、speech token、parquet；只夠 smoke / overfit，不夠正式 SFT。"],
        ["Cosy golden teacher corpus", "500 句 / 46.3 分鐘", "CosyVoice2 + clear_best2_7s 生成；目前 Matcha/ZipVoice 主要老師資料。"],
        ["IndexTTS2 teacher corpus", "500 句 / 45.2 分鐘", "給 Matcha/Piper 對照；聽感好但授權/手機 runtime 較難。"],
        ["Qwen3 1.7B corpus", "500 句 / 1.25 小時", "早期 Qwen → ZipVoice/Piper 老師資料。"],
    ]

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS 技術總覽精簡版 v1</title>
<style>
:root {{
  --paper:#f5f2ec; --panel:#fffdf8; --ink:#28241f; --muted:#6f685e;
  --line:#ded6ca; --accent:#a4362d; --green:#2f6f5e; --amber:#9a6a14;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC",sans-serif; line-height:1.62; }}
main {{ max-width:1280px; margin:0 auto; padding:34px 24px 72px; }}
h1 {{ font-size:34px; line-height:1.16; margin:0 0 8px; letter-spacing:0; }}
h2 {{ font-size:23px; margin:34px 0 12px; padding-top:22px; border-top:1px solid var(--line); }}
h3 {{ font-size:17px; margin:22px 0 8px; }}
p {{ margin:8px 0 13px; }}
.meta,.muted {{ color:var(--muted); }}
.lead {{ max-width:960px; font-size:18px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0 24px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:15px; }}
.card b {{ font-size:22px; }}
.ok {{ color:var(--green); font-weight:750; }}
.warn {{ color:var(--amber); font-weight:750; }}
.bad {{ color:var(--accent); font-weight:750; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin:12px 0 22px; }}
th,td {{ border-bottom:1px solid var(--line); padding:11px 12px; vertical-align:top; text-align:left; }}
th {{ background:#ebe3d8; font-weight:760; white-space:nowrap; }}
tr:last-child td {{ border-bottom:0; }}
audio {{ width:240px; max-width:100%; display:block; margin:4px 0; }}
code,pre {{ font-family:"SFMono-Regular",Consolas,monospace; }}
code {{ background:#eee6dc; padding:1px 4px; border-radius:4px; }}
pre {{ white-space:pre-wrap; background:#292520; color:#fffaf0; padding:14px; border-radius:8px; overflow:auto; }}
.flow {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin:14px 0 22px; }}
.step {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:13px; min-height:142px; }}
.step b {{ display:block; margin-bottom:6px; }}
.callout {{ border-left:4px solid var(--accent); background:#fff8ef; padding:12px 14px; border-radius:6px; margin:16px 0; }}
ul {{ padding-left:22px; }}
li {{ margin:5px 0; }}
a {{ color:#8e2c25; }}
@media (max-width:900px) {{
  main {{ padding:22px 14px 56px; }}
  h1 {{ font-size:28px; }}
  .grid,.flow {{ grid-template-columns:1fr; }}
  table {{ display:block; overflow-x:auto; }}
}}
</style>
</head>
<body><main>
<h1>TTS 技術總覽精簡版 v1</h1>
<div class="meta">產生時間：{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ｜ 目的：把目前所有 TTS 報告壓成可決策的一頁</div>
<p class="lead">目前最清楚的方向：<b>短期玩具/家人 demo 用 Cosy server zero-shot</b>；<b>手機離線先押 Matcha-Cosy 30k + vocoder</b>；<b>ZipVoice 還值得救，但要處理電子雜音與 few-step distillation</b>；<b>認真 Cosy fine-tune 需要先補乾淨授權資料到至少 10-30 分鐘</b>。</p>

<section class="grid">
  <div class="card"><span class="muted">現在可用 app</span><br><b>Cosy server</b><p>聲音最好，但靠 Mac/雲端 server，不是手機離線。</p></div>
  <div class="card"><span class="muted">手機離線主線</span><br><b>Matcha-Cosy</b><p>73MB acoustic ONNX，速度好，目前最值得打磨自然度。</p></div>
  <div class="card"><span class="muted">速度實驗</span><br><b>ZipVoice</b><p>175.8MB / step3 可到 1.11s 句，但乾淨度還不穩。</p></div>
  <div class="card"><span class="muted">正式聲音資產</span><br><b>Cosy SFT</b><p>先補資料，再 fine-tune，再蒸餾到手機模型。</p></div>
</section>

<h2>模型對照表</h2>
{table(["模型", "大小", "記憶體", "生成時間", "目前判斷", "實際範例"], model_rows)}

<h2>其他試過但非主線</h2>
{table(["模型", "方法", "量測", "判斷", "範例"], side_rows)}

<h2>目前路線圖</h2>
<div class="flow">
  <div class="step"><b>1. 現場 demo</b>CosyVoice2 zero-shot clone。用乾淨 reference + prompt text，server 常駐，Cloudflare Tunnel 給手機玩。</div>
  <div class="step"><b>2. 聲音資產化</b>補乾淨授權資料，做人聲切句、逐字稿、speaker metadata。目標先 10-30 分鐘，最好 1-3 小時。</div>
  <div class="step"><b>3. Cosy speaker fine-tune</b>更新 Cosy 的 llm/flow 權重，讓固定聲線更穩，減少 zero-shot reference 污染。</div>
  <div class="step"><b>4. Teacher corpus</b>用 SFT 後的 Cosy 生成數百到數千句高品質同聲線 paired data。</div>
  <div class="step"><b>5. 手機 student</b>蒸餾到 Matcha / ZipVoice，匯出 ONNX/Core ML，做 app 端載入一次、按住講話、ASR→TTS。</div>
</div>

<h2>模型怎麼模仿聲音</h2>
{table(["技術", "訓練權重", "輸入資料", "內部做什麼", "風險"], [
    ["Zero-shot clone", "不更新", "3-15 秒 reference wav + 對齊 prompt text", "模型用 reference 的 speaker/prosody conditioning 生成新句子。Cosy/F5/IndexTTS2 主要是這類。", "reference 有雜音、怪開頭、BGM、逐字稿不準時，輸出會被污染。"],
    ["Speaker cache", "不更新", "同 zero-shot", "先把 speaker info/cache 存起來，避免每次重新抽特徵。", "只改善速度/穩定管理，不會真的學更多聲音。"],
    ["Fine-tune / SFT", "會更新", "多句 wav + 正確文字 + speaker id", "從 base checkpoint 出發，讓 acoustic/LLM/flow 參數往固定聲線與語氣分佈靠近。", "資料太少會 overfit；逐字稿錯會訓壞發音。"],
    ["Teacher-student 蒸餾", "會更新 student", "teacher 產出的 text/wav corpus", "學生不直接學真人原音，而是學老師模型輸出的聲音分佈。", "student 小而快，但自然度、情緒、咬字常掉。"],
    ["Few-step distillation", "會更新 flow student", "teacher trajectory / paired data", "讓 16-step 品質被壓到 4/3-step，不能只在推論硬降步數。", "做不好會電子聲、漏字、糊。"],
])}

<h2>目前資料集</h2>
{table(["資料", "規模", "用途"], data_rows)}

<h2>Cosy fine-tune 要準備什麼資料</h2>
<div class="callout"><b>目標：</b>讓 Cosy 從「幾秒 reference 的 zero-shot clone」升級成「固定聲線 speaker SFT」。這不會讓模型變小，但會讓聲音更穩、更少 reference 污染，之後再拿它當 teacher 蒸餾到手機小模型。</div>
{table(["項目", "要求", "原因"], [
    ["授權", "講者/角色資料要明確允許 AI TTS 訓練、衍生模型、商用範圍。", "模型 license 只管模型本身，不幫你取得聲音/角色權利。"],
    ["總時長", "最低 10-30 分鐘乾淨單人語音；較穩建議 1-3 小時；25 秒只夠 smoke test。", "SFT 要學聲線分佈與不同文字覆蓋，幾秒 reference 不夠。"],
    ["切句", "每句 2-10 秒，單人、無重疊、句首句尾不要切掉；保留 100-200ms 自然空白。", "太短缺上下文，太長 alignment 容易壞。"],
    ["音質", "無 BGM、無旁白、無掌聲、無重疊人聲、無唱歌；統一 mono 24k/16k wav，響度約 -20dB RMS。", "雜訊會被模型當成聲音特徵學進去。"],
    ["逐字稿", "每句人工校正，文字與音檔逐字對齊；口語詞、語助詞要保留。", "錯字會直接導致念錯；Cosy zero-shot/SFT 都很吃 prompt text 對齊。"],
    ["metadata", "準備 wav.scp、text、utt2spk、spk2utt；單聲線可全部同一個 speaker id。", "Cosy 官方 recipe 的基本訓練格式。"],
    ["特徵", "抽 CampPlus speaker embedding、speech token，再打包 parquet。", "訓練腳本不直接吃散亂 wav，需要特徵和索引。"],
    ["切分", "train/dev split，例如 95/5；dev 不要和 train 重複。", "用 dev loss 和試聽避免只記住訓練句。"],
    ["評估句", "固定 20-50 句日常測試句，不進 train。", "每次 checkpoint 都用同一組句子比較音色、漏字、背景噪。"],
])}

<h3>Cosy SFT 實際檔案格式</h3>
<pre>wav.scp
utt_000001 /path/to/utt_000001.wav

text
utt_000001 你先不要急，我們慢慢來，把事情一件一件處理好。

utt2spk
utt_000001 haibara_authorized

spk2utt
haibara_authorized utt_000001 utt_000002 ...

Cosy preprocessing:
tools/extract_embedding.py      -> utt2embedding.pt / spk2embedding.pt
tools/extract_speech_token.py   -> utt2speech_token.pt
tools/make_parquet_list.py      -> parquet/data.list
cosyvoice/bin/train.py          -> fine-tune llm / flow / optional vocoder</pre>

<h2>決策建議</h2>
<ul>
  <li><b class="ok">現在要給人玩：</b>維持 Cosy server，先做 speaker cache、文字正規化、reference 清理。</li>
  <li><b class="ok">現在要手機離線：</b>先把 Matcha-Cosy 30k 放進 app，做語速、後處理、vocoder 改善。</li>
  <li><b class="warn">ZipVoice：</b>保留，專攻 few-step distillation + qVocoder/denoise；不要再只硬降 step。</li>
  <li><b class="bad">Piper：</b>除非只要超快語音提示，不建議繼續當主聲音。</li>
  <li><b class="ok">認真聲音：</b>補 10-30 分鐘以上乾淨授權資料，先跑 Cosy SFT，再重做 teacher corpus 和 mobile distillation。</li>
</ul>

<h2>原始重要報告入口</h2>
<ul>
  <li><a href="../report_index_v1/index.html">完整報告索引</a></li>
  <li><a href="../../../conan_authorized_voice_refs_v1/reports/cosy_clone_vs_finetune_v1/cosy_clone_vs_finetune_v1.html">Cosy clone vs fine-tune 技術報告</a></li>
  <li><a href="../matcha_cosy_longtrain_v1/matcha_cosy_longtrain_v1.html">Cosy → Matcha 長訓試聽</a></li>
  <li><a href="../cosy_zipvoice_true_distill_step_ladder_v1/cosy_zipvoice_true_distill_step_ladder_v1_standalone.html">Cosy → ZipVoice step ladder</a></li>
  <li><a href="../qwen_zipvoice_speedup_v1/qwen_zipvoice_speedup_v1_standalone.html">Qwen → ZipVoice speedup</a></li>
</ul>
</main></body></html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a detailed report comparing Cosy zero-shot clone and fine-tune."""

from __future__ import annotations

import datetime as dt
import html
import json
import statistics
from pathlib import Path


ROOT = Path("/Users/ader/Documents/App")
OUT_DIR = (
    ROOT
    / "distillation/conan_authorized_voice_refs_v1/reports/cosy_clone_vs_finetune_v1"
)
OUT = OUT_DIR / "cosy_clone_vs_finetune_v1.html"


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def load_json(path: str | Path, default=None):
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return default
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def du(path: str | Path) -> str:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return "不存在"
    total = 0
    if p.is_file():
        total = p.stat().st_size
    else:
        for child in p.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024 or unit == "TB":
            return f"{total:.1f}{unit}" if unit != "B" else f"{total}B"
        total /= 1024
    return f"{total:.1f}TB"


def timing_summary(path: str | Path) -> dict:
    rows = load_json(path, [])
    vals = [
        float(row["seconds"])
        for row in rows
        if isinstance(row, dict)
        and row.get("status", "ok") == "ok"
        and isinstance(row.get("seconds"), (int, float))
    ]
    if not vals:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }


def fmt_num(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "未量到"
    return f"{value:.2f}{suffix}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    haibara = load_json(
        "distillation/conan_authorized_voice_refs_v1/cosy_speaker_finetune_v1/dataset_summary.json",
        {},
    )
    current_inv = (
        "distillation/conan_authorized_voice_refs_v1/reports/clone_data_inventory_v1/index.html"
    )
    haibara_sft_report = (
        "distillation/conan_authorized_voice_refs_v1/reports/haibara_cosy_speaker_finetune_v1/"
        "haibara_cosy_speaker_finetune_v1.html"
    )
    char_audition_report = (
        "distillation/conan_authorized_voice_refs_v1/reports/conan_authorized_cosy_clone_audition_v1/"
        "conan_authorized_cosy_clone_audition_v1_standalone.html"
    )
    taiwan_cosy_report = (
        "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/"
        "cosy_zipvoice_distill_v1/cosy_vs_zipvoice_vs_distill_standalone.html"
    )

    t_haibara = timing_summary(
        "distillation/conan_authorized_voice_refs_v1/cosy_clone_audition_v1/cosy_clone_results.json"
    )
    t_taiwan = timing_summary(
        "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/"
        "cosy_zipvoice_distill_v1/cosy_sweep_results.json"
    )
    t_original = timing_summary(
        "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/"
        "clone_audition_v4_original_models/cosy_results.json"
    )

    rows = haibara.get("rows", [])
    clip_rows = []
    for row in rows:
        clip_rows.append(
            [
                f"<code>{esc(row.get('utt'))}</code>",
                f"{float(row.get('duration') or 0):.1f}s",
                esc(row.get("text")),
                esc(row.get("original_asr")),
            ]
        )

    generated = (
        ROOT
        / "distillation/conan_authorized_voice_refs_v1/haibara_cosy_web_v1/generated"
    )
    generated_count = len(list(generated.glob("*.wav"))) if generated.exists() else 0

    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cosy Clone vs Fine-tune 技術報告 v1</title>
<style>
:root {{
  --paper:#f6f3ed; --panel:#fffdf8; --ink:#2d2923; --muted:#746d62;
  --line:#ded7cc; --accent:#9f3029; --accent2:#2f6f5e; --code:#28241f;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC",sans-serif;
  line-height:1.62;
}}
main {{ max-width:1160px; margin:0 auto; padding:34px 24px 72px; }}
h1 {{ font-size:34px; line-height:1.18; margin:0 0 8px; letter-spacing:0; }}
h2 {{ font-size:22px; margin:34px 0 12px; padding-top:22px; border-top:1px solid var(--line); }}
h3 {{ font-size:17px; margin:24px 0 8px; }}
p {{ margin:8px 0 13px; }}
.meta,.small {{ color:var(--muted); font-size:14px; }}
.lead {{ font-size:18px; max-width:900px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:15px; }}
.label {{ color:var(--muted); font-size:13px; }}
.big {{ font-size:25px; font-weight:760; margin-top:2px; }}
.ok {{ color:var(--accent2); font-weight:760; }}
.warn {{ color:#a65f00; font-weight:760; }}
.bad {{ color:var(--accent); font-weight:760; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin:12px 0 22px; }}
th,td {{ border-bottom:1px solid var(--line); padding:10px 12px; text-align:left; vertical-align:top; }}
th {{ background:#ece4d8; font-weight:760; }}
tr:last-child td {{ border-bottom:0; }}
code,pre {{ font-family:"SFMono-Regular",Consolas,monospace; }}
code {{ background:#eee7dc; padding:1px 4px; border-radius:4px; }}
pre {{ white-space:pre-wrap; background:var(--code); color:#fffaf0; padding:14px; border-radius:8px; overflow:auto; }}
.flow {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin:14px 0 22px; }}
.step {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; min-height:132px; }}
.step b {{ display:block; margin-bottom:6px; }}
.callout {{ border-left:4px solid var(--accent); background:#fff8ef; padding:12px 14px; border-radius:6px; margin:16px 0; }}
ul {{ padding-left:21px; }}
li {{ margin:5px 0; }}
a {{ color:#8e2c25; }}
@media (max-width:900px) {{
  main {{ padding:22px 14px 56px; }}
  h1 {{ font-size:27px; }}
  .grid,.flow {{ grid-template-columns:1fr; }}
  table {{ display:block; overflow-x:auto; }}
}}
</style>
</head>
<body><main>
<h1>Cosy Clone vs Fine-tune 技術報告 v1</h1>
<div class="meta">產生時間：{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ｜ 範圍：CosyVoice2 zero-shot clone、speaker cache、正式 speaker fine-tune、我們目前做到哪裡</div>
<p class="lead">短版結論：現在網頁/app 的 Cosy 聲音主要是 <b>zero-shot clone</b>，不是權重已更新的 fine-tune。Clone 靠幾秒 reference 直接生成，速度快、成本低，但穩定性吃 reference 品質；fine-tune 是把資料做成訓練集後真的更新模型權重，成本高很多，但才有機會讓固定角色聲線更穩。</p>

<section class="grid">
  <div class="card"><div class="label">目前灰原 app 聲音</div><div class="big">4.5s ref</div><div>CosyVoice2 zero-shot；不是 fine-tune。</div></div>
  <div class="card"><div class="label">目前灰原可訓練資料</div><div class="big">{fmt_num(haibara.get("duration_seconds_total"), "s")}</div><div>{haibara.get("num_clips_total", 0)} 句，train {haibara.get("num_train", 0)} / dev {haibara.get("num_dev", 0)}。</div></div>
  <div class="card"><div class="label">CosyVoice2 base 目錄</div><div class="big">{du("external/CosyVoice/pretrained_models/CosyVoice2-0.5B")}</div><div>本機模型權重與 ONNX/JIT 資源。</div></div>
  <div class="card"><div class="label">目前真正 SFT 狀態</div><div class="big warn">未完成</div><div>資料前處理完成；本機缺 CUDA。</div></div>
</section>

<h2>Cosy clone 要多久</h2>
<p>以下是本機已跑過的實測，不含人挑 reference 的時間。若 server 已常駐、模型已載入，才接近這些數字；第一次啟動會先載入 4.5GB 模型，通常是分鐘級。</p>
{table(["情境", "reference", "輸出句數", "min", "median", "max", "備註"], [
    ["灰原/柯南/博士授權角色 audition", "10-15 秒 pack 或 4.5 秒短 ref", str(t_haibara["count"]), fmt_num(t_haibara["min"], "s"), fmt_num(t_haibara["median"], "s"), fmt_num(t_haibara["max"], "s"), "角色片段比較長，句子也偏長；median 約 12.2 秒/句。"],
    ["台灣女聲 Cosy sweep", "7-11 秒女聲 ref", str(t_taiwan["count"]), fmt_num(t_taiwan["min"], "s"), fmt_num(t_taiwan["median"], "s"), fmt_num(t_taiwan["max"], "s"), "你後來選 golden teacher 的那條線；median 約 6.2 秒/句。"],
    ["原始模型 audition v4 Cosy", "7.73 秒 pack_best2", str(t_original["count"]), fmt_num(t_original["min"], "s"), fmt_num(t_original["median"], "s"), fmt_num(t_original["max"], "s"), "多模型原始 clone 比較；median 約 7.5 秒/句。"],
])}
<div class="callout"><b>實務估算：</b>Mac server 給家人玩，短句通常會是 6-15 秒一則；GPU server 會快很多。若要很多人同時玩，需要 queue，因為同一顆 GPU/CPU 同時跑多個 TTS 會互相搶資源，延遲會拉長。</div>

<h2>Clone、Speaker Cache、Fine-tune 差別</h2>
{table(["方法", "是否訓練", "吃多少資料", "生成時做什麼", "優點", "缺點", "適合現在嗎"], [
    ["Zero-shot clone", "否", "3-15 秒 reference + 正確逐字稿", "每次用 prompt wav / prompt text 條件化產生新語音", "最快開始；換角色容易；目前灰原就是這個", "可能把 reference 裡的雜音、BGM、奇怪開頭、語氣污染帶進輸出；同角色穩定度有限", "<b class='ok'>適合 demo/server</b>"],
    ["Speaker cache / add_zero_shot_spk", "否", "同 zero-shot", "先把 reference 的 speaker info 存起來，生成時少做一部分 reference 前處理", "比較快、比較穩定管理固定角色；不改權重", "音色能力仍是 zero-shot，不會學到更多資料", "<b class='ok'>很適合下一步</b>"],
    ["Cosy speaker fine-tune / SFT", "是", "至少 10-30 分鐘乾淨單人語音；1-3 小時更穩", "用文字/音檔對更新 llm/flow，必要時也調 vocoder", "固定聲音更穩；reference 污染較少；可以學語尾/口音分佈", "需要 CUDA GPU、準確逐字稿、資料清洗；過少資料會 overfit 或發音壞掉", "<b class='warn'>值得做，但要補資料/GPU</b>"],
    ["Teacher-student 蒸餾到 Matcha/ZipVoice", "是", "通常數百到數千句 teacher wav", "讓小模型學 Cosy/Index/Qwen 生成出的聲音", "有機會跑手機；模型可小很多", "不是 Cosy 本體 fine-tune；自然度和乾淨度可能掉", "<b class='ok'>手機路線</b>"],
])}

<h2>現在我們的 Cosy 狀態</h2>
<ul>
  <li><b>灰原 web/app：</b>使用 <code>haibara_002</code> 4.5 秒 reference，CosyVoice2 zero-shot。這是為了避開 best3 裡「給打開」污染。</li>
  <li><b>台灣溫柔女聲：</b>使用你選中的 <code>clear_best2_7s / line_04</code> golden teacher reference，約 7.73 秒 reference。</li>
  <li><b>網頁已產出音檔：</b><code>{generated}</code> 目前有 {generated_count} 個 generated wav。</li>
  <li><b>灰原 SFT 前處理：</b>已完成資料格式、speaker embedding、speech token、parquet、GPU run script；尚未在 CUDA 主機上跑出 checkpoint。</li>
</ul>

<h2>認真 fine-tune 怎麼做</h2>
<div class="flow">
  <div class="step"><b>1. 收資料</b>同一角色、同一語言/腔調、乾淨無 BGM、無旁白、無重疊人聲。每句 2-10 秒最舒服。</div>
  <div class="step"><b>2. 切句</b>VAD 或手切成短 utterance；頭尾靜音留少量，不要把前後別人的聲音切進來。</div>
  <div class="step"><b>3. 音訊清理</b>統一 sample rate、單聲道、響度；移除唱歌、尖叫、混響太重、音樂太大、OS/旁白片段。</div>
  <div class="step"><b>4. 逐字稿</b>ASR 只是草稿，最後要人工校正。文字錯，模型會照錯誤 alignment 學。</div>
  <div class="step"><b>5. Cosy 特徵</b>建立 <code>wav.scp/text/utt2spk/spk2utt</code>，抽 CampPlus speaker embedding 和 speech token。</div>
  <div class="step"><b>6. SFT 訓練</b>打包 parquet 後從 base checkpoint fine-tune。先短跑驗證 loss，再正式跑多 epoch。</div>
</div>

<h3>官方 CosyVoice recipe 對應</h3>
<p>本地 repo 的官方範例在 <code>external/CosyVoice/examples/magicdata-read/cosyvoice/run.sh</code>。核心流程如下：</p>
<pre>stage 0: prepare wav.scp / text / utt2spk / spk2utt
stage 1: tools/extract_embedding.py      # CampPlus speaker embedding
stage 2: tools/extract_speech_token.py   # discrete speech tokens
stage 3: tools/make_parquet_list.py      # Cosy training parquet
stage 5: cosyvoice/bin/train.py          # train llm / flow / hifigan from pretrained checkpoints
stage 6: average_model.py + export_jit.py + export_onnx.py</pre>

<h3>我們已經替灰原做好的檔案</h3>
{table(["項目", "路徑 / 值"], [
    ["workspace", "<code>/Users/ader/Documents/App/distillation/conan_authorized_voice_refs_v1/cosy_speaker_finetune_v1</code>"],
    ["dataset size", f"{haibara.get('num_clips_total', 0)} clips / {fmt_num(haibara.get('duration_seconds_total'), 's')} total / train {fmt_num(haibara.get('duration_seconds_train'), 's')}"],
    ["train data", "<code>data/train/wav.scp</code>, <code>data/train/text</code>, <code>data/train/utt2spk</code>, <code>data/train/utt2speech_token.pt</code>"],
    ["dev data", "<code>data/dev/wav.scp</code>, <code>data/dev/text</code>, <code>data/dev/utt2spk</code>, <code>data/dev/utt2speech_token.pt</code>"],
    ["parquet", "<code>parquet/train/data.list</code>, <code>parquet/dev/data.list</code>"],
    ["SFT config", "<code>cosyvoice2_haibara_sft.yaml</code>"],
    ["GPU script", "<code>run_on_cuda_gpu.sh</code>"],
])}

<h2>資料量要多少才認真</h2>
{table(["資料量", "可做什麼", "風險", "我對目前專案的判斷"], [
    ["3-15 秒", "zero-shot clone reference", "很吃 reference 品質；不能學更多資料", "目前灰原 demo 已可用，但穩定性上限在這。"],
    ["25 秒", "pipeline smoke test、overfit 實驗", "太少，容易只記住幾句；任意輸入時可能壞", "目前灰原資料量就在這一段，不建議宣稱正式 SFT。"],
    ["10-30 分鐘", "單 speaker SFT 起跑線", "仍要逐字稿乾淨；情緒/語境覆蓋不足會偏", "如果要灰原/角色聲線穩，這是第一個務實目標。"],
    ["1-3 小時", "比較穩的固定聲音 fine-tune", "清洗成本高；授權和資料一致性要確定", "若要公開給多人玩，建議往這裡靠。"],
    ["10+ 小時", "更完整多情緒/多語境 speaker model", "成本高，但泛化最好", "除非要產品化，不一定第一階段就需要。"],
])}

<h2>Fine-tune 後會變多快嗎</h2>
<p>通常 <b>fine-tune 主要改善像不像、穩不穩、乾不乾淨，不保證生成速度大幅變快</b>。速度主要由模型架構、runtime、硬體、streaming、vocoder、是否 cache speaker info 決定。</p>
<ul>
  <li>Cosy 本體 fine-tune 後，模型大小大致仍是 CosyVoice2 等級，不會因為只訓一個聲音就自動變手機小模型。</li>
  <li>若只要 server 生成，Fine-tune + speaker cache + GPU 常駐會改善體感。</li>
  <li>若要手機離線，還是要再走 Matcha/ZipVoice/其他小模型蒸餾，或找原生 mobile-friendly clone TTS。</li>
</ul>

<h2>建議路線</h2>
<ol>
  <li><b>短期：</b>Cosy server 保持 zero-shot，但把每個角色改成 speaker cache，reference 固定、避免每次重抽特徵。</li>
  <li><b>灰原品質：</b>繼續補乾淨授權片段，目標先到 10-30 分鐘；每句人工校正逐字稿。</li>
  <li><b>正式 SFT：</b>租 CUDA GPU 跑 Cosy llm/flow SFT；先 1 小時內 smoke，再正式多 epoch。</li>
  <li><b>手機：</b>拿 SFT 後最穩的 Cosy 當 teacher，重新蒸餾到 Matcha/ZipVoice；不要直接拿目前 25 秒資料硬塞小模型。</li>
</ol>

<h2>相關報告</h2>
<ul>
  <li><a href="../../clone_data_inventory_v1/index.html">Clone / teacher / student 資料量盤點</a></li>
  <li><a href="../../haibara_cosy_speaker_finetune_v1/haibara_cosy_speaker_finetune_v1.html">灰原 CosyVoice2 Speaker Fine-tune 報告 v1</a></li>
  <li><a href="../../conan_authorized_cosy_clone_audition_v1/conan_authorized_cosy_clone_audition_v1_standalone.html">授權角色 Cosy clone audition</a></li>
  <li><a href="../../../../taiwan_mandarin_low_r/datasets/downloads_female_voice/cosy_zipvoice_distill_v1/cosy_vs_zipvoice_vs_distill_standalone.html">台灣女聲 Cosy / ZipVoice / distill 對比</a></li>
</ul>

<h2>目前灰原訓練資料逐句</h2>
{table(["ID", "長度", "修正版文字", "原 ASR"], clip_rows)}

<h2>我會怎麼判斷 fine-tune 成功</h2>
<ul>
  <li><b>內容正確：</b>Whisper/ASR similarity 不應比 zero-shot 更差，不能多出「真的」「給打開」這種污染字。</li>
  <li><b>音色相似：</b>同一組句子對比 reference speaker embedding，並且真人聽感確認。</li>
  <li><b>乾淨度：</b>背景噪、電子雜音、BGM 殘影要比 zero-shot 更少。</li>
  <li><b>穩定度：</b>短句、長句、日常句、情緒句都能維持同一聲線。</li>
  <li><b>部署成本：</b>列模型大小、RSS/VRAM、首次載入時間、單句生成時間。</li>
</ul>

<div class="callout"><b>一句話：</b>現在的 Cosy clone 是「拿幾秒聲音當 prompt 去講新句子」；認真 fine-tune 是「把很多句乾淨聲音變成訓練集，真的更新模型權重」。我們已經把灰原資料做完前處理，但 25 秒太少，加上本機沒有 CUDA，所以還沒產出真正 fine-tuned Cosy checkpoint。</div>
</main></body></html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

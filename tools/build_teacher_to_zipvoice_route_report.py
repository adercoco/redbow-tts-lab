from __future__ import annotations

import base64
import html
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path("/Users/ader/Documents/App")
REPORT_DIR = ROOT / "distillation/taiwan_mandarin_low_r/reports/teacher_to_zipvoice_route_v1"
GENERATED = REPORT_DIR / "generated"
PROCESSED = REPORT_DIR / "processed"
OUT = REPORT_DIR / "teacher_to_zipvoice_route_v1_standalone.html"
RAW_REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"
REF_AUDIO = PROCESSED / "reference/pack_best2_7s.wav"

DISPLAY_TEXT = {
    "daily_01": "等一下我先把資料整理好，晚點再跟你確認一次。",
    "daily_02": "今天先不要想太多，回家路上買杯熱的，慢慢來就好。",
    "daily_03": "你剛剛那句我有聽到，我覺得可以再溫柔一點說。",
}

MODEL_INFO = {
    "qwen3_0p6b_base": {
        "name": "Qwen 0.6B Base clone",
        "role": "小一點的 Qwen reference clone 老師候選",
        "size": "1.6GB cache",
        "contents": "LLM/TTS acoustic generator + speech tokenizer + speaker conditioning；吃 ref_audio/ref_text。",
    },
    "qwen3_1p7b_base": {
        "name": "Qwen 1.7B Base clone",
        "role": "較大的 Qwen reference clone 老師候選",
        "size": "2.2GB cache",
        "contents": "同 Qwen Base clone 架構，參數較大；用來驗證 1.7B 是否比 0.6B 更穩。",
    },
    "cosyvoice2_0p5b": {
        "name": "CosyVoice2 clone",
        "role": "目前聲音自然度最穩的 zero-shot teacher",
        "size": "4.5GB repo",
        "contents": "LLM semantic path + flow-matching acoustic model + vocoder + speaker embedding/tokenizer。",
    },
    "zipvoice_direct_original_16step": {
        "name": "Direct ZipVoice 16-step",
        "role": "不先做 teacher corpus，直接拿官方 ZipVoice zero-shot clone",
        "size": "175.8MB runtime core",
        "contents": "text_encoder_int8.onnx + fm_decoder_int8.onnx + Vocos vocoder；16-step flow sampling。",
    },
    "zipvoice_cosy_student_16step": {
        "name": "ZipVoice Cosy student 16-step",
        "role": "ZipVoice 學 Cosy teacher corpus 後的完整步數學生模型",
        "size": "175.8MB runtime core",
        "contents": "同 ZipVoice 架構，權重經 Cosy teacher corpus fine-tune；仍用 16-step 看學生上限。",
    },
    "zipvoice_cosy_student_8step": {
        "name": "ZipVoice Cosy student 8-step",
        "role": "同一顆 Cosy student，把推論步數降到 8",
        "size": "175.8MB runtime core",
        "contents": "同 ZipVoice Cosy student checkpoint；用 8-step flow sampling 聽速度/品質折衷。",
    },
    "zipvoice_cosy_student_4step": {
        "name": "ZipVoice Cosy student 4-step",
        "role": "同一顆 Cosy student，把推論步數降到 4",
        "size": "175.8MB runtime core",
        "contents": "同 ZipVoice Cosy student checkpoint；用 4-step flow sampling 測手機速度候選，但咬字與乾淨度風險較高。",
    },
    "zipvoice_qwen17_clone_mimic_16step": {
        "name": "ZipVoice mimics Qwen 1.7B 16-step",
        "role": "ZipVoice zero-shot 模仿新的 Qwen 1.7B Base clone 聲音",
        "size": "175.8MB runtime core",
        "contents": "用報告中的 Qwen 1.7B Base clone 音檔當 ZipVoice prompt/reference，再用官方 ZipVoice ONNX/int8 16-step 生成；這是 prompt mimic，不是完整 fine-tune。",
    },
    "zipvoice_qwen17_clone_mimic_8step": {
        "name": "ZipVoice mimics Qwen 1.7B 8-step",
        "role": "同一個 Qwen 1.7B clone prompt，把 ZipVoice 推論步數降到 8",
        "size": "175.8MB runtime core",
        "contents": "同官方 ZipVoice ONNX/int8 + Qwen 1.7B Base clone prompt；8-step 是速度/品質折衷，不是另訓的小模型。",
    },
    "zipvoice_qwen17_clone_mimic_4step": {
        "name": "ZipVoice mimics Qwen 1.7B 4-step",
        "role": "同一個 Qwen 1.7B clone prompt，把 ZipVoice 推論步數降到 4",
        "size": "175.8MB runtime core",
        "contents": "同官方 ZipVoice ONNX/int8 + Qwen 1.7B Base clone prompt；4-step 是手機速度候選，但容易犧牲乾淨度與咬字。",
    },
}

ORDER = [
    "qwen3_0p6b_base",
    "qwen3_1p7b_base",
    "cosyvoice2_0p5b",
    "zipvoice_direct_original_16step",
    "zipvoice_cosy_student_16step",
    "zipvoice_cosy_student_8step",
    "zipvoice_cosy_student_4step",
    "zipvoice_qwen17_clone_mimic_16step",
    "zipvoice_qwen17_clone_mimic_8step",
    "zipvoice_qwen17_clone_mimic_4step",
]


def load_rows() -> list[dict]:
    rows = []
    for name in ["qwen_results.json", "cosy_results.json", "zipvoice_results.json"]:
        path = PROCESSED / name
        if not path.exists():
            path = GENERATED / name
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def audio_src(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def audio(path: Path) -> str:
    return f'<audio controls preload="metadata" src="{audio_src(path)}"></audio>'


def fmt_s(value: float) -> str:
    return f"{value:.2f}s"


def fmt_gb(mb: float) -> str:
    return f"{mb / 1024:.2f}GB"


def summarize(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for key in ORDER:
        group = [row for row in rows if row["family"] == key]
        out[key] = {
            "avg_wall_s": statistics.mean(row["wall_s"] for row in group),
            "avg_audio_s": statistics.mean(row["audio_s"] for row in group),
            "avg_rtf": statistics.mean(row["rtf"] for row in group),
            "peak_rss_mb": max(row.get("peak_rss_mb", 0.0) for row in group),
            "load_s": group[0].get("load_s", 0.0),
            "load_peak_rss_mb": group[0].get("load_peak_rss_mb", 0.0),
            "steps": group[0].get("steps"),
        }
    return out


def metric_cards(summary: dict[str, dict], order: list[str]) -> str:
    cards = []
    for key in order:
        info = MODEL_INFO[key]
        item = summary[key]
        cards.append(
            f"""
            <article class="metric-card">
              <div class="metric-name">{html.escape(info["name"])}</div>
              <p>{html.escape(info["role"])}</p>
              <dl>
                <div><dt>大小</dt><dd>{html.escape(info["size"])}</dd></div>
                <div><dt>一句話生成</dt><dd>{fmt_s(item["avg_wall_s"])}</dd></div>
                <div><dt>Peak RSS</dt><dd>{fmt_gb(item["peak_rss_mb"])}</dd></div>
                <div><dt>RTF</dt><dd>{item["avg_rtf"]:.2f}</dd></div>
              </dl>
            </article>
            """
        )
    return "\n".join(cards)


def summary_table(summary: dict[str, dict], order: list[str]) -> str:
    rows = []
    for key in order:
        info = MODEL_INFO[key]
        item = summary[key]
        steps = item["steps"]
        rows.append(
            f"""
            <tr>
              <td><b>{html.escape(info["name"])}</b><small>{html.escape(info["role"])}</small></td>
              <td>{html.escape(info["size"])}</td>
              <td>{fmt_gb(item["peak_rss_mb"])}<small>本輪實測 peak RSS</small></td>
              <td>{fmt_s(item["avg_wall_s"])}<small>平均音檔 {item["avg_audio_s"]:.2f}s；RTF {item["avg_rtf"]:.2f}</small></td>
              <td>{steps if steps else "clone"}<small>{ "flow steps" if steps else "reference clone" }</small></td>
            </tr>
            """
        )
    return "\n".join(rows)


def listen_table(rows: list[dict], order: list[str]) -> str:
    by_key_sample = {(row["family"], row["sample_id"]): row for row in rows}
    sections = []
    for sample_id, text in DISPLAY_TEXT.items():
        cards = []
        for key in order:
            row = by_key_sample[(key, sample_id)]
            cards.append(
                f"""
                <article class="listen-card">
                  <h3>{html.escape(MODEL_INFO[key]["name"])}</h3>
                  {audio(Path(row["output"]))}
                  <small>{fmt_s(row["wall_s"])} / audio {row["audio_s"]:.2f}s · {row.get("steps") or "clone"}{ " steps" if row.get("steps") else "" }</small>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="listen-row">
              <div class="line-title">
                <span>{html.escape(sample_id)}</span>
                <h3>{html.escape(text)}</h3>
              </div>
              <div class="listen-grid">{''.join(cards)}</div>
            </section>
            """
        )
    return "\n".join(sections)


def postprocess_table(rows: list[dict]) -> str:
    grouped = []
    for key in ORDER:
        group = [row for row in rows if row["family"] == key and row.get("postprocess")]
        if not group:
            continue
        before = statistics.mean(row["postprocess"]["before_rms_db"] for row in group)
        after = statistics.mean(row["postprocess"]["after_rms_db"] for row in group)
        peak = max(row["postprocess"]["after_peak_db"] for row in group)
        grouped.append(
            f"""
            <tr>
              <td><b>{html.escape(MODEL_INFO[key]["name"])}</b></td>
              <td>{before:.1f} dBFS</td>
              <td>{after:.1f} dBFS</td>
              <td>{peak:.1f} dBFS</td>
              <td>高通 + STFT mild clean + soft gate + RMS 對齊 + peak limit</td>
            </tr>
            """
        )
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr><th>模型</th><th>原始平均 RMS</th><th>處理後平均 RMS</th><th>處理後最高 peak</th><th>處理鏈</th></tr></thead>
        <tbody>{''.join(grouped)}</tbody>
      </table>
    </div>
    """


def contents_cards(order: list[str]) -> str:
    cards = []
    for key in order:
        info = MODEL_INFO[key]
        cards.append(
            f"""
            <article class="content-card">
              <h3>{html.escape(info["name"])}</h3>
              <p>{html.escape(info["contents"])}</p>
            </article>
            """
        )
    return "\n".join(cards)


def build() -> str:
    rows = load_rows()
    summary = summarize(rows)
    generated = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    ref_text = "所以我當時就說，我想要做一張療癒人的專輯。開始當然就是我們的提案會議，我就提出了因為多年。"
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>同一 Reference 到 ZipVoice 蒸餾主線報告</title>
<style>
:root {{
  --paper:#f5f1e9; --panel:#fffdf7; --ink:#25231f; --muted:#6f685e;
  --line:#ddd4c5; --line2:#c8bda9; --red:#b7202f; --green:#52685a; --blue:#455d73; --gold:#8a6f3a;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:14px/1.54 -apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif; letter-spacing:0; }}
main {{ width:min(1360px, calc(100vw - 48px)); margin:0 auto; padding:30px 0 56px; }}
header {{ border-bottom:1px solid var(--line2); padding-bottom:16px; margin-bottom:16px; }}
.eyebrow {{ color:var(--red); font-weight:780; font-size:12px; margin-bottom:6px; }}
h1 {{ margin:0 0 8px; font-size:34px; line-height:1.1; letter-spacing:0; }}
h2 {{ margin:28px 0 10px; font-size:22px; letter-spacing:0; }}
h3 {{ margin:0 0 6px; font-size:16px; }}
p {{ margin:0; }}
.lead {{ max-width:1080px; color:var(--muted); font-size:16px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; }}
audio {{ width:100%; height:30px; }}
small {{ display:block; color:var(--muted); font-size:11px; margin-top:3px; }}
.note {{ background:#fff8e8; border:1px solid #ead8b7; border-radius:8px; padding:10px 12px; margin:10px 0; }}
.flow {{ display:grid; grid-template-columns:1fr 2.2fr 1.2fr 1.2fr; gap:10px; align-items:stretch; margin:16px 0; }}
.node {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; position:relative; min-height:76px; }}
.node strong {{ display:block; font-size:16px; margin-bottom:6px; }}
.node p {{ color:var(--muted); font-size:12px; }}
.node.ref {{ border-top:4px solid var(--red); }}
.node.teacher {{ border-top:4px solid var(--blue); }}
.node.student {{ border-top:4px solid var(--green); }}
.node.distill {{ border-top:4px solid var(--gold); }}
.split {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
.arrow-label {{ text-align:center; color:var(--muted); font-size:13px; margin-top:-12px; margin-bottom:8px; }}
.metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; }}
.metric-card,.content-card {{ background:rgba(255,253,247,.86); border:1px solid var(--line); border-radius:8px; padding:10px; }}
.metric-name {{ color:var(--red); font-weight:820; font-size:16px; margin-bottom:6px; }}
.metric-card p,.content-card p {{ color:var(--muted); font-size:13px; }}
dl {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 10px; margin:8px 0 0; }}
dt {{ color:var(--muted); font-size:11px; }}
dd {{ margin:0; font-weight:760; }}
table {{ border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); }}
th,td {{ border-bottom:1px solid var(--line); border-right:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
th {{ background:#eee5d8; color:#3a352e; }}
tr:last-child th,tr:last-child td {{ border-bottom:0; }}
td:last-child,th:last-child {{ border-right:0; }}
.table-wrap {{ border-radius:8px; overflow:hidden; }}
.listen-row {{ background:rgba(255,253,247,.72); border:1px solid var(--line); border-radius:8px; padding:10px; margin:10px 0; }}
.line-title {{ display:flex; gap:10px; align-items:flex-start; border-bottom:1px solid var(--line); padding-bottom:7px; margin-bottom:8px; }}
.line-title span {{ flex:0 0 auto; color:var(--red); font-weight:820; font-size:12px; padding-top:2px; }}
.line-title h3 {{ margin:0; font-size:16px; line-height:1.35; }}
.listen-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }}
.listen-card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px; min-width:0; }}
.listen-card h3 {{ font-size:14px; line-height:1.35; color:#3b352e; }}
.content-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }}
.teach {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.teach article {{ background:rgba(255,253,247,.86); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.teach ol {{ padding-left:20px; margin:8px 0 0; }}
.teach li {{ margin:6px 0; }}
.detail {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
.detail article {{ background:rgba(255,253,247,.86); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.detail p {{ color:var(--muted); }}
.formula {{ display:block; margin:7px 0; padding:8px 10px; background:#f0e7d9; border-radius:8px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; color:#332d25; overflow:auto; }}
.mono {{ background:#2a2925; color:#f7f0e4; border-radius:8px; padding:10px; overflow:auto; font-size:12px; line-height:1.45; }}
@media (max-width:1000px) {{ main {{ width:min(100vw - 28px,1360px); }} .flow,.metrics,.listen-grid,.content-grid,.teach,.detail {{ grid-template-columns:1fr; }} .split {{ grid-template-columns:1fr 1fr; }} h1 {{ font-size:29px; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Red Bow TTS · generated {generated} · desktop single-file HTML</div>
    <h1>同一份授權女聲 Reference → Teacher Clone → ZipVoice → ZipVoice 蒸餾</h1>
    <p class="lead">這份只看你現在要走的主線：同一份授權女聲 reference 先丟給 Qwen 0.6B Base、Qwen 1.7B Base、CosyVoice2、Direct ZipVoice；再看 ZipVoice 學 Cosy 的 16/8/4-step 階梯，以及 ZipVoice 用新的 Qwen 1.7B Base clone 音檔當 prompt 來模仿的 16/8/4-step 階梯。</p>
  </header>

  <section class="note">
    <b>Reference audio</b>：<code>pack_best2_7s.wav</code>。Qwen Base / Cosy / Direct ZipVoice / Cosy→ZipVoice 用這份授權女聲 reference。新的 Qwen→ZipVoice mimic 區塊則使用本報告生成的 <code>Qwen 1.7B Base clone / daily_01</code> 當 ZipVoice prompt，不再使用舊 <code>taiwan_mandarin_low_r</code> VoiceDesign corpus。TTS 輸入用簡中等價句，頁面顯示繁中，目的是讓中文讀字穩定。
    <div style="margin-top:8px">{audio(REF_AUDIO)}<small>{html.escape(ref_text)}</small></div>
  </section>

  <section class="flow">
    <div class="node ref"><strong>1. 授權女聲 reference</strong><p>7.7 秒乾淨片段 + 逐字稿，作為所有 clone / zero-shot prompt 的共同起點。</p></div>
    <div class="node teacher"><strong>2. Teacher / direct clone 候選</strong><div class="split">
      <p>Qwen 0.6B Base clone</p><p>Qwen 1.7B Base clone</p><p>CosyVoice2 clone</p><p>Direct ZipVoice</p>
    </div></div>
    <div class="node student"><strong>3. ZipVoice student / mimic</strong><p>Cosy→ZipVoice 是學老師語料；Qwen→ZipVoice 這輪是用新 Qwen clone 音檔做 prompt mimic。</p></div>
    <div class="node distill"><strong>4. ZipVoice 步數階梯</strong><p>兩條都列 16-step、8-step、4-step，直接觀察速度/品質階梯。</p></div>
  </section>
  <div class="arrow-label">目標不是只聽單句像不像，而是決定哪個 teacher 最值得拿去做大量 corpus + ZipVoice student + few-step distillation。</div>

  <h2>一眼看懂量測總表</h2>
  <div class="metrics">{metric_cards(summary, ORDER)}</div>
  <div class="table-wrap" style="margin-top:14px">
    <table>
      <thead><tr><th>模型</th><th>模型大小</th><th>記憶體</th><th>一句話生成</th><th>生成方式</th></tr></thead>
      <tbody>{summary_table(summary, ORDER)}</tbody>
    </table>
  </div>

  <h2>三句日常句子並排試聽</h2>
  <section class="note"><b>試聽說明：</b>這裡播的是後處理版：高通去低頻、STFT mild clean、soft gate、RMS 對齊到約 -19 dBFS、peak limiter。生成秒數仍是模型原始推論秒數，不含後處理時間。</section>
  {listen_table(rows, ORDER)}

  <h2>後處理量測</h2>
  {postprocess_table(rows)}

  <h2>不同模型內容物分析</h2>
  <div class="content-grid">{contents_cards(ORDER)}</div>

  <h2>數學與模型細節</h2>
  <section class="detail">
    <article>
      <h3>1. 條件式 TTS / clone 的問題定義</h3>
      <p>每個 clone 模型本質上都在估計條件分布：給定文字 <code>x</code>、reference audio <code>r</code>、reference transcript <code>t_r</code>，生成 waveform <code>y</code>。</p>
      <span class="formula">pθ(y | x, r, t_r) = pθ(y | text tokens, speaker/prosody condition)</span>
      <p>Qwen Base 和 CosyVoice2 的差別不是任務不同，而是內部 representation 不同：Qwen 比較像 audio-native LLM/TTS generator；CosyVoice2 把 semantic token、speaker embedding、flow acoustic model 和 vocoder 分得更清楚。</p>
    </article>
    <article>
      <h3>2. Speaker condition 怎麼進模型</h3>
      <p>reference 不會被「直接貼到新音檔」。模型會從 reference 抽出 speaker / style condition，常見是 speaker embedding、speech token、prompt acoustic features，然後在 decoder attention、FiLM/AdaLN、cross-attention 或 prefix conditioning 中使用。</p>
      <span class="formula">c = Enc_spk(r, t_r),  y = Decθ(tokens(x), c)</span>
      <p>所以 ref_text 對齊很重要：如果 reference 說的內容和逐字稿不一致，模型會把錯誤對齊學成口音、停頓或亂字。</p>
    </article>
    <article>
      <h3>3. ZipVoice / flow matching</h3>
      <p>ZipVoice 不是傳統 autoregressive 一點一點吐 waveform，而是從雜訊或簡單分布出發，沿著 flow trajectory 走到 acoustic features。模型學的是 velocity field。</p>
      <span class="formula">dx_t / dt = vθ(x_t, t, c)</span>
      <span class="formula">L_flow = E[ || vθ(x_t, t, c) - u_t ||² ]</span>
      <p>推論時的 steps 就是 ODE solver 的離散步數。16-step 代表用 16 次修正把 acoustic feature 從噪聲推到語音；4-step 代表每一步要走更大，速度快，但模型必須真的學會大步走，不然聲音會糊或電子化。</p>
    </article>
    <article>
      <h3>4. Teacher → ZipVoice student</h3>
      <p>資料蒸餾先不碰 teacher 內部 logits。它把大模型 teacher 產出的高品質 waveform 當 synthetic label，訓練 ZipVoice 學同一個文字到同一音色的 mapping。本報告現在同時保留 Cosy→ZipVoice 與 Qwen→ZipVoice 兩條 student branch。</p>
      <span class="formula">D_teacher = {{(x_i, y_i^T)}};  minimize L(ZipVoice(x_i), y_i^T)</span>
      <p>實作上就是先產 500-5000 句 teacher wav，做 train/dev split，轉 TSV、token、fbank、cuts manifest，再從官方 ZipVoice checkpoint fine-tune。</p>
    </article>
    <article>
      <h3>5. ZipVoice → ZipVoice few-step 蒸餾</h3>
      <p>few-step 蒸餾的目標不是換音色，而是讓少步數 student 逼近多步數 teacher。Qwen→ZipVoice 現在補了 16-step、8-step、4-step 三層：16-step 看上限，8-step 看折衷，4-step 看手機速度候選。</p>
      <span class="formula">y^T = Solver_16(v_teacher, x, c)</span>
      <span class="formula">y^S = Solver_4(v_student, x, c)</span>
      <span class="formula">L = λ_mel ||Mel(y^S)-Mel(y^T)||₁ + λ_flow L_flow + λ_spk (1-cos(e_S,e_T))</span>
      <p>如果只在推論時把 16-step 硬改 4-step，通常會壞；真正蒸餾是讓 student 在訓練中習慣 4 個大步。</p>
    </article>
    <article>
      <h3>6. Vocos / vocoder 的角色</h3>
      <p>ZipVoice 前半通常產 acoustic feature 或 mel-like representation，vocoder 再把它轉成 waveform。手機端常見瓶頸不只在 decoder，也在 vocoder。</p>
      <span class="formula">ŷ = Vocoderφ(acoustic_features)</span>
      <p>所以 int8 Vocos、Core ML / NNAPI delegate、chunked vocoder、streaming playback 都會影響實際體感延遲。</p>
    </article>
    <article>
      <h3>7. 後處理 DSP 數學</h3>
      <p>本報告的後處理是保守清理，不改模型內容。處理鏈是 DC removal、高通、STFT mild spectral subtraction、soft gate、RMS matching、peak limiting。</p>
      <span class="formula">x₁[n] = HighPass(x[n] - mean(x), f_c=70Hz)</span>
      <span class="formula">X = STFT(x₁);  |X_clean| = max(|X| - αN, β|X|)</span>
      <span class="formula">g = 10^((target_dB - rms_dB(x))/20);  y = limiter(g · ISTFT(X_clean))</span>
      <p>這能讓音量更一致、低頻更乾淨、底噪少一點；但不能修正 TTS 念錯字，也不能把嚴重電子雜訊變成自然語音。</p>
    </article>
    <article>
      <h3>8. 怎麼判斷下一步</h3>
      <p>工程上不要只看「模型大不大」。要同時看 speaker similarity、ASR 字錯率、RTF、Peak RSS、手機 runtime、reference cache 後的 per-sentence latency。</p>
      <span class="formula">score = w₁·speaker + w₂·ASR - w₃·RTF - w₄·RSS - w₅·artifact</span>
      <p>如果 Qwen 0.6B 聲音接近 1.7B，就不值得用 1.7B 當長期 teacher；如果 Cosy 明顯更自然，就應該用 Cosy 擴 corpus，再讓 ZipVoice/Matcha 學。</p>
    </article>
  </section>

  <h2>實作教學：每一步怎麼做</h2>
  <section class="teach">
    <article>
      <h3>Step 1：準備 reference</h3>
      <ol>
        <li>裁 5-10 秒乾淨授權女聲，避開男聲、配樂、唱歌、重疊說話。</li>
        <li>補逐字稿，reference audio 和 ref_text 要對得上；錯字會直接污染 clone。</li>
        <li>可做輕微 high-pass、RMS normalize，但不要把音色壓扁；reference 只修乾淨度，不做誇張 EQ。</li>
      </ol>
      <pre class="mono">ref_audio = pack_best2_7s.wav
ref_text  = 對齊的逐字稿
target    = 三句日常中文測試句</pre>
    </article>
    <article>
      <h3>Step 2：Qwen / Cosy 做 teacher clone</h3>
      <ol>
        <li>Qwen Base 使用 <code>ref_audio + ref_text + text</code>，這才是真 clone；VoiceDesign 是文字設計聲音，不等於 clone。</li>
        <li>CosyVoice2 使用 zero-shot inference，把 reference 的 speaker/prosody condition 到新文字。</li>
        <li>先用 3-12 句 audition 篩 teacher，再決定是否擴成 500-5000 句 corpus；擴 corpus 前要先固定 reference 和 postprocess policy。</li>
      </ol>
      <pre class="mono">generate_audio(
  model=Qwen3-TTS-Base,
  ref_audio=ref_audio,
  ref_text=ref_text,
  text=target_text
)</pre>
    </article>
    <article>
      <h3>Step 3：Direct ZipVoice baseline</h3>
      <ol>
        <li>不訓練，直接用官方 ZipVoice ONNX/int8 讀同一 reference。</li>
        <li>這是手機可行性的 baseline：小、可 ONNX，但聲音不一定像。</li>
        <li>16-step 是品質檢查；低 step 只看速度時容易犧牲咬字與乾淨度。本報告改聽比較穩的 4-step。</li>
      </ol>
      <pre class="mono">prompt_wav  = ref_audio
prompt_text = ref_text
num_step    = 16
model       = zipvoice_original_onnx_int8</pre>
    </article>
    <article>
      <h3>Step 4：Teacher corpus → ZipVoice student</h3>
      <ol>
        <li>選最好的 teacher，產大量 <code>(text_i, wav_i)</code>，例如 500 句日常對話。</li>
        <li>整理成 ZipVoice TSV：<code>utt_id  text  wav_path</code>。</li>
        <li>跑 token / fbank / cuts manifest，再從官方 ZipVoice checkpoint fine-tune。</li>
      </ol>
      <pre class="mono">custom_train.tsv
custom_dev.tsv
prepare_tokens -> compute_fbank -> train_zipvoice --finetune</pre>
    </article>
    <article>
      <h3>Step 5：ZipVoice → ZipVoice 蒸餾</h3>
      <ol>
        <li>完整 16-step ZipVoice 是 teacher，4-step ZipVoice 是 student。</li>
        <li>數學上是在學同一條 flow trajectory 的短路徑：讓 4 個大步逼近 16 個小步的結果。</li>
        <li>這才是速度真正有機會變快的地方；單純改 inference steps 通常會壞聲音。</li>
      </ol>
      <pre class="mono">teacher: ZipVoice 16-step
student: ZipVoice 4-step
loss: acoustic/flow matching + text/audio reconstruction</pre>
    </article>
    <article>
      <h3>Step 6：手機部署檢查</h3>
      <ol>
        <li>匯出 ONNX/int8，搭 sherpa-onnx/C++ runtime。</li>
        <li>把 speaker/reference features cache 起來，使用者第一次載入模型後不要每句重算。</li>
        <li>app 端做 chunked generation + streaming playback，主觀等待時間會比整句完成後播放短。</li>
      </ol>
      <pre class="mono">load model once
cache prompt features
generate chunks
play while next chunk renders</pre>
    </article>
    <article>
      <h3>Step 7：後處理與音量一致</h3>
      <ol>
        <li>所有候選輸出跑同一套 DSP，避免「誰比較大聲誰比較好聽」的偏誤。</li>
        <li>本輪 target RMS 約 -19 dBFS，peak ceiling 約 -1 dBFS。</li>
        <li>報告中的生成秒數不含後處理；app 若要即時用，這段 DSP 要用 Accelerate/vDSP 或 C++ 實作。</li>
      </ol>
      <pre class="mono">dc remove -> high-pass 70Hz -> STFT clean
soft gate -> RMS match -> peak limiter</pre>
    </article>
  </section>

  <section class="note">
    <b>讀法：</b>先聽三句並排，選最像、最乾淨、最自然的 teacher。若 Qwen 0.6B 已接近 1.7B，下一步應優先把 Qwen 0.6B teacher corpus 擴大，再訓練 ZipVoice student；若 Cosy 仍明顯最好，就繼續 Cosy→ZipVoice→few-step。
  </section>
</main>
</body>
</html>"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

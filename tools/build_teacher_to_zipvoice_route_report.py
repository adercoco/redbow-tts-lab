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
OUT = REPORT_DIR / "teacher_to_zipvoice_route_v1_standalone.html"
REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"

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
        "name": "ZipVoice student 16-step",
        "role": "ZipVoice 學 Cosy teacher corpus 後的學生模型",
        "size": "175.8MB runtime core",
        "contents": "同 ZipVoice 架構，權重經 Cosy teacher corpus fine-tune；仍用 16-step 看學生上限。",
    },
    "zipvoice_qwen_fewstep_distilled_3step": {
        "name": "ZipVoice distilled 3-step",
        "role": "ZipVoice→ZipVoice few-step 蒸餾後的速度候選",
        "size": "175.8MB runtime core",
        "contents": "同 ZipVoice int8 架構；把 decoding steps 從 16 壓到 3，主要換速度。",
    },
}

ORDER = [
    "qwen3_0p6b_base",
    "qwen3_1p7b_base",
    "cosyvoice2_0p5b",
    "zipvoice_direct_original_16step",
    "zipvoice_cosy_student_16step",
    "zipvoice_qwen_fewstep_distilled_3step",
]


def load_rows() -> list[dict]:
    rows = []
    for name in ["qwen_results.json", "cosy_results.json", "zipvoice_results.json"]:
        rows.extend(json.loads((GENERATED / name).read_text(encoding="utf-8")))
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


def metric_cards(summary: dict[str, dict]) -> str:
    cards = []
    for key in ORDER:
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


def summary_table(summary: dict[str, dict]) -> str:
    rows = []
    for key in ORDER:
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


def listen_table(rows: list[dict]) -> str:
    by_key_sample = {(row["family"], row["sample_id"]): row for row in rows}
    tr = []
    for sample_id, text in DISPLAY_TEXT.items():
        cells = []
        for key in ORDER:
            row = by_key_sample[(key, sample_id)]
            cells.append(
                f"""
                <td>
                  {audio(Path(row["output"]))}
                  <small>{fmt_s(row["wall_s"])} / audio {row["audio_s"]:.2f}s</small>
                </td>
                """
            )
        tr.append(
            f"""
            <tr>
              <th><b>{html.escape(text)}</b><small>{html.escape(sample_id)}</small></th>
              {''.join(cells)}
            </tr>
            """
        )
    heads = "".join(f"<th>{html.escape(MODEL_INFO[key]['name'])}</th>" for key in ORDER)
    return f"""
    <div class="wide-table">
      <table class="listen">
        <thead><tr><th>日常句子</th>{heads}</tr></thead>
        <tbody>{''.join(tr)}</tbody>
      </table>
    </div>
    """


def contents_cards() -> str:
    cards = []
    for key in ORDER:
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
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.68 -apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif; letter-spacing:0; }}
main {{ width:min(1500px, calc(100vw - 56px)); margin:0 auto; padding:42px 0 80px; }}
header {{ border-bottom:1px solid var(--line2); padding-bottom:22px; margin-bottom:24px; }}
.eyebrow {{ color:var(--red); font-weight:780; font-size:13px; margin-bottom:8px; }}
h1 {{ margin:0 0 10px; font-size:38px; line-height:1.12; letter-spacing:0; }}
h2 {{ margin:42px 0 14px; font-size:25px; letter-spacing:0; }}
h3 {{ margin:0 0 8px; font-size:17px; }}
p {{ margin:0; }}
.lead {{ max-width:980px; color:var(--muted); font-size:18px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; }}
audio {{ width:100%; height:34px; }}
small {{ display:block; color:var(--muted); font-size:12px; margin-top:4px; }}
.note {{ background:#fff8e8; border:1px solid #ead8b7; border-radius:8px; padding:14px 16px; margin:16px 0; }}
.flow {{ display:grid; grid-template-columns:1fr 2.3fr 1.1fr 1.1fr; gap:12px; align-items:stretch; margin:24px 0; }}
.node {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; position:relative; min-height:94px; }}
.node strong {{ display:block; font-size:16px; margin-bottom:6px; }}
.node p {{ color:var(--muted); font-size:13px; }}
.node.ref {{ border-top:4px solid var(--red); }}
.node.teacher {{ border-top:4px solid var(--blue); }}
.node.student {{ border-top:4px solid var(--green); }}
.node.distill {{ border-top:4px solid var(--gold); }}
.split {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
.arrow-label {{ text-align:center; color:var(--muted); font-size:13px; margin-top:-12px; margin-bottom:8px; }}
.metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
.metric-card,.content-card {{ background:rgba(255,253,247,.86); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.metric-name {{ color:var(--red); font-weight:820; font-size:16px; margin-bottom:6px; }}
.metric-card p,.content-card p {{ color:var(--muted); font-size:13px; }}
dl {{ display:grid; grid-template-columns:1fr 1fr; gap:8px 12px; margin:12px 0 0; }}
dt {{ color:var(--muted); font-size:11px; }}
dd {{ margin:0; font-weight:760; }}
table {{ border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); }}
th,td {{ border-bottom:1px solid var(--line); border-right:1px solid var(--line); padding:11px 12px; text-align:left; vertical-align:top; }}
th {{ background:#eee5d8; color:#3a352e; }}
tr:last-child th,tr:last-child td {{ border-bottom:0; }}
td:last-child,th:last-child {{ border-right:0; }}
.wide-table {{ overflow-x:auto; border-radius:8px; }}
table.listen {{ min-width:1380px; }}
table.listen th:first-child {{ width:260px; }}
table.listen td {{ width:180px; }}
.content-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
.teach {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.teach article {{ background:rgba(255,253,247,.86); border:1px solid var(--line); border-radius:8px; padding:16px; }}
.teach ol {{ padding-left:20px; margin:8px 0 0; }}
.teach li {{ margin:6px 0; }}
.mono {{ background:#2a2925; color:#f7f0e4; border-radius:8px; padding:12px; overflow:auto; font-size:12px; line-height:1.55; }}
@media (max-width:1000px) {{ main {{ width:min(100vw - 28px,1500px); }} .flow,.metrics,.content-grid,.teach {{ grid-template-columns:1fr; }} .split {{ grid-template-columns:1fr 1fr; }} h1 {{ font-size:31px; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Red Bow TTS · generated {generated} · desktop single-file HTML</div>
    <h1>同一份授權女聲 Reference → Teacher Clone → ZipVoice → ZipVoice 蒸餾</h1>
    <p class="lead">這份只看你現在要走的路：同一份授權女聲 reference 先丟給 Qwen 0.6B Base、Qwen 1.7B Base、CosyVoice2、Direct ZipVoice；再看 ZipVoice 學 teacher 後，以及 ZipVoice few-step 蒸餾後的速度與聲音差異。</p>
  </header>

  <section class="note">
    <b>Reference audio</b>：<code>pack_best2_7s.wav</code>。所有本輪 audition 都用同一份 reference prompt；TTS 輸入用簡中等價句，頁面顯示繁中，目的是讓中文讀字穩定。
    <div style="margin-top:8px">{audio(REF_AUDIO)}<small>{html.escape(ref_text)}</small></div>
  </section>

  <section class="flow">
    <div class="node ref"><strong>1. 授權女聲 reference</strong><p>7.7 秒乾淨片段 + 逐字稿，作為所有 clone / zero-shot prompt 的共同起點。</p></div>
    <div class="node teacher"><strong>2. Teacher / direct clone 候選</strong><div class="split">
      <p>Qwen 0.6B Base clone</p><p>Qwen 1.7B Base clone</p><p>CosyVoice2 clone</p><p>Direct ZipVoice</p>
    </div></div>
    <div class="node student"><strong>3. ZipVoice student</strong><p>把選出的 teacher corpus 轉成 text/wav manifest、fbank/token，再 fine-tune ZipVoice。</p></div>
    <div class="node distill"><strong>4. ZipVoice 蒸餾</strong><p>用完整 step 的 ZipVoice 當老師，訓練 few-step/低步數學生，換取手機端速度。</p></div>
  </section>
  <div class="arrow-label">目標不是只聽單句像不像，而是決定哪個 teacher 最值得拿去做大量 corpus + ZipVoice student + few-step distillation。</div>

  <h2>一眼看懂量測總表</h2>
  <div class="metrics">{metric_cards(summary)}</div>
  <div class="wide-table" style="margin-top:14px">
    <table>
      <thead><tr><th>模型</th><th>模型大小</th><th>記憶體</th><th>一句話生成</th><th>生成方式</th></tr></thead>
      <tbody>{summary_table(summary)}</tbody>
    </table>
  </div>

  <h2>三句日常句子並排試聽</h2>
  {listen_table(rows)}

  <h2>不同模型內容物分析</h2>
  <div class="content-grid">{contents_cards()}</div>

  <h2>實作教學：每一步怎麼做</h2>
  <section class="teach">
    <article>
      <h3>Step 1：準備 reference</h3>
      <ol>
        <li>裁 5-10 秒乾淨授權女聲，避開男聲、配樂、唱歌、重疊說話。</li>
        <li>補逐字稿，reference audio 和 ref_text 要對得上；錯字會直接污染 clone。</li>
        <li>可做輕微 high-pass、RMS normalize，但不要把音色壓扁。</li>
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
        <li>先用 3-12 句 audition 篩 teacher，再決定是否擴成 500-5000 句 corpus。</li>
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
        <li>16-step 是品質檢查；低 step 只看速度時容易犧牲咬字與乾淨度。</li>
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
        <li>完整 16-step ZipVoice 是 teacher，few-step ZipVoice 是 student。</li>
        <li>數學上是在學同一條 flow trajectory 的短路徑：讓 3-4 個大步逼近 16 個小步的結果。</li>
        <li>這才是速度真正有機會變快的地方；單純改 inference steps 通常會壞聲音。</li>
      </ol>
      <pre class="mono">teacher: ZipVoice 16-step
student: ZipVoice 3/4-step
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

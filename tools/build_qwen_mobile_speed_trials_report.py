#!/usr/bin/env python3
"""Build a standalone phone-readable report for Qwen->ZipVoice mobile speed trials."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
BENCH = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1"
REPORT_DIR = BASE / "reports" / "qwen_zipvoice_mobile_speed_trials_v1"
REPORT = REPORT_DIR / "qwen_zipvoice_mobile_speed_trials_v1_standalone.html"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, unit: str = "s") -> str:
    return f"{value:.2f}{unit}"


def mb(value: float) -> str:
    return f"{value:.1f}MB"


def audio_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'


def row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def metric_card(title: str, value: str, detail: str) -> str:
    return f"""
    <section class="metric">
      <div class="metric-title">{html.escape(title)}</div>
      <div class="metric-value">{html.escape(value)}</div>
      <p>{html.escape(detail)}</p>
    </section>
    """


def find_row(summary: dict[str, Any], *, mode: str | None = None, step: int | None = None, sample_id: str | None = None) -> dict[str, Any]:
    for item in summary["rows"]:
        if mode is not None and item.get("mode") != mode:
            continue
        if step is not None and int(item.get("step", item.get("steps", -1))) != step:
            continue
        if sample_id is not None and item.get("sample_id") != sample_id:
            continue
        return item
    raise KeyError((mode, step, sample_id))


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    original = read_json(BENCH / "sherpa_runtime" / "benchmark.json")
    quant = read_json(BENCH / "vocoder_quant" / "sherpa_runtime" / "benchmark.json")
    speed = read_json(BENCH / "vocoder_quant_speed_1p12" / "sherpa_runtime" / "benchmark.json")
    chunked = read_json(BENCH / "chunked_playback_vocoder_quant_speed_1p12" / "benchmark.json")
    tiny = read_json(BENCH / "tiny_student_smoke_runtime" / "benchmark.json")
    threads1 = read_json(BENCH / "vocoder_quant_threads1" / "sherpa_runtime" / "benchmark.json")
    threads2 = read_json(BENCH / "vocoder_quant_threads2" / "sherpa_runtime" / "benchmark.json")

    original3 = original["aggregates"]["sherpa_step3"]
    quant3 = quant["aggregates"]["sherpa_step3"]
    quant4 = quant["aggregates"]["sherpa_step4"]
    speed3 = speed["aggregates"]["sherpa_step3"]
    tiny3 = tiny["aggregates"]["sherpa_step3"]

    long1_full = find_row(chunked, mode="full_sentence", step=3, sample_id="long_01")
    long1_chunk = find_row(chunked, mode="chunked_playback", step=3, sample_id="long_01")
    long2_full = find_row(chunked, mode="full_sentence", step=3, sample_id="long_02")
    long2_chunk = find_row(chunked, mode="chunked_playback", step=3, sample_id="long_02")

    audio_items = [
        (
            "原始 Vocos / 3-step",
            "你先不要急，我们慢慢来，把事情一件一件处理好。",
            BENCH / "sherpa_runtime" / "audio" / "sherpa_step3_s04.wav",
        ),
        (
            "量化 Vocos / 3-step",
            "同一句。比較音質是否有變薄、破音或尾音怪。",
            BENCH / "vocoder_quant" / "sherpa_runtime" / "audio" / "sherpa_step3_s04.wav",
        ),
        (
            "量化 Vocos / 3-step / speed 1.12",
            "同一句。這是體感更快的 demo 候選。",
            BENCH / "vocoder_quant_speed_1p12" / "sherpa_runtime" / "audio" / "sherpa_step3_speed1p12_s04.wav",
        ),
        (
            "長句完整生成",
            long2_full["text"],
            Path(long2_full["output"]),
        ),
        (
            "長句分段邊播",
            "同一段文字，app 可在第一段生成後先播放。",
            Path(long2_chunk["output"]),
        ),
        (
            "Tiny student smoke",
            "只訓練 2 iter，音質不可用；用來證明小架構可進 sherpa pipeline。",
            BENCH / "tiny_student_smoke_runtime" / "audio" / "sherpa_step3_s04.wav",
        ),
    ]

    audio_html = "\n".join(
        f"""
        <article class="audio-card">
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(text)}</p>
          {audio_tag(path)}
        </article>
        """
        for title, text, path in audio_items
    )

    comparison_rows = "\n".join(
        [
            row([
                "原始 Qwen→ZipVoice int8",
                "175.8MB",
                mb(original["final_peak_rss_mb"]),
                fmt(original3["avg_wall_s"]),
                f"RTF {original3['avg_rtf']:.3f}",
                "可用 baseline",
            ]),
            row([
                "量化 vocoder",
                mb(quant["model_size_mb"]["total_mb"]),
                mb(quant["final_peak_rss_mb"]),
                fmt(quant3["avg_wall_s"]),
                f"RTF {quant3['avg_rtf']:.3f}",
                "目前最穩",
            ]),
            row([
                "量化 vocoder + speed 1.12",
                mb(speed["model_size_mb"]["total_mb"]),
                mb(speed["final_peak_rss_mb"]),
                fmt(speed3["avg_wall_s"]),
                f"RTF {speed3['avg_rtf']:.3f}",
                "體感最快，但音長變短",
            ]),
            row([
                "量化 vocoder / 4-step",
                mb(quant["model_size_mb"]["total_mb"]),
                mb(quant["final_peak_rss_mb"]),
                fmt(quant4["avg_wall_s"]),
                f"RTF {quant4['avg_rtf']:.3f}",
                "音質保守候選",
            ]),
            row([
                "Tiny student smoke",
                mb(tiny["model_size_mb"]["total_mb"]),
                mb(tiny["final_peak_rss_mb"]),
                fmt(tiny3["avg_wall_s"]),
                "音訊長度失真",
                "需長訓，現在不可用",
            ]),
        ]
    )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen→ZipVoice 手機速度實驗</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #667085;
      --line: #e5e7eb;
      --soft: #f7f7f4;
      --accent: #b91c1c;
      --accent-soft: #fee2e2;
      --ok: #0f766e;
      --warn: #a16207;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #fafafa;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "PingFang TC", "Segoe UI", sans-serif;
      line-height: 1.6;
      font-size: 16px;
    }}
    main {{
      width: min(100%, 860px);
      margin: 0 auto;
      padding: 24px 16px 56px;
    }}
    header {{
      padding: 22px 0 14px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h1 {{
      margin: 8px 0 10px;
      font-size: clamp(28px, 8vw, 42px);
      line-height: 1.12;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 23px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 6px;
      font-size: 17px;
      line-height: 1.35;
      letter-spacing: 0;
    }}
    p {{ margin: 0 0 12px; }}
    .lead {{ color: var(--muted); font-size: 17px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin: 18px 0 8px;
    }}
    .metric, .panel, .audio-card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric-title {{ color: var(--muted); font-size: 13px; font-weight: 700; }}
    .metric-value {{ font-size: 27px; font-weight: 800; margin: 4px 0 4px; }}
    .metric p, .audio-card p {{ color: var(--muted); font-size: 14px; }}
    .tag {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      margin: 0 6px 6px 0;
      font-size: 12px;
      font-weight: 700;
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .tag.ok {{ background: #ccfbf1; color: var(--ok); }}
    .tag.warn {{ background: #fef3c7; color: var(--warn); }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ width: 100%; min-width: 720px; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ font-size: 13px; color: var(--muted); background: var(--soft); }}
    tr:last-child td {{ border-bottom: 0; }}
    .audio-grid {{ display: grid; gap: 12px; }}
    audio {{ width: 100%; margin-top: 8px; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
      background: #f2f4f7;
      padding: 1px 4px;
      border-radius: 5px;
    }}
    .flow {{
      white-space: pre-wrap;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.55;
    }}
    .note {{
      border-left: 4px solid var(--accent);
      background: #fff;
      padding: 12px 14px;
      border-radius: 6px;
      color: #3f3f46;
    }}
    @media (max-width: 520px) {{
      main {{ padding: 18px 14px 46px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      h2 {{ font-size: 21px; }}
      table {{ min-width: 680px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">2026-06-05・Qwen→ZipVoice mobile speed trials</div>
    <h1>真正有效的加速方向：已跑完第一輪</h1>
    <p class="lead">這份只回答一件事：要讓手機離線 TTS 更即時，哪些做法真的有效。結論先講：今天最可用是 <b>Qwen student int8 + 量化 Vocos + 3-step + 分段邊播</b>；真正更小更快要靠小架構學生長訓。</p>
  </header>

  <section class="metrics">
    {metric_card("目前可用最快", fmt(speed3["avg_wall_s"]), "3-step + 量化 vocoder + speed 1.12；屬於體感加速。")}
    {metric_card("模型包大小", mb(quant["model_size_mb"]["total_mb"]), "正式 qwen student + 量化 vocoder。")}
    {metric_card("長句先出聲", f"{long2_chunk['ttfa_s']:.2f}s", "分段邊播的第一段等待時間；完整長句總時間約 " + fmt(long2_chunk["total_wall_s"]) + "。")}
    {metric_card("Tiny 目標包", mb(tiny["model_size_mb"]["total_mb"]), "小架構 smoke 已可匯出 ONNX/sherpa，但音質尚不可用。")}
  </section>

  <section class="panel">
    <span class="tag ok">已有效</span>
    <span class="tag ok">可放進 app resources</span>
    <span class="tag warn">native runtime 尚未接</span>
    <p>今天不是只改報告。已完成：量化 vocoder、分段邊播 benchmark、iOS model pack 準備、Xcode simulator build、tiny student 訓練 smoke、tiny ONNX export。尚未完成：Swift 端直接呼叫 sherpa-onnx C/C++ runtime，因為專案裡目前沒有 sherpa-onnx iOS XCFramework / Swift wrapper。</p>
  </section>

  <h2>一眼比較</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>方案</th>
          <th>模型大小</th>
          <th>Python peak RSS</th>
          <th>一句話生成</th>
          <th>RTF / 備註</th>
          <th>判斷</th>
        </tr>
      </thead>
      <tbody>{comparison_rows}</tbody>
    </table>
  </div>

  <h2>音檔試聽</h2>
  <div class="audio-grid">{audio_html}</div>

  <h2>分段邊播結果</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>文字</th><th>完整生成</th><th>分段第一段出聲</th><th>分段總時間</th><th>判斷</th></tr></thead>
      <tbody>
        {row(["long_01", fmt(long1_full["total_wall_s"]), fmt(long1_chunk["ttfa_s"]), fmt(long1_chunk["total_wall_s"]), "第一段可提早約 " + fmt(long1_full["total_wall_s"] - long1_chunk["ttfa_s"])])}
        {row(["long_02", fmt(long2_full["total_wall_s"]), fmt(long2_chunk["ttfa_s"]), fmt(long2_chunk["total_wall_s"]), "第一段可提早約 " + fmt(long2_full["total_wall_s"] - long2_chunk["ttfa_s"])])}
      </tbody>
    </table>
  </div>

  <h2>Vocoder 最佳化</h2>
  <div class="panel">
    <p><b>做法：</b>把 <code>vocos_24khz.onnx</code> 做 ONNX Runtime dynamic int8 quantization。</p>
    <p><b>結果：</b>vocoder 從 51.6MB 降到 13.1MB；整包從 175.8MB 降到 137.3MB。3-step 平均一句從 {fmt(original3["avg_wall_s"])} 降到 {fmt(quant3["avg_wall_s"])}；peak RSS 從 {mb(original["final_peak_rss_mb"])} 降到 {mb(quant["final_peak_rss_mb"])}。</p>
    <p><b>判斷：</b>這是今天最乾淨有效的 wins。它不會讓速度暴衝，但同時降低 app 包大小、記憶體和一點生成時間。</p>
  </div>

  <h2>Thread 數測試</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>threads</th><th>一句話生成</th><th>peak RSS</th><th>判斷</th></tr></thead>
      <tbody>
        {row(["4", fmt(quant3["avg_wall_s"]), mb(quant["final_peak_rss_mb"]), "最佳平衡"])}
        {row(["2", fmt(threads2["aggregates"]["sherpa_step3"]["avg_wall_s"]), mb(threads2["final_peak_rss_mb"]), "慢很多，記憶體幾乎沒省"])}
        {row(["1", fmt(threads1["aggregates"]["sherpa_step3"]["avg_wall_s"]), mb(threads1["final_peak_rss_mb"]), "更慢，記憶體也沒省"])}
      </tbody>
    </table>
  </div>

  <h2>Native App 狀態</h2>
  <div class="panel">
    <p><b>已做：</b>新增 <code>RedBowVoice/Resources/Models/QwenZipVoiceDistillInt8</code>，裡面是這次真正測過的 Qwen→ZipVoice student int8 + 量化 vocoder。Xcode 已用 iPhone 17 simulator build 成功。</p>
    <p><b>尚未做完：</b>Swift 的 <code>VoiceEngine</code> 目前仍是打本機 HTTP server，不是 native sherpa。要真正手機離線，需要加入 sherpa-onnx iOS XCFramework，寫 Swift/C wrapper，然後把 output audio buffer 接到 AVAudioEngine 或 AVAudioPlayerNode。</p>
    <p><b>我對這條路的判斷：</b>native C++/sherpa app 會比 Python server 更接近真實手機數字，也有機會少掉 Python RSS 的額外 overhead。這是下一個最值得做的工程項。</p>
  </div>

  <h2>小架構 Student</h2>
  <div class="panel">
    <p><b>做法：</b>新增 <code>conf/zipvoice_mobile_student_qwen_smoke.json</code>，把 ZipVoice 從 122.7M 參數縮到 33.5M。用現有 Qwen teacher fbank 跑 2 iteration smoke，再匯出 ONNX/int8。</p>
    <p><b>結果：</b>tiny student 推論包約 {mb(tiny["model_size_mb"]["total_mb"])}，載入峰值約 {mb(tiny["load_peak_rss_mb"])}。它能進 sherpa 產出音訊，但因為只訓練 2 iter，duration 和音質都還不可信。</p>
    <p><b>判斷：</b>這是唯一真正可能把模型降到 50MB 級、同時改善速度的路。不是今天立刻可用，但值得接著跑長訓和真正 few-step distillation。</p>
  </div>

  <h2>流程圖</h2>
  <pre class="flow">Qwen 1.7B teacher voice
  ↓ 產生台灣低卷舌女聲 teacher corpus
ZipVoice base
  ↓ fine-tune / distill 學老師音色與說話方式
Qwen→ZipVoice student
  ↓ ONNX export + MatMul int8
ZipVoice int8 3/4-step
  ↓ 量化 vocoder
137MB mobile pack
  ↓ 分段邊生成邊播放
手機上先聽到第一段

下一條大路：
Qwen teacher corpus
  ↓ train 33.5M tiny ZipVoice student
  ↓ long training + few-step distillation
約 50MB 級真正小學生</pre>

  <h2>我會怎麼繼續</h2>
  <div class="note">
    <p><b>短期 demo：</b>用現在的 137MB 量化包，手機 app 走 native sherpa + 3-step + 分段播放。</p>
    <p><b>中期速度：</b>把 tiny student 長訓到先能正常念句子，再做 4/3-step distillation。</p>
    <p><b>記憶體：</b>Python peak RSS 不能直接等同 iPhone native；真正數字要在 iOS runtime 上用 Instruments 量。</p>
  </div>
</main>
</body>
</html>
"""
    REPORT.write_text(html_text, encoding="utf-8")
    print(REPORT)
    print(f"size_mb={REPORT.stat().st_size / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

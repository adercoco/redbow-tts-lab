#!/usr/bin/env python3
"""Build a standalone report for Qwen->ZipVoice speedup experiments."""

from __future__ import annotations

import base64
import html
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
BENCH = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1"
REPORT_DIR = BASE / "reports" / "qwen_zipvoice_speedup_v1"
OUT = REPORT_DIR / "qwen_zipvoice_speedup_v1_standalone.html"
TEACHER_DIR = BASE / "reports" / "zipvoice_three_way" / "teacher_qwen3_1p7b_compare" / "audio"

TESTS = [
    ("s01", "smoke_01", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("s02", "smoke_02", "我知道你现在有点紧张，可是先不要急着证明自己，慢慢说，我会听完。"),
    ("s03", "smoke_03", "这件事我们先放在同一个地方整理，等线索够清楚，再决定下一步要怎么做。"),
    ("s04", None, "你先不要急，我们慢慢来，把事情一件一件处理好。"),
    ("s05", None, "等一下我先看一下讯息，晚一点再跟你说。"),
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def data_uri(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def pct_faster(old: float, new: float) -> str:
    return f"{(1 - new / old) * 100:.0f}%"


def collect() -> dict[str, Any]:
    zip4 = load_json(BENCH / "zipvoice_cached_onnx" / "benchmark_threads4.json")
    zip_threads = {
        str(th): load_json(BENCH / "zipvoice_cached_onnx" / f"benchmark_threads{th}.json")
        for th in [1, 2, 4, 8]
    }
    sherpa_threads = {
        str(th): load_json(BENCH / "sherpa_runtime" / f"benchmark_threads{th}.json")
        for th in [1, 2, 4, 8]
    }
    return {
        "zip4": zip4,
        "zip_threads": zip_threads,
        "sherpa_threads": sherpa_threads,
    }


def model_total_mb(data: dict[str, Any]) -> float:
    size = data["model_size_mb"]
    if "total_mb" in size:
        return float(size["total_mb"])
    return float(size["text_encoder_int8_mb"]) + float(size["fm_decoder_int8_mb"]) + 52.0


def metric_rows(data: dict[str, Any]) -> str:
    rows = []
    for runtime_label, runs in [
        ("ZipVoice repo cached", data["zip_threads"]),
        ("Sherpa-ONNX runtime", data["sherpa_threads"]),
    ]:
        for th in ["1", "2", "4", "8"]:
            bench = runs[th]
            for step in [3, 4]:
                key = (
                    f"zipvoice_repo_cached_step{step}"
                    if runtime_label.startswith("ZipVoice")
                    else f"sherpa_step{step}"
                )
                agg = bench["aggregates"][key]
                rows.append(
                    f"""
                    <tr>
                      <td>{esc(runtime_label)}</td>
                      <td>{th}</td>
                      <td>{step}</td>
                      <td>{fmt(agg['avg_wall_s'], 2)}s</td>
                      <td>{fmt(agg['avg_rtf'], 3)}</td>
                      <td>{fmt(bench.get('final_peak_rss_mb', bench.get('load_peak_rss_mb')), 0)}MB</td>
                    </tr>
                    """
                )
    return "\n".join(rows)


def audio_card(title: str, path: Path | None, meta: str = "") -> str:
    if path is None or not path.exists():
        return f"""
        <section class="audio-card missing">
          <b>{esc(title)}</b>
          <p>{esc(meta or '沒有這句老師原音')}</p>
        </section>
        """
    return f"""
    <section class="audio-card">
      <b>{esc(title)}</b>
      <audio controls preload="metadata" src="{data_uri(path)}"></audio>
      {f'<p>{esc(meta)}</p>' if meta else ''}
    </section>
    """


def audio_rows(data: dict[str, Any]) -> str:
    sherpa = load_json(BENCH / "sherpa_runtime" / "benchmark_threads4.json")
    sherpa_by_step_id = {
        (str(row["step"]), row["sample_id"]): row
        for row in sherpa["rows"]
    }
    rows: list[str] = []
    for sample_id, teacher_id, text in TESTS:
        teacher = TEACHER_DIR / f"{teacher_id}.wav" if teacher_id else None
        step3 = BENCH / "sherpa_runtime" / "audio" / f"sherpa_step3_{sample_id}.wav"
        step4 = BENCH / "sherpa_runtime" / "audio" / f"sherpa_step4_{sample_id}.wav"
        meta3 = sherpa_by_step_id.get(("3", sample_id), {})
        meta4 = sherpa_by_step_id.get(("4", sample_id), {})
        rows.append(
            f"""
            <article class="listen-row">
              <p class="line-text">{esc(text)}</p>
              <div class="audio-grid">
                {audio_card("Qwen 1.7B teacher", teacher)}
                {audio_card("Sherpa 3-step", step3, f"{fmt(meta3.get('wall_s'), 2)}s / RTF {fmt(meta3.get('rtf'), 3)}")}
                {audio_card("Sherpa 4-step", step4, f"{fmt(meta4.get('wall_s'), 2)}s / RTF {fmt(meta4.get('rtf'), 3)}")}
              </div>
            </article>
            """
        )
    return "\n".join(rows)


def copy_audio_artifacts() -> None:
    audio_dir = REPORT_DIR / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)
    for source in [
        *(BENCH / "sherpa_runtime" / "audio").glob("sherpa_step*_s*.wav"),
        *(BENCH / "zipvoice_cached_onnx" / "audio").glob("cached_step*_s*.wav"),
    ]:
        shutil.copy2(source, audio_dir / source.name)


def build_html() -> str:
    data = collect()
    copy_audio_artifacts()
    zip4 = data["zip_threads"]["4"]
    sherpa4 = data["sherpa_threads"]["4"]
    best = sherpa4["aggregates"]["sherpa_step3"]
    step4 = sherpa4["aggregates"]["sherpa_step4"]
    zip_cached3 = zip4["aggregates"]["zipvoice_repo_cached_step3"]
    zip_cached4 = zip4["aggregates"]["zipvoice_repo_cached_step4"]
    zip_uncached = load_json(
        BENCH / "zipvoice_cached_onnx" / "benchmark_uncached_threads4.json"
    )
    uncached3 = zip_uncached["aggregates"].get("zipvoice_repo_uncached_step3", {})
    uncached4 = zip_uncached["aggregates"].get("zipvoice_repo_uncached_step4", {})
    cache_delta3 = float(uncached3.get("avg_wall_s", zip_cached3["avg_wall_s"])) - float(
        zip_cached3["avg_wall_s"]
    )
    cache_delta4 = float(uncached4.get("avg_wall_s", zip_cached4["avg_wall_s"])) - float(
        zip_cached4["avg_wall_s"]
    )
    total_mb = model_total_mb(sherpa4)
    decoder_mb = sherpa4["model_size_mb"]["decoder_int8_mb"]
    encoder_mb = sherpa4["model_size_mb"]["encoder_int8_mb"]
    vocoder_mb = sherpa4["model_size_mb"]["vocoder_mb"]

    summary = {
        "best_candidate": {
            "runtime": "Sherpa-ONNX Python binding, model already loaded",
            "steps": 3,
            "threads": 4,
            "avg_wall_seconds": best["avg_wall_s"],
            "avg_rtf": best["avg_rtf"],
            "peak_rss_mb": sherpa4["final_peak_rss_mb"],
            "model_size_mb": total_mb,
        },
        "step4_candidate": {
            "avg_wall_seconds": step4["avg_wall_s"],
            "avg_rtf": step4["avg_rtf"],
        },
        "cache_delta_seconds": {"step3": cache_delta3, "step4": cache_delta4},
        "report": str(OUT),
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen→ZipVoice 加速實驗 v1</title>
  <style>
    :root {{
      --bg: #f5f2ec;
      --paper: #fffefa;
      --ink: #24211d;
      --muted: #6b6258;
      --line: #d9d0c4;
      --red: #b23b38;
      --blue: #255f7f;
      --green: #3f7659;
      --amber: #8b6427;
      --soft: #eee5da;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      font-size: 16px;
      line-height: 1.62;
    }}
    main {{
      width: min(1080px, 100%);
      margin: 0 auto;
      padding: 20px 14px 56px;
    }}
    .eyebrow {{ color: var(--red); font-size: 13px; font-weight: 800; }}
    h1 {{ margin: 6px 0 10px; font-size: clamp(30px, 8vw, 48px); line-height: 1.08; letter-spacing: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 23px; line-height: 1.25; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ margin: 7px 0; }}
    .lead {{ color: var(--muted); max-width: 780px; }}
    section, .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .metric {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric b {{ display: block; color: var(--blue); font-size: 28px; line-height: 1.1; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .verdict {{
      border-left: 5px solid var(--green);
      background: #fbfff9;
    }}
    .warning {{
      border-left: 5px solid var(--amber);
      background: #fffaf1;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ color: var(--muted); font-size: 12px; }}
    .table-wrap {{ overflow-x: auto; }}
    .listen-row {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }}
    .line-text {{ font-weight: 800; }}
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .audio-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fffaf2;
    }}
    .audio-card b {{ display: block; margin-bottom: 6px; }}
    .audio-card p {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    .missing {{ opacity: .68; }}
    audio {{ width: 100%; }}
    ul {{ padding-left: 20px; margin: 8px 0; }}
    li {{ margin: 6px 0; }}
    code {{
      background: #eee4d8;
      border-radius: 5px;
      padding: 2px 5px;
      word-break: break-word;
    }}
    .path {{ color: var(--muted); word-break: break-word; font-size: 12px; }}
    @media (max-width: 780px) {{
      main {{ padding-inline: 10px; }}
      .hero-metrics, .audio-grid {{ grid-template-columns: 1fr; }}
      table {{ min-width: 720px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Qwen→ZipVoice speedup experiment v1</div>
      <h1>3-step 是目前最快可用候選</h1>
      <p class="lead">這次不是只改報告文字，有重新跑實測：ZipVoice repo ONNX cached / uncached、Sherpa-ONNX runtime、1/2/4/8 threads。結論是：真正瓶頸在 decoder step loop，不在 reference cache。</p>
      <div class="hero-metrics">
        <div class="metric"><b>{fmt(best['avg_wall_s'], 2)}s</b><span>最佳：Sherpa 3-step / 4 threads / 平均每句</span></div>
        <div class="metric"><b>{fmt(step4['avg_wall_s'], 2)}s</b><span>Sherpa 4-step / 4 threads / 平均每句</span></div>
        <div class="metric"><b>{fmt(sherpa4['final_peak_rss_mb'], 0)}MB</b><span>Sherpa Python binding peak RSS</span></div>
        <div class="metric"><b>{fmt(total_mb, 0)}MB</b><span>ONNX int8 + vocoder 模型檔</span></div>
      </div>
    </header>

    <section class="verdict">
      <h2>短結論</h2>
      <p><strong>有加快，而且方向明確：</strong>舊 8-step 約 2.7s/句、6-step 約 2.1s/句；這次最佳 Sherpa 3-step 約 {fmt(best['avg_wall_s'], 2)}s/句，比 8-step 快約 {pct_faster(2.7, float(best['avg_wall_s']))}，比 6-step 快約 {pct_faster(2.1, float(best['avg_wall_s']))}。</p>
      <p><strong>cache 結果：</strong>speaker/reference cache 沒有明顯加速。3-step cached vs uncached 差約 {fmt(cache_delta3, 3)}s，4-step 差約 {fmt(cache_delta4, 3)}s。原因是 prompt 前處理太小，主要時間都花在 decoder loop。</p>
      <p><strong>目前建議：</strong>demo app 預設用 Sherpa-ONNX 3-step / 4 threads；提供 4-step 當品質模式。2-step 之前聽感不穩，暫時不採用。</p>
    </section>

    <section>
      <h2>模型與記憶體</h2>
      <ul>
        <li>權重：Qwen teacher distill stage2 few-step10 ONNX int8。</li>
        <li>模型大小：encoder {fmt(encoder_mb, 1)}MB + decoder {fmt(decoder_mb, 1)}MB + vocoder {fmt(vocoder_mb, 1)}MB = 約 {fmt(total_mb, 1)}MB。</li>
        <li>Sherpa runtime peak RSS：4 threads 約 {fmt(sherpa4['final_peak_rss_mb'], 0)}MB；8 threads 約 {fmt(data['sherpa_threads']['8']['final_peak_rss_mb'], 0)}MB。</li>
        <li>ZipVoice repo Python peak RSS：約 {fmt(zip4['final_peak_rss_mb'], 0)}MB；比 Sherpa binding 高。</li>
      </ul>
    </section>

    <section>
      <h2>完整速度表</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Runtime</th><th>Threads</th><th>Steps</th><th>一句平均</th><th>RTF</th><th>Peak RSS</th></tr>
          </thead>
          <tbody>{metric_rows(data)}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>試聽比較</h2>
      <p>前三句有 Qwen 1.7B teacher 原音可比；後兩句只放 3-step / 4-step 加速候選。請優先聽有沒有漏字、字音跑掉、尾音變硬。</p>
      {audio_rows(data)}
    </section>

    <section class="warning">
      <h2>這次學到什麼</h2>
      <ul>
        <li><strong>真正 few-step 有效：</strong>速度主要靠 3/4-step，而不是再縮模型。</li>
        <li><strong>cache 不神：</strong>speaker reference 可以載入一次，但它不是主要瓶頸；不要期待 cache 帶來 2 倍加速。</li>
        <li><strong>4 threads 最好：</strong>1/2 threads 慢，8 threads 沒更快，反而有 overhead。手機上要按機型再測，但預設 4 是合理起點。</li>
        <li><strong>C++/mobile runtime 仍必要：</strong>Python binding 已經呼叫 C++/ONNX 核心，所以桌機上不會突然再快 2 倍；但 app 端必須用 sherpa-onnx native，才能避免 Python、不必要 I/O、HTTP server 和記憶體膨脹。</li>
      </ul>
    </section>

    <section>
      <h2>下一刀</h2>
      <p>若要從約 1.1s 再往「接近即時」推，最值得做的是：短句 streaming / 邊合成邊播放、vocoder 最佳化、3-step 品質微調，以及手機實機 benchmark。再訓練更多 epoch 可能讓聲音更穩，但不會本質上讓 3-step 更快。</p>
      <p class="path">Benchmark dir: {esc(str(BENCH))}</p>
      <p class="path">Report: {esc(str(OUT))}</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print(OUT)
    print(REPORT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

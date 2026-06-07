#!/usr/bin/env python3
"""Build the v4 original-clone listening report with ASR and speaker scoring."""

from __future__ import annotations

import base64
import collections
import difflib
import html
import json
import mimetypes
import os
import re
import statistics
from pathlib import Path

import librosa
import onnxruntime
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "clone_audition_v4_original_models"
RESULTS = OUT / "all_results_asr_scored.json"
REPORT = OUT / "index.html"
STANDALONE = OUT / "clone_audition_v4_original_models_standalone.html"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
CAMPLUS = ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B" / "campplus.onnx"

RESULT_FILES = [
    "cosy_results.json",
    "f5_results.json",
    "gpt_results.json",
    "index_results.json",
    "qwen_results.json",
    "sherpa_zipvoice_results.json",
    "pocket_results.json",
]

MODEL_ORDER = [
    "CosyVoice2-0.5B original",
    "F5-TTS v1 Base original",
    "GPT-SoVITS v2 original single-ref",
    "IndexTTS2 original",
    "Qwen3 1.7B VoiceDesign ref attempt",
    "Sherpa ZipVoice int8 zh-min original 16-step",
    "Sherpa PocketTTS int8 original",
]

TEXT_LABELS = {
    "line_01": "先別急",
    "line_02": "慢慢說",
    "line_03": "慢慢決定",
    "line_04": "一起確認",
    "line_05": "下班吃東西",
    "line_06": "換個做法",
}

MODEL_META = {
    "CosyVoice2-0.5B original": {
        "size_paths": [ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"],
        "role": "原始 zero-shot clone teacher 候選；固定 pack_best2_7s reference。",
        "runtime": "server/teacher",
    },
    "F5-TTS v1 Base original": {
        "size_paths": [Path.home() / ".cache" / "huggingface" / "hub" / "models--SWivid--F5-TTS"],
        "role": "原始 F5 base zero-shot clone；CLI 每句冷啟動，時間偏保守。",
        "runtime": "server/teacher",
    },
    "GPT-SoVITS v2 original single-ref": {
        "size_paths": [ROOT / "external" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"],
        "role": "原始 GPT-SoVITS v2 zero-shot；這版用單段 3 秒 reference，避免 aux reference leakage。",
        "runtime": "server/teacher",
    },
    "IndexTTS2 original": {
        "size_paths": [ROOT / "external" / "index-tts" / "checkpoints"],
        "role": "原始 IndexTTS2 clone；高品質候選但體積和授權都不適合直接手機商用。",
        "runtime": "server/teacher",
    },
    "Qwen3 1.7B VoiceDesign ref attempt": {
        "size_paths": [Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"],
        "role": "Qwen VoiceDesign 原始模型 + ref attempt；不是保證純 clone，但可聽模型本身聲音方向。",
        "runtime": "teacher/reference",
    },
    "Sherpa ZipVoice int8 zh-min original 16-step": {
        "size_paths": [
            ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            ROOT / "models" / "sherpa" / "vocos_24khz.onnx",
        ],
        "role": "ZipVoice 原始品質聽感測試；這版沒有放 4-step/qint8 降階。",
        "runtime": "mobile candidate",
    },
    "Sherpa PocketTTS int8 original": {
        "size_paths": [ROOT / "models" / "sherpa" / "sherpa-onnx-pocket-tts-int8-2026-01-26"],
        "role": "小模型 clone baseline；本機 README 標 non-commercial，且內容穩定性需 ASR gate。",
        "runtime": "mobile experiment",
    },
}

PENDING = [
    {
        "name": "ChatterboxTTS original",
        "status": "installed, load timed out",
        "note": "已建 .venv-chatterbox311 並安裝 chatterbox-tts 0.1.7，但首次 from_pretrained 載入超過 3 分鐘沒有進入生成；先不阻塞 v4 報告。",
    },
    {
        "name": "Fish Speech / Fish Audio",
        "status": "not installed",
        "note": "本機沒有 repo/weights；偏 teacher/server 級，待單獨裝權重再測。",
    },
    {
        "name": "OpenVoice V2",
        "status": "not installed",
        "note": "本機沒有 repo/weights；偏 tone-color conversion pipeline，待單獨補測。",
    },
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_pack() -> dict:
    packs = json.loads(PACKS.read_text(encoding="utf-8"))
    return next(pack for pack in packs if pack["pack_id"] == "pack_best2_7s")


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for name in RESULT_FILES:
        path = OUT / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("status") == "ok" and row.get("output"):
                rows.append(row)
    return rows


def normalize_zh(text: str) -> str:
    replacements = {
        "別": "别",
        "欸": "诶",
        "緊": "紧",
        "願": "愿",
        "話": "话",
        "線": "线",
        "覺": "觉",
        "這": "这",
        "們": "们",
        "後": "后",
        "東": "东",
        "點": "点",
        "簡": "简",
        "單": "单",
        "換": "换",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text.lower()))


def asr_similarity(expected: str, actual: str) -> float:
    a = normalize_zh(expected)
    b = normalize_zh(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def repetition_junk(text: str) -> bool:
    compact = normalize_zh(text)
    if len(compact) < 4:
        return True
    return len(compact) >= 12 and len(set(compact)) / len(compact) < 0.22


def pass_asr(expected: str, actual: str) -> bool:
    compact = normalize_zh(actual)
    if len(compact) < 4 or repetition_junk(actual):
        return False
    if len(compact) > max(1, len(normalize_zh(expected))) * 3:
        return False
    return asr_similarity(expected, actual) >= 0.45


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    return audio


def embed(session: onnxruntime.InferenceSession, path: Path) -> torch.Tensor:
    audio = load_audio(path)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
    vector = torch.tensor(embedding, dtype=torch.float32)
    return torch.nn.functional.normalize(vector, dim=0)


def score_speaker(rows: list[dict]) -> None:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(CAMPLUS), sess_options=option, providers=["CPUExecutionProvider"])
    cache: dict[str, torch.Tensor] = {}

    def get(path_value: str) -> torch.Tensor | None:
        path = Path(path_value)
        if not path.exists():
            return None
        key = str(path)
        if key not in cache:
            cache[key] = embed(session, path)
        return cache[key]

    for row in rows:
        out = get(row["output"])
        ref = get(row["ref_audio"])
        if out is None or ref is None:
            row["speaker_cosine"] = None
            continue
        row["speaker_cosine"] = round(torch.dot(ref, out).item(), 4)
        row["speaker_score_note"] = "campplus_cosine_reference_to_output"


def transcribe_rows(rows: list[dict]) -> None:
    model = whisper.load_model("tiny")
    for idx, row in enumerate(rows, 1):
        path = Path(row["output"])
        info = sf.info(str(path))
        row["audio_duration_sec"] = round(float(info.duration), 3)
        row["audio_file_size"] = path.stat().st_size
        y, sr = sf.read(str(path), always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000)
        result = model.transcribe(
            y.astype("float32"),
            language="zh",
            fp16=False,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = result.get("text", "").strip()
        row["asr_text"] = text
        row["asr_similarity"] = round(asr_similarity(row.get("text", ""), text), 4)
        row["asr_pass"] = pass_asr(row.get("text", ""), text)
        print(
            f"[{idx:02d}/{len(rows)}] {row['family']} {row['text_id']} "
            f"asr={row['asr_pass']} sim={row['asr_similarity']:.3f} {text}"
        )


def bytes_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def fmt_bytes(value: int | None) -> str:
    if not value:
        return "-"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return "-"


def model_size(family: str) -> str:
    paths = MODEL_META[family].get("size_paths", [])
    return fmt_bytes(sum(bytes_of(Path(path)) for path in paths))


def fmt_sec(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "-"


def fmt_score(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def audio_src(path_value: str, *, standalone: bool) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    if standalone:
        return data_uri(path)
    return esc(os.path.relpath(path, OUT))


def audio_tag(path_value: str, *, standalone: bool) -> str:
    src = audio_src(path_value, standalone=standalone)
    if not src:
        return '<div class="missing">沒有音檔</div>'
    return f'<audio controls preload="none" src="{src}"></audio>'


def stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for family in MODEL_ORDER:
        items = [row for row in rows if row.get("family") == family]
        if not items:
            continue
        secs = [float(row.get("seconds") or 0) for row in items]
        scores = [float(row["speaker_cosine"]) for row in items if row.get("speaker_cosine") is not None]
        sims = [float(row.get("asr_similarity") or 0) for row in items]
        out[family] = {
            "n": len(items),
            "avg_sec": statistics.mean(secs) if secs else None,
            "asr_pass": sum(1 for row in items if row.get("asr_pass")),
            "avg_asr": statistics.mean(sims) if sims else None,
            "avg_speaker": statistics.mean(scores) if scores else None,
            "size": model_size(family),
        }
    return out


def badge(label: str, kind: str = "") -> str:
    return f'<span class="badge {esc(kind)}">{esc(label)}</span>'


def render_cards(rows: list[dict]) -> str:
    s = stats(rows)
    cards = []
    for family in MODEL_ORDER:
        if family not in s:
            continue
        item = s[family]
        meta = MODEL_META[family]
        status = "ASR 全過" if item["asr_pass"] == item["n"] else f"ASR {item['asr_pass']}/{item['n']}"
        cards.append(
            f"""
            <article class="model-card">
              <div class="model-top"><h3>{esc(family)}</h3>{badge(status, "ok" if item["asr_pass"] == item["n"] else "warn")}</div>
              <div class="grid">
                <div><span>模型大小</span><b>{esc(item['size'])}</b></div>
                <div><span>平均生成</span><b>{fmt_sec(item['avg_sec'])}</b></div>
                <div><span>聲紋分數</span><b>{fmt_score(item['avg_speaker'])}</b></div>
                <div><span>ASR 相似</span><b>{fmt_score(item['avg_asr'])}</b></div>
              </div>
              <p>{esc(meta['role'])}</p>
              <p class="small">{esc(meta['runtime'])}</p>
            </article>
            """
        )
    return "".join(cards)


def render_model_sections(rows: list[dict], *, standalone: bool) -> str:
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    sections = []
    for family in MODEL_ORDER:
        items = sorted(by_family.get(family, []), key=lambda row: row.get("text_id", ""))
        if not items:
            continue
        audio_items = []
        for row in items:
            pass_label = "ASR ok" if row.get("asr_pass") else "ASR fail"
            audio_items.append(
                f"""
                <article class="sample">
                  <div class="sample-head">
                    <b>{esc(TEXT_LABELS.get(row.get('text_id'), row.get('text_id')))}</b>
                    <span>{fmt_sec(row.get('seconds'))} · sim {fmt_score(row.get('speaker_cosine'))}</span>
                  </div>
                  <p class="expected">{esc(row.get('text'))}</p>
                  <div class="badges">{badge(pass_label, "ok" if row.get("asr_pass") else "warn")}{badge("ASR " + fmt_score(row.get("asr_similarity")), "")}</div>
                  {audio_tag(row.get('output', ''), standalone=standalone)}
                  <p class="asr"><b>ASR</b> {esc(row.get('asr_text', ''))}</p>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="section">
              <h2>{esc(family)}</h2>
              <p class="small">{esc(MODEL_META[family]['role'])}</p>
              <div class="sample-list">{''.join(audio_items)}</div>
            </section>
            """
        )
    return "".join(sections)


def render_reference(*, standalone: bool) -> str:
    pack = load_pack()
    return f"""
      <section class="section">
        <h2>固定 Reference</h2>
        <p class="small">這版所有 pack-capable 模型都用同一個 pack_best2_7s；GPT/Qwen 用其中單段 3 秒 reference。</p>
        <article class="sample">
          <div class="sample-head"><b>pack_best2_7s</b><span>{esc(', '.join(pack.get('ref_files', [])))}</span></div>
          <p>{esc(pack['ref_text'])}</p>
          {audio_tag(pack['pack_audio'], standalone=standalone)}
        </article>
      </section>
    """


def render_pending() -> str:
    return "".join(
        f"""
        <article class="pending">
          <div class="sample-head"><b>{esc(item['name'])}</b><span>{esc(item['status'])}</span></div>
          <p>{esc(item['note'])}</p>
        </article>
        """
        for item in PENDING
    )


def render_html(rows: list[dict], *, standalone: bool) -> str:
    mode = "standalone：音檔已嵌入，可上傳雲端或手機開" if standalone else "local：音檔使用相對路徑"
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>原始 Clone 模型試聽 v4</title>
  <style>
    :root {{
      --bg:#f6f4ef; --paper:#fffdf8; --ink:#22201c; --muted:#776f65;
      --line:#ddd4c6; --accent:#b84040; --ok:#226950; --warn:#a65f00; --soft:#eee7dc;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--ink);
      font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
      line-height:1.58;
    }}
    main {{ width:min(980px,100%); margin:0 auto; padding:22px 14px 46px; }}
    header {{ padding:10px 2px 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }}
    .eyebrow {{ color:var(--accent); font-size:13px; font-weight:700; }}
    h1 {{ font-size:clamp(28px,7vw,46px); line-height:1.08; margin:7px 0 10px; letter-spacing:0; }}
    h2 {{ font-size:22px; line-height:1.25; margin:0 0 10px; }}
    h3 {{ font-size:17px; line-height:1.3; margin:0; }}
    p {{ margin:8px 0; }}
    .small,.asr {{ color:var(--muted); font-size:14px; }}
    .callout {{ background:#fff3e0; border:1px solid #e5cfa8; border-radius:8px; padding:12px; margin:14px 0; }}
    .models,.sample-list {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; }}
    .model-card,.section,.sample,.pending {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:13px; }}
    .section {{ margin:15px 0; }}
    .model-top,.sample-head {{ display:flex; gap:8px; justify-content:space-between; align-items:flex-start; }}
    .sample-head span {{ color:var(--muted); font-size:12px; text-align:right; min-width:96px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:11px 0; }}
    .grid div,.expected {{ background:#f7f1e8; border-radius:8px; padding:8px; }}
    .grid span {{ display:block; color:var(--muted); font-size:12px; }}
    .grid b {{ font-size:15px; overflow-wrap:anywhere; }}
    .expected {{ font-size:14px; color:var(--ink); border:1px solid var(--line); }}
    audio {{ width:100%; margin-top:8px; }}
    .badge {{ display:inline-flex; border-radius:999px; border:1px solid var(--line); padding:3px 8px; font-size:12px; color:var(--muted); background:#fff; white-space:nowrap; }}
    .badge.ok {{ color:var(--ok); border-color:#9ccdbd; background:#eef9f5; }}
    .badge.warn {{ color:var(--warn); border-color:#e2bd75; background:#fff8e8; }}
    .badges {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }}
    @media (max-width:620px) {{
      main {{ padding-inline:12px; }}
      .models,.sample-list {{ grid-template-columns:1fr; }}
      .model-top,.sample-head {{ display:block; }}
      .sample-head span {{ display:block; text-align:left; margin-top:4px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Original clone audition v4 · {esc(mode)}</div>
    <h1>原始 Clone 模型試聽</h1>
    <p>這版先不放 ZipVoice 4-step/qint8 降階，只聽各模型原始 clone 聲音。每個模型 6 句，全部跑 ASR gate 和 speaker cosine。</p>
  </header>
  <section class="callout">
    <b>已生成：</b> CosyVoice2、F5-TTS、GPT-SoVITS 單 ref、IndexTTS2、Qwen3 ref attempt、ZipVoice 16-step、PocketTTS。
  </section>
  <section>
    <h2>模型總表</h2>
    <div class="models">{render_cards(rows)}</div>
  </section>
  {render_reference(standalone=standalone)}
  {render_model_sections(rows, standalone=standalone)}
  <section class="section">
    <h2>待補測</h2>
    <div>{render_pending()}</div>
  </section>
  <section class="section">
    <h2>檢查規則</h2>
    <p><b>ASR gate</b>：Whisper tiny 轉文字後，必須接近原句；只會說中文但內容跑掉也算 fail。</p>
    <p><b>聲紋分數</b>：CAMPPlus cosine，比較 reference 與輸出音色接近度；它不等於主觀好聽。</p>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(f"rows={len(rows)}")
    score_speaker(rows)
    transcribe_rows(rows)
    RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(render_html(rows, standalone=False), encoding="utf-8")
    STANDALONE.write_text(render_html(rows, standalone=True), encoding="utf-8")
    print(RESULTS)
    print(REPORT)
    print(STANDALONE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

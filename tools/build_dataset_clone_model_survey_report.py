#!/usr/bin/env python3
"""Build a phone-readable dataset voice-clone survey with ASR sanity checks."""

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
V2 = BASE / "clone_audition_v2_large_models"
OUT = BASE / "clone_audition_v3_dataset_models"
RESULTS = OUT / "all_results_asr_scored.json"
REPORT = OUT / "index.html"
STANDALONE = OUT / "clone_audition_v3_dataset_models_standalone.html"
CAMPLUS = ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B" / "campplus.onnx"

MODEL_ORDER = [
    "CosyVoice2-0.5B pack",
    "F5-TTS v1 Base pack",
    "GPT-SoVITS v2 aux refs",
    "IndexTTS2 pack",
    "F5-TTS fine-tuned 40 updates",
    "Sherpa ZipVoice int8 zh-min 8-step",
    "Sherpa ZipVoice int8 zh-min 4-step",
    "Sherpa ZipVoice int8 zh-min 4-step qint8 vocoder",
    "Sherpa PocketTTS int8 5-step",
]

TEXT_LABELS = {
    "tw_soft": "先別急",
    "tw_calm": "慢慢說",
    "tw_decide": "慢慢決定",
    "tw_confirm": "一起確認",
}

MODEL_META = {
    "CosyVoice2-0.5B pack": {
        "size_paths": [ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"],
        "runtime": "Teacher / server",
        "license": "Apache-2.0 route",
        "note": "Zero-shot clone；目前本機聽感最值得當老師候選之一，不適合直接塞手機。",
    },
    "F5-TTS v1 Base pack": {
        "size_label": "約 1.3GB HF/model cache",
        "runtime": "Teacher / server",
        "license": "MIT",
        "note": "Zero-shot clone；品質可比，但推論慢，不是手機 runtime。",
    },
    "GPT-SoVITS v2 aux refs": {
        "size_paths": [ROOT / "external" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"],
        "runtime": "Teacher / server",
        "license": "MIT code/model route in local setup",
        "note": "使用 1 主 ref + 多段 aux refs；適合驗證多 reference 對聲音穩定性的幫助。",
    },
    "IndexTTS2 pack": {
        "size_paths": [ROOT / "external" / "index-tts" / "checkpoints"],
        "runtime": "Teacher / server",
        "license": "IndexTTS license, commercial use needs separate review",
        "note": "大模型 clone 候選；技術可聽，但授權和體積不適合作為商用手機 runtime。",
    },
    "F5-TTS fine-tuned 40 updates": {
        "size_paths": [ROOT / ".venv-f5" / "lib" / "python3.12" / "ckpts" / "downloads_female_voice_finetune_v1"],
        "runtime": "Fine-tune proof",
        "license": "MIT base + your licensed dataset",
        "note": "用你授權資料做過短訓練，只是 proof-of-work；checkpoint 太大，不能當手機方案。",
    },
    "Sherpa ZipVoice int8 zh-min 8-step": {
        "size_paths": [
            ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            ROOT / "models" / "sherpa" / "vocos_24khz.onnx",
        ],
        "runtime": "Mobile candidate",
        "license": "Sherpa/ZipVoice model pack route; verify final app license before release",
        "note": "手機路線優先候選。8-step 是品質基準，速度比 4-step 慢。",
    },
    "Sherpa ZipVoice int8 zh-min 4-step": {
        "size_paths": [
            ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            ROOT / "models" / "sherpa" / "vocos_24khz.onnx",
        ],
        "runtime": "Mobile candidate",
        "license": "Sherpa/ZipVoice model pack route; verify final app license before release",
        "note": "同模型降推論 steps，主要追速度；必須靠 ASR/聽感確認是否崩。",
    },
    "Sherpa ZipVoice int8 zh-min 4-step qint8 vocoder": {
        "size_paths": [
            ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min",
            ROOT / "models" / "sherpa" / "vocos_24khz_dynamic_qint8.onnx",
        ],
        "runtime": "Mobile candidate",
        "license": "Sherpa/ZipVoice model pack route; verify final app license before release",
        "note": "用更小 vocoder 測速度/大小；聲音若變粗或雜訊增加就淘汰。",
    },
    "Sherpa PocketTTS int8 5-step": {
        "size_paths": [ROOT / "models" / "sherpa" / "sherpa-onnx-pocket-tts-int8-2026-01-26"],
        "runtime": "Mobile experiment",
        "license": "Non-commercial local pack",
        "note": "小型 clone 候選，但本地 README 標示 non-commercial，只能技術試聽。",
    },
}

NOT_RUN = [
    {
        "name": "Fish Audio S2 / Fish Speech",
        "local": "not installed",
        "reason": "官方 S2-Pro 是 4B teacher 級，適合 server/老師，不是手機 runtime；本機目前沒有 repo/weights。",
        "source": "https://github.com/fishaudio/fish-speech",
    },
    {
        "name": "Chatterbox / Chatterbox Turbo",
        "local": "not installed",
        "reason": "MIT、5 秒 reference zero-shot，值得下一輪補測；本機目前沒有 package/weights。",
        "source": "https://www.resemble.ai/learn/models/chatterbox",
    },
    {
        "name": "OpenVoice V2",
        "local": "not installed",
        "reason": "MIT tone-color cloning，比較像 voice conversion pipeline；可補測，但不是目前已配置 runtime。",
        "source": "https://github.com/myshell-ai/OpenVoice",
    },
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def normalize_zh(text: str) -> str:
    text = text.lower()
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
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def repetition_junk(text: str) -> bool:
    compact = normalize_zh(text)
    if len(compact) < 4:
        return True
    unique_ratio = len(set(compact)) / max(1, len(compact))
    return len(compact) >= 12 and unique_ratio < 0.22


def asr_similarity(expected: str, actual: str) -> float:
    a = normalize_zh(expected)
    b = normalize_zh(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def pass_asr(expected: str, actual: str) -> bool:
    compact = normalize_zh(actual)
    if len(compact) < 4 or repetition_junk(actual):
        return False
    expected_len = max(1, len(normalize_zh(expected)))
    if len(compact) > expected_len * 3:
        return False
    return asr_similarity(expected, actual) >= 0.45


def bytes_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def fmt_bytes(value: int | None) -> str:
    if not value:
        return "-"
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"


def model_size_label(family: str) -> str:
    meta = MODEL_META.get(family, {})
    if "size_label" in meta:
        return str(meta["size_label"])
    paths = [Path(p) for p in meta.get("size_paths", [])]
    total = sum(bytes_of(path) for path in paths)
    return fmt_bytes(total)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in [V2 / "all_results_scored.json", OUT / "sherpa_mobile_results.json"]:
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(row for row in loaded if row.get("status") == "ok" and row.get("output"))
    return rows


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
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        key = str(path)
        if key not in cache:
            cache[key] = embed(session, path)
        return cache[key]

    for row in rows:
        out = get(str(row.get("output", "")))
        refs = [get(str(row.get("ref_audio", "")))]
        refs.extend(get(str(path)) for path in row.get("aux_ref_audio_paths", []))
        refs = [ref for ref in refs if ref is not None]
        if out is None or not refs:
            row["speaker_cosine"] = None
            row["speaker_score_note"] = "not_scored"
            continue
        scores = [torch.dot(ref, out).item() for ref in refs]
        row["speaker_cosine_primary"] = round(scores[0], 4)
        row["speaker_cosine_avg"] = round(sum(scores) / len(scores), 4)
        row["speaker_cosine_max"] = round(max(scores), 4)
        row["speaker_cosine"] = row["speaker_cosine_max"]
        row["speaker_score_note"] = "campplus_cosine_max_over_reference_pool"


def transcribe_rows(rows: list[dict]) -> None:
    model = whisper.load_model("tiny")
    for index, row in enumerate(rows, 1):
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
            f"[{index:02d}/{len(rows)}] {row.get('family')} {row.get('pack_id')} {row.get('text_id')} "
            f"asr={row['asr_pass']} sim={row['asr_similarity']:.3f} {text}"
        )


def stats_by_model(rows: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for family in MODEL_ORDER:
        items = [row for row in rows if row.get("family") == family]
        if not items:
            continue
        secs = [float(row.get("seconds") or 0) for row in items if row.get("seconds") is not None]
        scores = [float(row["speaker_cosine"]) for row in items if row.get("speaker_cosine") is not None]
        asr = [bool(row.get("asr_pass")) for row in items]
        sims = [float(row.get("asr_similarity") or 0) for row in items]
        stats[family] = {
            "n": len(items),
            "avg_sec": statistics.mean(secs) if secs else None,
            "min_sec": min(secs) if secs else None,
            "max_sec": max(secs) if secs else None,
            "avg_score": statistics.mean(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "asr_pass": sum(asr),
            "asr_total": len(asr),
            "avg_asr_similarity": statistics.mean(sims) if sims else None,
            "size": model_size_label(family),
        }
    return stats


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
    if not path_value:
        return ""
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


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {esc(kind)}">{esc(text)}</span>'


def summary_cards(stats: dict[str, dict]) -> str:
    cards: list[str] = []
    for family in MODEL_ORDER:
        if family not in stats:
            continue
        meta = MODEL_META[family]
        item = stats[family]
        pass_ratio = f"{item['asr_pass']}/{item['asr_total']}"
        status = "可聽" if item["asr_pass"] == item["asr_total"] else "需人工聽"
        if item["asr_pass"] == 0:
            status = "淘汰候選"
        cards.append(
            f"""
            <article class="model-card">
              <div class="model-top">
                <h3>{esc(family)}</h3>
                {badge(status, "ok" if status == "可聽" else "warn")}
              </div>
              <div class="grid">
                <div><span>模型大小</span><b>{esc(item['size'])}</b></div>
                <div><span>平均生成</span><b>{fmt_sec(item['avg_sec'])}</b></div>
                <div><span>ASR 通過</span><b>{pass_ratio}</b></div>
                <div><span>聲紋分數</span><b>{fmt_score(item['avg_score'])}</b></div>
              </div>
              <p>{esc(meta['note'])}</p>
              <p class="small">{esc(meta['runtime'])} · {esc(meta['license'])}</p>
            </article>
            """
        )
    return "".join(cards)


def render_references(rows: list[dict], *, standalone: bool) -> str:
    refs: dict[str, dict] = {}
    for row in rows:
        if row.get("ref_id") and row.get("ref_audio"):
            refs.setdefault(row["ref_id"], row)
    pieces = []
    for ref_id, row in sorted(refs.items()):
        pieces.append(
            f"""
            <article class="sample ref">
              <div class="sample-head"><b>{esc(ref_id)}</b><span>{esc(', '.join(row.get('ref_files', [])))}</span></div>
              <p>{esc(row.get('ref_text'))}</p>
              {audio_tag(row.get('ref_audio', ''), standalone=standalone)}
            </article>
            """
        )
    return "".join(pieces)


def render_sample(row: dict, *, standalone: bool) -> str:
    pass_label = "ASR ok" if row.get("asr_pass") else "ASR check"
    return f"""
      <article class="sample">
        <div class="sample-head">
          <b>{esc(row.get('family'))}</b>
          <span>{esc(row.get('pack_id'))} · {fmt_sec(row.get('seconds'))} · sim {fmt_score(row.get('speaker_cosine'))}</span>
        </div>
        <div class="badges">{badge(pass_label, "ok" if row.get("asr_pass") else "warn")}{badge("ASR " + fmt_score(row.get("asr_similarity")), "")}</div>
        {audio_tag(row.get('output', ''), standalone=standalone)}
        <p class="asr"><b>ASR</b> {esc(row.get('asr_text', ''))}</p>
      </article>
    """


def render_samples(rows: list[dict], *, standalone: bool) -> str:
    rows = [row for row in rows if row.get("pack_id") == "pack_best2_7s"]
    by_text: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_text[row.get("text_id", "")].append(row)
    sections = []
    for text_id in ["tw_soft", "tw_calm", "tw_decide", "tw_confirm"]:
        items = sorted(by_text.get(text_id, []), key=lambda row: MODEL_ORDER.index(row["family"]) if row.get("family") in MODEL_ORDER else 999)
        if not items:
            continue
        expected = items[0].get("text", "")
        sections.append(
            f"""
            <section class="section">
              <h2>{esc(TEXT_LABELS.get(text_id, text_id))}</h2>
              <p class="expected">{esc(expected)}</p>
              <div class="sample-list">{''.join(render_sample(row, standalone=standalone) for row in items)}</div>
            </section>
            """
        )
    return "".join(sections)


def render_all_by_model(rows: list[dict], *, standalone: bool) -> str:
    sections = []
    for family in MODEL_ORDER:
        items = [row for row in rows if row.get("family") == family and row.get("pack_id") == "pack_best3_11s"]
        if not items:
            continue
        items = sorted(items, key=lambda row: row.get("text_id", ""))
        sections.append(
            f"""
            <details class="section">
              <summary>{esc(family)} · pack_best3_11s 補聽</summary>
              <div class="sample-list">{''.join(render_sample(row, standalone=standalone) for row in items)}</div>
            </details>
            """
        )
    return "".join(sections)


def render_not_run() -> str:
    items = []
    for row in NOT_RUN:
        items.append(
            f"""
            <article class="pending">
              <h3>{esc(row['name'])}</h3>
              <p>{esc(row['reason'])}</p>
              <a href="{esc(row['source'])}">{esc(row['source'])}</a>
            </article>
            """
        )
    return "".join(items)


def render_html(rows: list[dict], *, standalone: bool) -> str:
    stats = stats_by_model(rows)
    mode = "standalone：音檔已嵌入，適合上傳雲端/手機直接開" if standalone else "local：音檔用相對路徑載入"
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>授權女聲 Dataset Clone 模型試聽 v3</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --paper: #fffdf8;
      --ink: #22201c;
      --muted: #776f65;
      --line: #ddd4c6;
      --accent: #b84040;
      --ok: #226950;
      --warn: #a65f00;
      --soft: #eee7dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif;
      line-height: 1.58;
    }}
    main {{
      width: min(980px, 100%);
      margin: 0 auto;
      padding: 22px 14px 44px;
    }}
    header {{
      padding: 12px 2px 18px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }}
    h1 {{
      font-size: clamp(27px, 6.5vw, 46px);
      line-height: 1.08;
      margin: 7px 0 10px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 22px;
      line-height: 1.25;
      margin: 0 0 10px;
    }}
    h3 {{
      font-size: 17px;
      line-height: 1.3;
      margin: 0;
    }}
    p {{ margin: 8px 0; }}
    .small, .asr, .expected {{ color: var(--muted); font-size: 14px; }}
    .expected {{
      color: var(--ink);
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .callout {{
      background: #fff3e0;
      border: 1px solid #e5cfa8;
      border-radius: 8px;
      padding: 12px;
      margin: 14px 0;
    }}
    .models {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin: 16px 0 20px;
    }}
    .model-card, .section, .sample, .pending {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
    }}
    .model-top, .sample-head {{
      display: flex;
      gap: 8px;
      align-items: flex-start;
      justify-content: space-between;
    }}
    .sample-head span {{
      color: var(--muted);
      font-size: 12px;
      text-align: right;
      min-width: 96px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 11px 0;
    }}
    .grid div {{
      background: #f7f1e8;
      border-radius: 8px;
      padding: 8px;
    }}
    .grid span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .grid b {{
      font-size: 15px;
      overflow-wrap: anywhere;
    }}
    .section {{
      margin: 15px 0;
    }}
    .sample-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
      white-space: nowrap;
    }}
    .badge.ok {{ color: var(--ok); border-color: #9ccdbd; background: #eef9f5; }}
    .badge.warn {{ color: var(--warn); border-color: #e2bd75; background: #fff8e8; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }}
    details summary {{
      cursor: pointer;
      font-weight: 700;
      font-size: 18px;
    }}
    .pending {{
      margin: 10px 0;
    }}
    .pending a {{
      color: var(--accent);
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    @media (max-width: 620px) {{
      main {{ padding-inline: 12px; }}
      .models, .sample-list {{ grid-template-columns: 1fr; }}
      .model-top, .sample-head {{ display: block; }}
      .sample-head span {{ display: block; text-align: left; margin-top: 4px; }}
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Dataset clone audition v3 · {esc(mode)}</div>
    <h1>授權女聲 Dataset：各模型 Clone 試聽</h1>
    <p>同一組 reference pack、同四句日常台詞。每個輸出都補跑 Whisper tiny ASR gate：至少要聽得出中文內容，否則不能當可用候選。</p>
  </header>

  <section class="callout">
    <b>目前可直接比較：</b>
    CosyVoice2、F5 base、GPT-SoVITS、IndexTTS2、F5 fine-tune、Sherpa ZipVoice 8/4-step、Sherpa PocketTTS。
    Fish / Chatterbox / OpenVoice 本機還沒有完整 runtime，先列在「待補測」。
  </section>

  <section>
    <h2>模型總表</h2>
    <div class="models">{summary_cards(stats)}</div>
  </section>

  <section class="section">
    <h2>Reference Pack</h2>
    <p class="small">這是你授權資料裁好的 reference，不是模型輸出。用來判斷 clone 音色方向。</p>
    <div class="sample-list">{render_references(rows, standalone=standalone)}</div>
  </section>

  {render_samples(rows, standalone=standalone)}

  {render_all_by_model(rows, standalone=standalone)}

  <section class="section">
    <h2>待補測模型</h2>
    <p class="small">這些不是已生成音檔；目前只是根據官方資料和本機狀態列出下一輪可裝可測項。</p>
    {render_not_run()}
  </section>

  <section class="section">
    <h2>評分怎麼看</h2>
    <p><b>聲紋分數</b> 是 CAMPPlus reference/output cosine，只看音色接近，不保證內容正確。</p>
    <p><b>ASR 通過</b> 是 Whisper tiny 聽輸出是否還像中文且大致符合台詞。它抓不出所有音質問題，但能擋掉空白、重複亂碼、電子逼聲。</p>
    <p><b>生成時間</b> 沿用當次腳本量測；大模型多半含載入或 warmup 差異，Sherpa mobile 這輪是在同一 process 熱機後量測，較接近 app 內「模型已載入」情境。</p>
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

#!/usr/bin/env python3
"""Build a standalone HTML report for authorized Conan CosyVoice auditions."""

from __future__ import annotations

import base64
import collections
import html
import json
import mimetypes
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "conan_authorized_voice_refs_v1"
OUT = BASE / "cosy_clone_audition_v1"
RESULTS = OUT / "cosy_clone_results.json"
CLIP_MANIFEST = BASE / "clip_manifest.json"
REPORT_DIR = BASE / "reports" / "conan_authorized_cosy_clone_audition_v1"
LOCAL_HTML = REPORT_DIR / "conan_authorized_cosy_clone_audition_v1.html"
STANDALONE_HTML = REPORT_DIR / "conan_authorized_cosy_clone_audition_v1_standalone.html"
COSY_MODEL = ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"

ROLE_NAMES = {
    "haibara": "灰原",
    "conan": "柯南",
    "agasa": "阿笠博士",
}


def escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def fmt_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}TB"


def fmt_seconds(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
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
    return escape(os.path.relpath(path, REPORT_DIR))


def audio_tag(path_value: str, *, standalone: bool) -> str:
    src = audio_src(path_value, standalone=standalone)
    if not src:
        return '<div class="missing">音檔不存在</div>'
    return f'<audio controls preload="none" src="{src}"></audio>'


def render(*, standalone: bool) -> str:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    clips = json.loads(CLIP_MANIFEST.read_text(encoding="utf-8"))
    clips_by_path = {str(row["clip"]): row for row in clips}
    by_role: dict[str, list[dict]] = collections.defaultdict(list)
    first_by_role: dict[str, dict] = {}
    for row in rows:
        by_role[str(row["role"])].append(row)
        first_by_role.setdefault(str(row["role"]), row)

    role_sections = []
    for role in ["haibara", "conan", "agasa"]:
        first = first_by_role[role]
        refs = []
        for ref_file in first["ref_files"]:
            ref_row = clips_by_path.get(ref_file, {})
            refs.append(
                f"""
                <div class="ref-clip">
                  <div class="meta">
                    <b>{escape(Path(ref_file).name)}</b>
                    <span>{fmt_seconds(ref_row.get('duration'))}</span>
                  </div>
                  <div class="transcript">{escape(ref_row.get('transcript', ''))}</div>
                  {audio_tag(ref_file, standalone=standalone)}
                </div>
                """
            )

        samples = []
        for row in sorted(by_role[role], key=lambda item: item["text_id"]):
            status = "OK" if row.get("status") == "ok" else escape(row.get("error", "failed"))
            samples.append(
                f"""
                <article class="sample">
                  <div class="sample-head">
                    <span>{escape(row['text_id'])}</span>
                    <b>{fmt_seconds(row.get('seconds'))} / {status}</b>
                  </div>
                  <p>{escape(row['text'])}</p>
                  {audio_tag(row.get('output', ''), standalone=standalone)}
                </article>
                """
            )

        role_sections.append(
            f"""
            <section class="role" id="{escape(role)}">
              <div class="role-title">
                <h2>{escape(ROLE_NAMES[role])}</h2>
                <div>{escape(first['pack_id'])} / reference {fmt_seconds(first.get('ref_duration'))}</div>
              </div>
              <div class="grid two">
                <div>
                  <h3>Reference pack</h3>
                  <p class="note">CosyVoice2 這次吃的是這個三段合併 reference。ASR 草稿如果錯，clone 可能會受影響。</p>
                  {audio_tag(first['ref_audio'], standalone=standalone)}
                  <div class="transcript pack-text">{escape(first['ref_text'])}</div>
                </div>
                <div>
                  <h3>選用原音片段</h3>
                  {''.join(refs)}
                </div>
              </div>
              <h3>CosyVoice2 clone samples</h3>
              <div class="samples">{''.join(samples)}</div>
            </section>
            """
        )

    standalone_note = "音檔已嵌入 HTML，可離線打開" if standalone else "本機 HTML，音檔以相對路徑載入"
    html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conan Authorized Voices - CosyVoice2 Clone Audition</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #20201d;
      --muted: #68645d;
      --line: #ddd8cf;
      --paper: #f7f5ef;
      --panel: #fffdf8;
      --accent: #b44337;
      --soft: #ece8de;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "PingFang TC", "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      letter-spacing: 0;
    }}
    header {{
      padding: 42px clamp(20px, 5vw, 72px) 28px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf6;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(28px, 4vw, 48px);
      line-height: 1.12;
      font-weight: 650;
    }}
    h2 {{ margin: 0; font-size: 28px; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ line-height: 1.65; }}
    audio {{ width: 100%; height: 38px; }}
    .sub {{ max-width: 980px; color: var(--muted); font-size: 16px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .badge {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 8px 12px;
      border-radius: 4px;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{ padding: 24px clamp(16px, 4vw, 64px) 56px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .stat, .role, .sample, .ref-clip {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .stat {{ padding: 16px; }}
    .stat b {{ display: block; margin-top: 6px; font-size: 20px; }}
    .stat span {{ color: var(--muted); font-size: 13px; }}
    .role {{ padding: 22px; margin-top: 18px; }}
    .role-title {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    .role-title div {{ color: var(--muted); }}
    .grid.two {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .sample, .ref-clip {{ padding: 14px; }}
    .sample-head, .ref-clip .meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .sample p {{ margin: 0 0 10px; font-size: 16px; }}
    .transcript {{
      margin: 10px 0;
      padding: 10px;
      background: var(--soft);
      border-radius: 4px;
      color: #48443d;
      line-height: 1.55;
      font-size: 14px;
    }}
    .pack-text {{ min-height: 92px; }}
    .note {{ color: var(--muted); margin-top: 0; }}
    .ref-clip + .ref-clip {{ margin-top: 10px; }}
    .missing {{ color: var(--accent); font-size: 14px; }}
    @media (max-width: 840px) {{
      .summary, .grid.two, .samples {{ grid-template-columns: 1fr; }}
      .role-title {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Conan 授權聲音 CosyVoice2 Clone 試聽</h1>
    <p class="sub">這份報告只測 CosyVoice2-0.5B zero-shot clone：用你提供的授權角色聲音做 reference，分別生成灰原、柯南、阿笠博士的測試台詞。重點先聽像不像與穩不穩，再決定要不要把其中某個角色做成後續 ZipVoice/Matcha 手機學生模型。</p>
    <div class="badges">
      <span class="badge">{escape(standalone_note)}</span>
      <span class="badge">生成時間是 Mac teacher 測試，不是手機速度</span>
      <span class="badge">ASR 逐字稿為草稿，可人工修正後重跑</span>
    </div>
  </header>
  <main>
    <section class="summary">
      <div class="stat"><span>模型</span><b>CosyVoice2-0.5B</b></div>
      <div class="stat"><span>本機模型資料夾</span><b>{fmt_bytes(dir_size(COSY_MODEL))}</b></div>
      <div class="stat"><span>角色</span><b>灰原 / 柯南 / 阿笠</b></div>
      <div class="stat"><span>Clone samples</span><b>{len(rows)} 句</b></div>
    </section>
    {''.join(role_sections)}
  </main>
</body>
</html>
"""
    return html_doc


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_HTML.write_text(render(standalone=False), encoding="utf-8")
    STANDALONE_HTML.write_text(render(standalone=True), encoding="utf-8")
    print(LOCAL_HTML)
    print(STANDALONE_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build cloud-friendly embedded-audio review pages for the 500 Cosy clips."""

from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
CORPUS_DIR = BASE / "teacher_cosy_clear_best2_golden_daily_500_v1"
MANIFEST = CORPUS_DIR / "manifest.json"
REPORT_DIR = BASE / "reports" / "cosy_teacher_500_cloud_audio_v1"
M4A_DIR = REPORT_DIR / "m4a_48k"
PAGE_SIZE = 100


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_items() -> list[dict[str, object]]:
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise TypeError(f"Expected list manifest: {MANIFEST}")
    return items


def resolve_audio_path(item: dict[str, object]) -> Path:
    path = Path(str(item["audio"]))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def encode_audio(items: list[dict[str, object]], limit: int | None = None) -> None:
    M4A_DIR.mkdir(parents=True, exist_ok=True)
    selected = items[:limit] if limit else items
    for index, item in enumerate(selected, start=1):
        item_id = str(item["id"])
        src = resolve_audio_path(item)
        dst = M4A_DIR / f"{item_id}.m4a"
        if dst.exists() and dst.stat().st_size > 0 and dst.stat().st_mtime >= src.stat().st_mtime:
            if index % 25 == 0 or index == len(selected):
                print(f"[encode] skip {index}/{len(selected)}")
            continue
        print(f"[encode] {index}/{len(selected)} {item_id}")
        subprocess.run(
            [
                "afconvert",
                "-f",
                "m4af",
                "-d",
                "aac",
                "-b",
                "48000",
                str(src),
                str(dst),
            ],
            check=True,
        )


def data_uri(path: Path) -> str:
    return "data:audio/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_clip(item: dict[str, object], global_index: int) -> str:
    item_id = str(item["id"])
    m4a = M4A_DIR / f"{item_id}.m4a"
    text = str(item.get("text", ""))
    category = str(item.get("category", ""))
    return f"""
    <article class="clip" data-id="{esc(item_id)}" data-text="{esc(text)}" data-category="{esc(category)}">
      <div class="clip-head">
        <div>
          <div class="clip-index">#{global_index:03d} · {esc(item_id)}</div>
          <h3>{esc(text)}</h3>
        </div>
        <span class="tag">{esc(category)}</span>
      </div>
      <audio controls preload="none" src="{data_uri(m4a)}"></audio>
      <div class="review-grid">
        <label>
          <span>判定</span>
          <select data-role="status">
            <option value="">未選</option>
            <option value="keep">留</option>
            <option value="maybe">備選</option>
            <option value="reject">刪</option>
            <option value="regen">重生</option>
          </select>
        </label>
        <label>
          <span>註記</span>
          <input data-role="note" placeholder="像 / 字跑掉 / 尾音好 / 不自然">
        </label>
      </div>
    </article>
    """


def build_page(items: list[dict[str, object]], part: int, total_parts: int) -> str:
    start = (part - 1) * PAGE_SIZE
    end = min(len(items), start + PAGE_SIZE)
    clips = "\n".join(build_clip(item, start + offset + 1) for offset, item in enumerate(items[start:end]))
    prev_link = f"part_{part - 1:02d}.html" if part > 1 else ""
    next_link = f"part_{part + 1:02d}.html" if part < total_parts else ""
    nav = []
    if prev_link:
        nav.append(f'<a class="button" href="{prev_link}">上一頁</a>')
    nav.append(f'<a class="button secondary" href="index.html">索引</a>')
    if next_link:
        nav.append(f'<a class="button" href="{next_link}">下一頁</a>')
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosy Teacher 500 雲端試聽 Part {part}</title>
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
      --soft: #eee5da;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      font-size: 16px;
      line-height: 1.55;
    }}
    main {{ width: min(980px, 100%); margin: 0 auto; padding: 18px 12px 64px; }}
    .eyebrow {{ color: var(--red); font-size: 13px; font-weight: 800; }}
    h1 {{ margin: 6px 0 8px; font-size: clamp(28px, 8vw, 44px); line-height: 1.08; letter-spacing: 0; }}
    h3 {{ margin: 4px 0 10px; font-size: 17px; line-height: 1.45; letter-spacing: 0; }}
    p {{ margin: 7px 0; }}
    .lead {{ color: var(--muted); max-width: 760px; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(245, 242, 236, .96);
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
      backdrop-filter: blur(8px);
    }}
    .toolbar-inner {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 8px;
      align-items: end;
    }}
    input, select, button, textarea, .button {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--paper);
      color: var(--ink);
      font: inherit;
      min-height: 40px;
      padding: 8px 10px;
    }}
    .button, button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      cursor: pointer;
      background: var(--blue);
      color: white;
      border-color: var(--blue);
      font-weight: 800;
      white-space: nowrap;
    }}
    .button.secondary {{ background: var(--paper); color: var(--blue); }}
    label span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 800; margin-bottom: 3px; }}
    .clip {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin: 10px 0;
    }}
    .clip[data-current-status="keep"] {{ border-color: rgba(63, 118, 89, .75); }}
    .clip[data-current-status="maybe"] {{ border-color: rgba(141, 100, 33, .75); }}
    .clip[data-current-status="reject"] {{ opacity: .7; }}
    .clip-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
    .clip-index {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
    .tag {{ flex: 0 0 auto; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); padding: 4px 8px; font-size: 12px; background: var(--soft); }}
    audio {{ width: 100%; margin: 4px 0 8px; }}
    .review-grid {{ display: grid; grid-template-columns: 150px 1fr; gap: 8px; }}
    textarea {{ width: 100%; min-height: 170px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }}
    .export {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 14px 0; }}
    .nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }}
    @media (max-width: 720px) {{
      main {{ padding-inline: 10px; }}
      .toolbar-inner {{ grid-template-columns: 1fr 1fr; }}
      .toolbar-inner label {{ grid-column: 1 / -1; }}
      .review-grid {{ grid-template-columns: 1fr; }}
      .clip-head {{ display: block; }}
      .tag {{ display: inline-block; margin-bottom: 4px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Cloud audio review · Part {part}/{total_parts}</div>
      <h1>Cosy Teacher 500 句雲端試聽</h1>
      <p class="lead">這頁把音檔壓成 48kbps AAC 並直接嵌入 HTML，離開家裡 Wi‑Fi 也能聽。第 {start + 1} 到 {end} 句，共 {len(items)} 句。</p>
      <div class="nav">{''.join(nav)}</div>
    </header>
    <section class="toolbar">
      <div class="toolbar-inner">
        <label><span>搜尋本頁文字 / ID</span><input id="search" placeholder="例如：慢慢來、珍奶、0030"></label>
        <button id="exportBtn" type="button">匯出本頁選擇</button>
      </div>
    </section>
    <section class="export">
      <textarea id="exportBox" placeholder="按匯出後，這裡會出現本頁已標記的句子。"></textarea>
    </section>
    <div id="clips">{clips}</div>
    <div class="nav">{''.join(nav)}</div>
  </main>
  <script>
    const storageKey = 'cosy_teacher_500_cloud_audio_v1_part_{part:02d}';
    const clips = Array.from(document.querySelectorAll('.clip'));
    const search = document.getElementById('search');
    const exportBox = document.getElementById('exportBox');
    function loadState() {{
      try {{ return JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }}
      catch {{ return {{}}; }}
    }}
    function saveState(state) {{ localStorage.setItem(storageKey, JSON.stringify(state)); }}
    function applySavedState() {{
      const state = loadState();
      for (const clip of clips) {{
        const saved = state[clip.dataset.id] || {{ status: '', note: '' }};
        clip.querySelector('[data-role="status"]').value = saved.status || '';
        clip.querySelector('[data-role="note"]').value = saved.note || '';
        clip.dataset.currentStatus = saved.status || '';
      }}
    }}
    function updateState(event) {{
      const clip = event.target.closest('.clip');
      const state = loadState();
      state[clip.dataset.id] = {{
        status: clip.querySelector('[data-role="status"]').value,
        note: clip.querySelector('[data-role="note"]').value
      }};
      clip.dataset.currentStatus = state[clip.dataset.id].status;
      saveState(state);
    }}
    function render() {{
      const q = search.value.trim().toLowerCase();
      for (const clip of clips) {{
        clip.hidden = q && !(clip.dataset.id + ' ' + clip.dataset.text).toLowerCase().includes(q);
      }}
    }}
    function exportSelection() {{
      const state = loadState();
      const rows = clips.map(clip => {{
        const saved = state[clip.dataset.id] || {{ status: '', note: '' }};
        return {{ id: clip.dataset.id, category: clip.dataset.category, text: clip.dataset.text, status: saved.status || '', note: saved.note || '' }};
      }}).filter(row => row.status || row.note);
      exportBox.value = JSON.stringify(rows, null, 2);
      exportBox.focus();
      exportBox.select();
    }}
    document.addEventListener('change', event => {{
      if (event.target.matches('[data-role="status"]')) updateState(event);
    }});
    document.addEventListener('input', event => {{
      if (event.target.matches('[data-role="note"]')) updateState(event);
      if (event.target === search) render();
    }});
    document.getElementById('exportBtn').addEventListener('click', exportSelection);
    applySavedState();
  </script>
</body>
</html>
"""


def build_index(total_parts: int) -> str:
    links = "\n".join(
        f'<a class="button" href="part_{part:02d}.html">Part {part:02d} · {((part - 1) * PAGE_SIZE) + 1}-{min(part * PAGE_SIZE, 500)}</a>'
        for part in range(1, total_parts + 1)
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosy Teacher 500 雲端試聽索引</title>
  <style>
    body {{
      margin: 0;
      background: #f5f2ec;
      color: #24211d;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      font-size: 17px;
      line-height: 1.6;
    }}
    main {{ width: min(760px, 100%); margin: 0 auto; padding: 24px 14px 56px; }}
    h1 {{ font-size: clamp(30px, 8vw, 44px); line-height: 1.08; letter-spacing: 0; margin: 0 0 10px; }}
    p {{ color: #6b6258; }}
    .links {{ display: grid; gap: 10px; margin-top: 18px; }}
    .button {{
      display: block;
      text-decoration: none;
      background: #255f7f;
      color: white;
      border-radius: 8px;
      padding: 14px 16px;
      font-weight: 800;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Cosy Teacher 500 句雲端試聽</h1>
    <p>每頁 100 句，音檔已嵌入 HTML。上傳到 Drive 後請用我給你的 Drive 連結開，這個本機索引的相對連結只在同一資料夾內有效。</p>
    <div class="links">{links}</div>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-encode", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    items = load_items()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_encode:
        encode_audio(items, args.limit)

    total_items = min(args.limit or len(items), len(items))
    selected = items[:total_items]
    total_parts = (len(selected) + PAGE_SIZE - 1) // PAGE_SIZE
    for part in range(1, total_parts + 1):
        page = REPORT_DIR / f"part_{part:02d}.html"
        page.write_text(build_page(selected, part, total_parts), encoding="utf-8")
        print(f"[page] {page} {page.stat().st_size / 1024 / 1024:.1f} MB")

    index = REPORT_DIR / "index.html"
    index.write_text(build_index(total_parts), encoding="utf-8")
    summary = {
        "items": len(selected),
        "page_size": PAGE_SIZE,
        "parts": total_parts,
        "format": "AAC/M4A 48kbps embedded as data:audio/mp4",
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[index] {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

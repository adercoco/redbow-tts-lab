#!/usr/bin/env python3
"""Build a local review page for selecting Cosy teacher corpus clips."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
CORPUS_DIR = BASE / "teacher_cosy_clear_best2_golden_daily_500_v1"
MANIFEST = CORPUS_DIR / "manifest.json"
REPORT_DIR = BASE / "reports" / "cosy_teacher_500_selection_v1"
OUT_HTML = REPORT_DIR / "cosy_teacher_500_selection_v1.html"
OUT_CSV = REPORT_DIR / "cosy_teacher_500_selection_v1.csv"
OUT_JSON = REPORT_DIR / "cosy_teacher_500_selection_items.json"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def resolve_audio_uri(item: dict[str, object]) -> str:
    audio = Path(str(item["audio"]))
    if not audio.is_absolute():
        audio = ROOT / audio
    return audio.resolve().as_uri()


def load_items() -> list[dict[str, object]]:
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise TypeError(f"Expected list manifest: {MANIFEST}")
    return items


def write_csv(items: list[dict[str, object]]) -> None:
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "category", "text", "audio", "status", "note"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item.get("id", ""),
                    "category": item.get("category", ""),
                    "text": item.get("text", ""),
                    "audio": item.get("audio", ""),
                    "status": "",
                    "note": "",
                }
            )


def build_rows(items: list[dict[str, object]]) -> str:
    rows: list[str] = []
    for index, item in enumerate(items, start=1):
        item_id = str(item.get("id", ""))
        text = str(item.get("text", ""))
        category = str(item.get("category", ""))
        status = str(item.get("status", ""))
        audio_uri = resolve_audio_uri(item)
        rows.append(
            f"""
            <article class="clip" data-id="{esc(item_id)}" data-text="{esc(text)}" data-category="{esc(category)}">
              <div class="clip-head">
                <div>
                  <div class="clip-index">#{index:03d} · {esc(item_id)}</div>
                  <h3>{esc(text)}</h3>
                </div>
                <span class="tag">{esc(category)}</span>
              </div>
              <audio controls preload="none" src="{esc(audio_uri)}"></audio>
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
                  <input data-role="note" placeholder="例如：像、字跑掉、尾音好、太機械">
                </label>
              </div>
              <div class="tiny">{esc(status)} · {esc(str(item.get("teacher_model", "")))}</div>
            </article>
            """
        )
    return "\n".join(rows)


def build_html(items: list[dict[str, object]]) -> str:
    rows = build_rows(items)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cosy Teacher 500 句挑選表</title>
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
      --amber: #8d6421;
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
    main {{
      width: min(1080px, 100%);
      margin: 0 auto;
      padding: 18px 14px 64px;
    }}
    header {{
      padding: 14px 0 8px;
    }}
    .eyebrow {{
      color: var(--red);
      font-size: 13px;
      font-weight: 800;
    }}
    h1 {{
      margin: 6px 0 8px;
      font-size: clamp(30px, 8vw, 48px);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    p {{ margin: 7px 0; }}
    .lead {{ color: var(--muted); max-width: 760px; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 4;
      background: rgba(245, 242, 236, .96);
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
      backdrop-filter: blur(8px);
    }}
    .toolbar-inner {{
      display: grid;
      grid-template-columns: 1.2fr .7fr .7fr auto;
      gap: 8px;
      align-items: end;
    }}
    label span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 3px;
    }}
    input, select, button, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--paper);
      color: var(--ink);
      font: inherit;
      min-height: 40px;
      padding: 8px 10px;
    }}
    button {{
      cursor: pointer;
      background: var(--blue);
      color: white;
      border-color: var(--blue);
      font-weight: 800;
      white-space: nowrap;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }}
    .stat {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .stat b {{
      display: block;
      color: var(--blue);
      font-size: 24px;
      line-height: 1.1;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .clip {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin: 10px 0;
    }}
    .clip[data-current-status="keep"] {{ border-color: rgba(63, 118, 89, .75); }}
    .clip[data-current-status="maybe"] {{ border-color: rgba(141, 100, 33, .75); }}
    .clip[data-current-status="reject"] {{ opacity: .68; }}
    .clip-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
    }}
    .clip-index {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    h3 {{
      margin: 4px 0 10px;
      font-size: 17px;
      line-height: 1.45;
      letter-spacing: 0;
    }}
    .tag {{
      flex: 0 0 auto;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      padding: 4px 8px;
      font-size: 12px;
      background: var(--soft);
    }}
    audio {{
      width: 100%;
      margin: 4px 0 8px;
    }}
    .review-grid {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 8px;
    }}
    .tiny {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      word-break: break-word;
    }}
    .export {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-top: 14px;
    }}
    textarea {{
      min-height: 180px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
    .path {{
      color: var(--muted);
      word-break: break-word;
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      main {{ padding-inline: 10px; }}
      .toolbar-inner {{ grid-template-columns: 1fr 1fr; }}
      .toolbar-inner label:first-child {{ grid-column: 1 / -1; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .review-grid {{ grid-template-columns: 1fr; }}
      .clip-head {{ display: block; }}
      .tag {{ display: inline-block; margin-bottom: 4px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">CosyVoice2 teacher corpus review</div>
      <h1>500 句挑選表</h1>
      <p class="lead">先用耳朵篩掉不像、字跑掉、情緒怪、雜音、節奏不自然的老師語料。下一輪 ZipVoice fine-tune 只吃你選過的好句，聲音會比盲吃 500 句穩。</p>
      <p class="path">CSV：{esc(str(OUT_CSV))}</p>
    </header>

    <section class="toolbar">
      <div class="toolbar-inner">
        <label>
          <span>搜尋文字 / ID</span>
          <input id="search" placeholder="例如：慢慢來、珍奶、golden_daily_0030">
        </label>
        <label>
          <span>類別</span>
          <select id="categoryFilter">
            <option value="">全部</option>
          </select>
        </label>
        <label>
          <span>判定</span>
          <select id="statusFilter">
            <option value="">全部</option>
            <option value="keep">留</option>
            <option value="maybe">備選</option>
            <option value="reject">刪</option>
            <option value="regen">重生</option>
            <option value="unpicked">未選</option>
          </select>
        </label>
        <button id="exportBtn" type="button">匯出</button>
      </div>
    </section>

    <section class="stats" aria-label="review stats">
      <div class="stat"><b id="totalCount">{len(items)}</b><span>總句數</span></div>
      <div class="stat"><b id="keepCount">0</b><span>留</span></div>
      <div class="stat"><b id="maybeCount">0</b><span>備選</span></div>
      <div class="stat"><b id="rejectCount">0</b><span>刪</span></div>
      <div class="stat"><b id="regenCount">0</b><span>重生</span></div>
    </section>

    <section class="export">
      <strong>挑選標準</strong>
      <p>留：像 reference、台灣感自然、字都有講對、沒有音樂/男聲/唱歌/爆音。重生：文字好但音色或發音壞掉。刪：句子或聲音不適合拿來教學生。</p>
      <textarea id="exportBox" placeholder="按匯出後，這裡會出現你選的 ID / 文字 / 註記。"></textarea>
    </section>

    <div id="clips">
      {rows}
    </div>
  </main>
  <script>
    const storageKey = 'cosy_teacher_500_selection_v1';
    const clips = Array.from(document.querySelectorAll('.clip'));
    const search = document.getElementById('search');
    const categoryFilter = document.getElementById('categoryFilter');
    const statusFilter = document.getElementById('statusFilter');
    const exportBox = document.getElementById('exportBox');
    const categories = Array.from(new Set(clips.map(c => c.dataset.category).filter(Boolean))).sort();
    for (const category of categories) {{
      const option = document.createElement('option');
      option.value = category;
      option.textContent = category;
      categoryFilter.appendChild(option);
    }}
    function loadState() {{
      try {{ return JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }}
      catch {{ return {{}}; }}
    }}
    function saveState(state) {{
      localStorage.setItem(storageKey, JSON.stringify(state));
    }}
    function getStateFor(clip, state) {{
      return state[clip.dataset.id] || {{ status: '', note: '' }};
    }}
    function applySavedState() {{
      const state = loadState();
      for (const clip of clips) {{
        const saved = getStateFor(clip, state);
        clip.querySelector('[data-role="status"]').value = saved.status || '';
        clip.querySelector('[data-role="note"]').value = saved.note || '';
        clip.dataset.currentStatus = saved.status || '';
      }}
    }}
    function updateStateFromInput(event) {{
      const clip = event.target.closest('.clip');
      const state = loadState();
      const status = clip.querySelector('[data-role="status"]').value;
      const note = clip.querySelector('[data-role="note"]').value;
      state[clip.dataset.id] = {{ status, note }};
      clip.dataset.currentStatus = status;
      saveState(state);
      render();
    }}
    function render() {{
      const q = search.value.trim().toLowerCase();
      const category = categoryFilter.value;
      const status = statusFilter.value;
      const counts = {{ keep: 0, maybe: 0, reject: 0, regen: 0 }};
      let visible = 0;
      for (const clip of clips) {{
        const currentStatus = clip.dataset.currentStatus || '';
        if (counts[currentStatus] !== undefined) counts[currentStatus] += 1;
        const textMatch = !q || (clip.dataset.text + ' ' + clip.dataset.id).toLowerCase().includes(q);
        const categoryMatch = !category || clip.dataset.category === category;
        const statusMatch = !status || (status === 'unpicked' ? !currentStatus : currentStatus === status);
        const show = textMatch && categoryMatch && statusMatch;
        clip.hidden = !show;
        if (show) visible += 1;
      }}
      document.getElementById('totalCount').textContent = visible + '/' + clips.length;
      document.getElementById('keepCount').textContent = counts.keep;
      document.getElementById('maybeCount').textContent = counts.maybe;
      document.getElementById('rejectCount').textContent = counts.reject;
      document.getElementById('regenCount').textContent = counts.regen;
    }}
    function exportSelection() {{
      const state = loadState();
      const rows = clips
        .map(clip => {{
          const saved = getStateFor(clip, state);
          return {{
            id: clip.dataset.id,
            category: clip.dataset.category,
            text: clip.dataset.text,
            status: saved.status || '',
            note: saved.note || ''
          }};
        }})
        .filter(row => row.status || row.note);
      exportBox.value = JSON.stringify(rows, null, 2);
      exportBox.focus();
      exportBox.select();
    }}
    document.addEventListener('change', event => {{
      if (event.target.matches('[data-role="status"]')) updateStateFromInput(event);
    }});
    document.addEventListener('input', event => {{
      if (event.target.matches('[data-role="note"]')) updateStateFromInput(event);
      if (event.target === search) render();
    }});
    categoryFilter.addEventListener('change', render);
    statusFilter.addEventListener('change', render);
    document.getElementById('exportBtn').addEventListener('click', exportSelection);
    applySavedState();
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    items = load_items()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(items)
    OUT_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_HTML.write_text(build_html(items), encoding="utf-8")
    print(OUT_HTML)
    print(OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

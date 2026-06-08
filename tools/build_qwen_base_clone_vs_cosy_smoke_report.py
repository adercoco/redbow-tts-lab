from __future__ import annotations

import base64
import html
from pathlib import Path


ROOT = Path("/Users/ader/Documents/App")
OUT = ROOT / "distillation/taiwan_mandarin_low_r/reports/qwen_base_clone_vs_cosy_smoke_v1"
REPORT = OUT / "qwen_base_clone_vs_cosy_smoke_v1_standalone.html"

REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"
COSY_DIR = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/clone_audition_v4_original_models/audio/cosyvoice2_0p5b_pack_best2"
QWEN_DIR = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/qwen_base_clone_smoke_v1/audio/qwen3_0p6b_base_pack_best2"

ROWS = [
    ("line_01", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("line_02", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("line_03", "我觉得这件事情可以慢慢来，不需要马上决定。"),
]


def data_audio(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def audio(path: Path) -> str:
    return f'<audio controls preload="metadata" src="{data_audio(path)}"></audio>'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row_html = []
    for line_id, text in ROWS:
        cosy = COSY_DIR / f"{line_id}.wav"
        qwen = QWEN_DIR / f"{line_id}_000.wav"
        row_html.append(
            f"""
            <tr>
              <td><b>{html.escape(line_id)}</b><p>{html.escape(text)}</p></td>
              <td>{audio(cosy)}<small>CosyVoice2-0.5B zero-shot clone</small></td>
              <td>{audio(qwen)}<small>Qwen3-TTS 0.6B Base 4bit clone smoke</small></td>
            </tr>
            """
        )

    doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Base clone vs Cosy clone smoke v1</title>
<style>
:root {{ --paper:#f7f5ef; --ink:#25231f; --muted:#69645b; --line:#ddd6c8; --panel:#fffdf8; --red:#b7202f; --soft:#efe7d8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.7 -apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif; }}
main {{ max-width:1260px; margin:0 auto; padding:42px 32px 72px; }}
h1 {{ font-size:34px; margin:0 0 10px; }}
.lead {{ max-width:980px; font-size:18px; }}
.note {{ background:#fff8eb; border:1px solid #ead8b7; border-radius:8px; padding:14px 16px; margin:18px 0; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
th,td {{ text-align:left; vertical-align:top; padding:13px 14px; border-bottom:1px solid var(--line); }}
th {{ background:var(--soft); }}
tr:last-child td {{ border-bottom:0; }}
td:first-child {{ width:30%; }}
audio {{ width:100%; margin:4px 0 5px; }}
small {{ display:block; color:var(--muted); }}
p {{ margin:6px 0 0; color:var(--muted); }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
</style>
</head>
<body>
<main>
  <h1>Qwen Base clone vs Cosy clone smoke v1</h1>
  <p class="lead">同一個授權女聲 reference pack、同一句話，直接比較 CosyVoice2 zero-shot clone 和 Qwen3-TTS 0.6B Base 4bit clone smoke。這是先確認 Qwen Base clone 方向值不值得放大，不是最終 1.7B Base 正式報告。</p>
  <div class="note">
    <b>Reference：</b><code>pack_best2_7s.wav</code>，逐字稿：<code>所以我当时就说,我想要做一张疗愈人的专辑。 开始当然就是我们的提案会议,我就提出了因为多年</code>
    <div style="margin-top:8px">{audio(REF_AUDIO)}<small>同一份 reference audio，兩個模型都餵這個。</small></div>
  </div>
  <table>
    <thead><tr><th>測試句</th><th>CosyVoice2</th><th>Qwen Base clone</th></tr></thead>
    <tbody>
      {''.join(row_html)}
    </tbody>
  </table>
  <div class="note">
    <b>下一步：</b>如果這三句 Qwen Base clone 有值得聽的方向，再跑完整同 dataset：6-12 句、speaker cosine、ASR 可懂度、生成時間與 peak RSS。再決定要不要下載/跑 Qwen3-TTS 1.7B Base。
  </div>
</main>
</body>
</html>
"""
    REPORT.write_text(doc, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

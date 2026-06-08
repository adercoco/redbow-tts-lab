from __future__ import annotations

import base64
import re
from pathlib import Path


ROOT = Path("/Users/ader/Documents/App")
REPORT_DIR = ROOT / "distillation/taiwan_mandarin_low_r/reports/target_route_deep_technical_v1"
SRC = REPORT_DIR / "index.html"
DST = REPORT_DIR / "target_route_deep_technical_v1_standalone.html"


def embed_audio(match: re.Match[str]) -> str:
    src = match.group(1)
    audio_path = REPORT_DIR / src
    data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f'src="data:audio/wav;base64,{data}"'


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    embedded = re.sub(r'src="(assets/[^"]+\.wav)"', embed_audio, html)
    embedded = embedded.replace(
        "紅色蝴蝶結 TTS 主線深度技術報告 v1｜桌機好讀版",
        "紅色蝴蝶結 TTS 主線深度技術報告 v1｜桌機單檔版",
    )
    embedded = embedded.replace(
        "Generated at /Users/ader/Documents/App/distillation/taiwan_mandarin_low_r/reports/target_route_deep_technical_v1/index.html",
        "Standalone HTML: audio is embedded as data:audio/wav;base64. No assets folder is required.",
    )
    DST.write_text(embedded, encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()

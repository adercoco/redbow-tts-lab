#!/usr/bin/env python3
"""Build phone-readable PNG cards for the TTS distillation report."""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/ader/Documents/App")
OUT_DIR = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "reports"
    / "distillation_methodology"
    / "phone_cards"
)
FONT_REGULAR = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"

W = 1290
H = 3600
SCALE = W / 430


def s(value: int | float) -> int:
    return round(value * SCALE)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, s(size))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            trial = current + char
            if draw.textbbox((0, 0), trial, font=font_obj)[2] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font_obj, max_width):
        if line:
            draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=s(8), fill=fill, outline=outline, width=s(1) if outline else 1)


@dataclass
class Metric:
    label: str
    value: str
    note: str


@dataclass
class Card:
    filename: str
    eyebrow: str
    title: str
    intro: str
    hero: str
    sections: list[tuple[str, str]]
    metrics: list[Metric]
    footer: str


CARDS = [
    Card(
        filename="01_distillation_principle.png",
        eyebrow="TTS 蒸餾原理",
        title="大老師教小學生，學的是聲音軌跡",
        intro="TTS 蒸餾不是只學文字答案。學生要學到音色、發音、停頓、音高、能量、語速和尾音。",
        hero="完整蒸餾 = 不再依賴 teacher reference audio",
        sections=[
            ("Teacher", "Qwen3 1.7B 產生台灣溫柔女生聲音，負責定義目標。"),
            ("Student", "ZipVoice / ZipVoice-Distill 學會同一種聲音，但要更小、更快、可部署。"),
            ("關鍵差異", "reference prompt demo 只是臨場模仿；完整蒸餾是聲音進入學生權重。"),
        ],
        metrics=[
            Metric("老師模型", "約 2.2GB", "聲音好，但手機離線太重"),
            Metric("學生目標", "ONNX int8", "可交給手機 runtime"),
        ],
        footer="判斷標準：不用 Qwen、不用 teacher audio，學生自己能講出目標聲線。",
    ),
    Card(
        filename="02_pipeline.png",
        eyebrow="我們實際做法",
        title="Qwen teacher → ZipVoice-Distill → ONNX int8",
        intro="這次已經跑過第一輪完整鏈路，不只是拿 teacher reference 讓 ZipVoice zero-shot 模仿。",
        hero="第一輪完整鏈路已完成，但還不是產品級完成",
        sections=[
            ("1. Teacher corpus", "Qwen3 產生 500 句 taiwan_mandarin_low_r 老師語料。"),
            ("2. Prosody curriculum", "用 F0、能量、停頓挑 Phase 1 平穩核心與 Phase 2 自然尾音。"),
            ("3. Fine-tune", "ZipVoice 從 Qwen teacher corpus 學音色與台灣語調。"),
            ("4. Distill + export", "ZipVoice-Distill 壓低生成成本，再匯出 ONNX int8。"),
        ],
        metrics=[
            Metric("Phase 1", "300 句", "早期平穩核心"),
            Metric("Phase 2", "197 句", "後期加回自然尾音"),
            Metric("Train TSV", "284 句", "進 ZipVoice fine-tune"),
            Metric("Dev TSV", "32 句", "檢查過度平坦"),
        ],
        footer="早期偏平是 curriculum，不是最終目標。後期要保留自然尾音。",
    ),
    Card(
        filename="03_current_result.png",
        eyebrow="目前結果與下一步",
        title="速度已經有候選，記憶體與聽感還要手機驗證",
        intro="現在的部署候選是 ZipVoice-Distill ONNX int8。模型比 Qwen 小很多，但仍不是 tiny model。",
        hero="部署包約 177MB；3-step 約 0.97s/句",
        sections=[
            ("目前可說完成", "teacher data 已進入學生訓練，產出 checkpoint，並匯出 ONNX int8。"),
            ("目前還不能宣稱", "還沒在真手機 native runtime 量 peak memory、冷啟動與連續生成。"),
            ("下一步", "在 3-step / 4-step / 6-step 之間做聽感選擇，再做 iOS/Android 實測。"),
        ],
        metrics=[
            Metric("Qwen teacher", "約 2.2GB", "peak RSS 約 2.2-2.33GB"),
            Metric("ZipVoice int8", "約 177MB", "text encoder + decoder + vocoder"),
            Metric("3-step", "RTF 0.1505", "live preload 約 0.97s/句"),
            Metric("4-step", "RTF 0.1990", "live preload 約 1.25s/句"),
        ],
        footer="若 177MB 可接受，先做 app；若不接受，要壓 vocoder/decoder，不是只調 steps。",
    ),
]


def draw_card(card: Card) -> Path:
    img = Image.new("RGB", (W, H), "#f6f7f4")
    draw = ImageDraw.Draw(img)
    margin = s(24)
    width = W - margin * 2
    y = s(34)

    draw.text((margin, y), card.eyebrow, font=font(15, True), fill="#a32235")
    y += s(34)
    y = draw_wrapped(draw, (margin, y), card.title, font(31, True), "#181a1b", width, s(9))
    y += s(8)
    y = draw_wrapped(draw, (margin, y), card.intro, font(17), "#626b70", width, s(8))
    y += s(22)

    hero_h = s(188)
    rounded(draw, (margin, y, W - margin, y + hero_h), "#1d1a18")
    hy = y + s(23)
    draw.text((margin + s(18), hy), "核心判斷", font=font(14, True), fill="#d9cbc4")
    hy += s(38)
    draw_wrapped(draw, (margin + s(18), hy), card.hero, font(25, True), "#ffffff", width - s(36), s(8))
    y += hero_h + s(18)

    for title, body in card.sections:
        box_top = y
        body_font = font(16)
        title_font = font(18, True)
        body_lines = wrap_text(draw, body, body_font, width - s(32))
        box_h = s(58) + len(body_lines) * (body_font.size + s(7))
        rounded(draw, (margin, box_top, W - margin, box_top + box_h), "#ffffff", "#d9dfdc")
        draw.text((margin + s(16), box_top + s(13)), title, font=title_font, fill="#17684f")
        draw_wrapped(
            draw,
            (margin + s(16), box_top + s(47)),
            body,
            body_font,
            "#626b70",
            width - s(32),
            s(7),
        )
        y = box_top + box_h + s(11)

    y += s(6)
    draw.text((margin, y), "關鍵數字", font=font(20, True), fill="#181a1b")
    y += s(42)

    metric_gap = s(10)
    metric_w = (width - metric_gap) // 2
    metric_h = s(142)
    for index, metric in enumerate(card.metrics):
        col = index % 2
        row = index // 2
        x = margin + col * (metric_w + metric_gap)
        yy = y + row * (metric_h + metric_gap)
        rounded(draw, (x, yy, x + metric_w, yy + metric_h), "#fff0f2", "#ecc7ce")
        draw.text((x + s(13), yy + s(13)), metric.label, font=font(12, True), fill="#626b70")
        draw.text((x + s(13), yy + s(43)), metric.value, font=font(24, True), fill="#a32235")
        draw_wrapped(draw, (x + s(13), yy + s(84)), metric.note, font(13), "#626b70", metric_w - s(26), s(5))
    y += ((len(card.metrics) + 1) // 2) * (metric_h + metric_gap) + s(10)

    y += s(12)
    footer_h = s(170)
    rounded(draw, (margin, y, W - margin, y + footer_h), "#edf5fb", "#c3d9e8")
    draw.text((margin + s(16), y + s(14)), "結論", font=font(16, True), fill="#265f86")
    draw_wrapped(
        draw,
        (margin + s(16), y + s(50)),
        card.footer,
        font(16),
        "#33424b",
        width - s(32),
        s(7),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / card.filename
    img.save(path, quality=95)
    return path


def build_index(paths: list[Path]) -> None:
    cards = "\n".join(
        f'<article><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></article>'
        for path in paths
    )
    (OUT_DIR / "index.html").write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TTS 蒸餾手機圖卡</title>
  <style>
    body {{ margin:0; background:#0f1112; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; }}
    main {{ width:min(100%,430px); margin:0 auto; padding:12px; }}
    h1 {{ color:white; font-size:22px; line-height:1.2; margin:8px 0 12px; letter-spacing:0; }}
    p {{ color:#b8c0c5; font-size:15px; line-height:1.55; margin:0 0 14px; }}
    article {{ margin:0 0 18px; }}
    img {{ width:100%; display:block; border-radius:8px; }}
  </style>
</head>
<body>
  <main>
    <h1>TTS 蒸餾做法與原理圖卡</h1>
    <p>三張手機直式圖：原理、流程、目前結果。可直接存圖或丟到聊天裡看。</p>
    {cards}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    paths = [draw_card(card) for card in CARDS]
    build_index(paths)
    for path in paths:
        print(path)
    print(OUT_DIR / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

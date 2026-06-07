#!/usr/bin/env python3
"""Create distillation texts for the taiwan_mandarin_low_r teacher voice."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "distill_texts_v1.jsonl"


OPENINGS = [
    "欸，你先不要急啦",
    "等一下，我想先确认一件事",
    "如果是我的话",
    "我知道你现在很紧张",
    "这边先不要下结论",
    "嗯，我懂你的意思",
    "你可以慢慢说",
    "我觉得这件事有点奇怪",
    "先停一下，好不好",
    "这句话听起来很重要",
]

MIDDLES = [
    "我们把时间和人物关系整理清楚",
    "证据还不够完整，所以不能乱猜",
    "刚刚那个细节可能不是巧合",
    "对方的说法前后有一点不一样",
    "你不用马上证明自己没有错",
    "真正的问题可能藏在旁边",
    "我会陪你一起看，但我们不要冲动",
    "尾音自然一点会比较像平常聊天",
    "卷舌不要太明显，讲话清楚就好",
    "声音不要太甜，也不要变成主播腔",
    "先把资料放在同一个地方比较安全",
    "如果消息传出去，事情可能会更难处理",
]

ENDINGS = [
    "这样我们才知道下一步要怎么做。",
    "你先冷静一点，我有在听。",
    "等我们确认以后，再决定要不要继续查。",
    "不要担心，我不是在怪你。",
    "这件事还有机会处理，不是完全没办法。",
    "我只是希望你不要一个人硬撑。",
    "你刚刚那句话，可以再讲清楚一点吗？",
    "听起来要真实一点，像台湾女生平常说话。",
    "如果这句自然，手机模型应该会比较稳。",
    "我们慢慢来，不用急着把答案说死。",
]

CATEGORIES = [
    "calm",
    "reasoning",
    "soft_care",
    "question",
    "voice_check",
    "gentle_warning",
    "analysis",
    "conversation",
]


SPECIAL_LINES = [
    ("warmup", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("voice_check", "这样听起来有比较像台湾人吗？我希望尾音自然一点，不要太卷舌。"),
    ("voice_check", "我想要的不是主播腔，也不是娃娃音，是聪明、温柔、真实的声音。"),
    ("voice_check", "卷舌不用太明显，讲话清楚就好，听起来要像台湾人平常聊天。"),
    ("soft_care", "没关系，你慢慢说，我有在听。不要急，想到哪里就先讲哪里。"),
    ("reasoning", "如果是我的话，我会先把时间、地点、人物关系全部列出来。"),
    ("question", "欸，先停一下。你有没有发现，刚刚那个人的说法前后不太一样？"),
    ("firm", "这不是玩笑欸。你如果继续装没事，事情只会越来越麻烦。"),
]


def main() -> int:
    rows = []
    seen = set()
    for category, text in SPECIAL_LINES:
        rows.append({"id": f"distill_{len(rows) + 1:04d}", "text": text, "category": category})
        seen.add(text)

    i = 0
    while len(rows) < 500:
        opening = OPENINGS[i % len(OPENINGS)]
        middle = MIDDLES[(i // len(OPENINGS)) % len(MIDDLES)]
        ending = ENDINGS[(i // (len(OPENINGS) * len(MIDDLES))) % len(ENDINGS)]
        text = f"{opening}，{middle}，{ending}"
        if text not in seen:
            rows.append(
                {
                    "id": f"distill_{len(rows) + 1:04d}",
                    "text": text,
                    "category": CATEGORIES[i % len(CATEGORIES)],
                }
            )
            seen.add(text)
        i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(OUT)
    print(f"items={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

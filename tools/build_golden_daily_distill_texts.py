#!/usr/bin/env python3
"""Create a larger daily-dialogue text set for the selected golden teacher."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "distill_texts_golden_daily_v1.jsonl"
)

OPENINGS = [
    "欸，我刚到楼下",
    "你先不用急",
    "我刚刚在开会",
    "等一下我们再确认一次",
    "如果你愿意的话",
    "今天先简单一点就好",
    "我觉得这样也可以",
    "你慢慢说",
    "我等一下顺路过去",
    "这件事我们不要想太复杂",
    "我刚刚有看到讯息",
    "你先喝一点水",
    "我不是在催你",
    "没关系啦",
    "我想了一下",
    "你这样讲我懂",
    "我先把东西放好",
    "今天外面有点热",
    "你等我一下",
    "我们晚一点再说",
]

MIDDLES = [
    "你慢慢来就好",
    "我在这边等你",
    "不用马上回我",
    "先把心情放松一点",
    "我会陪你一起看",
    "不要一个人硬撑",
    "晚餐我们吃简单一点",
    "饮料我帮你买半糖少冰",
    "这句话可以再讲清楚一点",
    "我们先确认最重要的那件事",
    "你刚刚说的我有听到",
    "声音放轻一点会比较自然",
    "不要讲得太像主播",
    "尾音软一点就很好听",
    "卷舌不用太明显",
    "讲话清楚但不要太用力",
    "像平常聊天那样就好",
    "我觉得你已经做得很好了",
    "先不要急着证明自己",
    "我们一步一步来",
    "我想听你真实的想法",
    "不要为了讨好别人委屈自己",
    "你如果累了就先休息",
    "这件事没有你想的那么糟",
]

ENDINGS = [
    "等一下再一起确认一次。",
    "我在，不用担心。",
    "这样比较像我们平常说话。",
    "听起来会比较真实一点。",
    "我觉得这样就满好了。",
    "不要急，我们慢慢来。",
    "你可以先跟我说没关系。",
    "我不会觉得你很麻烦。",
    "如果不舒服，我们就先停一下。",
    "你想清楚以后再决定也可以。",
    "我只是希望你轻松一点。",
    "这样手机上应该会比较稳。",
    "我觉得这个声音会比较耐听。",
    "不用太甜，温柔一点就好。",
    "我们先做到自然，再来追求更像。",
]

SPECIAL_LINES = [
    ("golden_anchor", "如果你愿意的话，我们等一下再一起确认一次。"),
    ("daily", "欸我刚到楼下，你不用急，慢慢来就好，我在这边等你。"),
    ("daily", "你要喝什么？我等一下顺路买，珍奶半糖少冰可以吗？"),
    ("daily", "今天有点累欸，我们晚餐简单吃就好，不要跑太远。"),
    ("daily", "我刚刚在开会没有看到讯息，不是故意不回你啦。"),
    ("soft", "没关系，你慢慢说，我有在听，想到哪里就先讲哪里。"),
    ("soft", "我不是要你马上回答，只是想知道你心里真正的感觉。"),
    ("voice", "卷舌不用太明显，尾音自然一点，像台湾女生平常聊天就好。"),
    ("voice", "我想要的不是主播腔，也不是娃娃音，是温柔、聪明、真实的声音。"),
    ("voice", "这句如果听起来舒服，后面做成手机模型才有意义。"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    rows = []
    seen = set()
    for category, text in SPECIAL_LINES:
        rows.append({"id": f"golden_daily_{len(rows) + 1:04d}", "text": text, "category": category})
        seen.add(text)

    candidates = [
        f"{opening}，{middle}，{ending}"
        for opening in OPENINGS
        for middle in MIDDLES
        for ending in ENDINGS
    ]
    random.Random(42).shuffle(candidates)
    for text in candidates:
        if text not in seen:
            category = "voice" if "声音" in text or "卷舌" in text or "尾音" in text else "daily"
            rows.append(
                {
                    "id": f"golden_daily_{len(rows) + 1:04d}",
                    "text": text,
                    "category": category,
                }
            )
            seen.add(text)
        if len(rows) >= args.count:
            break

    if len(rows) < args.count:
        raise RuntimeError(f"Only generated {len(rows)} unique rows; requested {args.count}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(args.out)
    print(f"items={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

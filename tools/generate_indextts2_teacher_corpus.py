#!/usr/bin/env python3
"""Generate a small IndexTTS2 teacher corpus for mobile-student experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = ROOT / "external" / "index-tts"
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
DEFAULT_OUT = BASE / "teacher_indextts2_distill_smoke_v1"
DEFAULT_REF_AUDIO = (
    BASE
    / "datasets"
    / "downloads_female_voice"
    / "reference_packs_v1"
    / "pack_best2_7s.wav"
)
DEFAULT_REF_TEXT = "所以我当时就说,我想要做一张疗愈人的专辑。 开始当然就是我们的提案会议,我就提出了因为多年"

TEXTS = [
    "你先不要急，我们慢慢来，把事情一件一件处理好。",
    "我刚刚看了一下，应该不是你的问题，你不用太担心。",
    "等一下如果方便的话，我们可以一起确认那个设定。",
    "今天有点累，但听到你这样说，我心情有比较放松。",
    "这件事我觉得可以再想一下，不一定要马上决定。",
    "没关系啦，你先讲，我在这边听，真的不用紧张。",
    "我帮你留意一下，如果有变化，我再马上跟你说。",
    "其实这样也不错，简单一点，反而比较舒服。",
    "你刚刚那句话我懂，感觉就是有点卡住，对不对。",
    "我们先试一个小版本，确定可以之后再慢慢加功能。",
    "如果声音太硬，就把语气放轻一点，尾音自然收掉。",
    "我希望它听起来像真的人在讲话，不要太像机器。",
    "这句你再听一次，应该会比较接近我们想要的感觉。",
    "我现在有一点想法，不过我们可以先不要急着下结论。",
    "你晚一点到也没关系，我会先把东西整理好。",
    "这个版本我觉得方向对了，只是还需要再更自然一点。",
    "我们今天先做到这里就好，剩下的明天再继续。",
    "我有听懂你的意思，只是想确认一下细节有没有错。",
    "这样讲好像比较台湾一点，卷舌不要太重，语尾也轻一点。",
    "不用证明什么啦，先把能听的版本做出来比较重要。",
    "如果手机跑得动，那这个方向就真的很有机会。",
    "我想要的是温柔清亮，但不要太夸张，也不要太撒娇。",
    "你先喝口水，我们等一下再继续测试下一句。",
    "好，那我们就照这个方向做，慢慢把品质磨起来。",
]


def build_extra_texts(target_count: int) -> list[str]:
    """Build deterministic daily-dialogue lines for a larger teacher corpus."""
    openers = [
        "你先不要急",
        "我刚刚想了一下",
        "我们慢慢确认",
        "其实这个方向可以",
        "如果你觉得不舒服",
        "我会先帮你看",
        "你不用太担心",
        "这件事可以简单一点",
        "等一下我们再试一次",
        "我觉得声音可以再轻一点",
    ]
    middles = [
        "把最重要的部分先处理好",
        "不要一下子改太多东西",
        "先听听看自然不自然",
        "让语气听起来温柔一点",
        "把尾音收得自然一点",
        "确认它真的有讲对文字",
        "先不要急着证明效果",
        "让反应速度维持快一点",
        "把卷舌压低一点",
        "让整句话更像台湾人在聊天",
    ]
    endings = [
        "这样应该会比较稳。",
        "我觉得会比较接近我们想要的感觉。",
        "听起来也会比较舒服。",
        "我们再慢慢把品质磨起来。",
        "你等一下听听看就知道。",
        "如果不对我们再重来没关系。",
        "先做出能用的版本比较重要。",
        "这样放到手机上才有意义。",
        "不要太夸张，真实一点就好。",
        "声音要清亮，但不要太用力。",
    ]
    followups = [
        "欸，这句我想要轻轻地讲，不要太像播报。",
        "没关系啦，我们今天先抓方向，明天再修细节。",
        "你听一下这个版本，如果怪怪的我马上改。",
        "我希望它像朋友在旁边讲话，不是机器人念稿。",
        "如果手机可以即时跑，那这个东西就真的好玩。",
        "我们先把中文讲清楚，再追求更像原本的声音。",
        "这个句子短一点，比较适合测试反应速度。",
        "这句稍微长一点，可以测试它会不会中间断掉。",
        "语尾不要太硬，轻轻放下来就会比较台湾。",
        "我觉得不用太甜，温柔聪明一点就很好。",
    ]
    connectors = [
        "然后",
        "所以",
        "不过",
        "等一下",
        "如果可以的话",
        "我觉得",
        "你听听看",
        "我们先这样",
    ]
    texts = list(TEXTS)
    candidates: list[str] = []
    candidates.extend(followups)
    for opener in openers:
        for middle in middles:
            for ending in endings:
                candidates.append(f"{opener}，{middle}，{ending}")
    for opener in openers:
        for connector in connectors:
            for ending in endings:
                candidates.append(f"{opener}，{connector}，{ending}")
    for middle in middles:
        for connector in connectors:
            for ending in endings:
                candidates.append(f"{middle}，{connector}，{ending}")

    for text in candidates:
        if text not in texts:
            texts.append(text)
        if len(texts) >= target_count:
            break
    if len(texts) < target_count:
        raise ValueError(f"Only generated {len(texts)} unique texts; requested {target_count}")
    return texts[:target_count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--texts-file", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    ref_audio = args.ref_audio if args.ref_audio.is_absolute() else ROOT / args.ref_audio
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not ref_audio.exists():
        raise FileNotFoundError(ref_audio)

    os.chdir(INDEX_ROOT)
    sys.path.insert(0, str(INDEX_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        device=args.device,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    rows = []
    manifest = out / "manifest.json"
    partial_manifest = out / "manifest.partial.json"
    if args.texts_file:
        texts_file = args.texts_file if args.texts_file.is_absolute() else ROOT / args.texts_file
        texts = [line.strip() for line in texts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        texts = build_extra_texts(args.limit)

    for idx, text in enumerate(texts[: args.limit], start=1):
        item_id = f"indextts2_tw_{idx:04d}"
        output = audio_dir / f"{item_id}.wav"
        status = "ok"
        error = ""
        started = time.perf_counter()
        if args.overwrite or not output.exists():
            try:
                tts.infer(
                    spk_audio_prompt=str(ref_audio),
                    text=text,
                    output_path=str(output),
                    emo_vector=[0, 0, 0, 0, 0, 0, 0, 0.8],
                    use_random=False,
                    verbose=False,
                )
            except Exception as exc:  # pragma: no cover - experiment script
                status = "failed"
                error = str(exc)
        seconds = time.perf_counter() - started
        row = {
            "id": item_id,
            "text": text,
            "audio": str(output),
            "status": status if output.exists() else "failed",
            "seconds": seconds,
            "error": error,
            "teacher": "IndexTTS2",
            "reference_audio": str(ref_audio),
            "reference_text": args.ref_text,
        }
        rows.append(row)
        manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        partial_manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{item_id}: {row['status']} {seconds:.2f}s")

    (out / "README.md").write_text(
        "# IndexTTS2 Teacher Corpus Smoke v1\n\n"
        "Fixed-text teacher corpus for Piper/VITS and Matcha/FastSpeech-style mobile-student experiments.\n\n"
        f"- Items requested: {min(args.limit, len(texts))}\n"
        f"- Reference audio: `{ref_audio}`\n"
        f"- Manifest: `{manifest}`\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

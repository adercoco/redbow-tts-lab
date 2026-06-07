#!/usr/bin/env python3
"""Download the minimum GPT-SoVITS official assets needed for v2 inference."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "GPT-SoVITS"
PRETRAINED = REPO / "GPT_SoVITS" / "pretrained_models"
TEXT = REPO / "GPT_SoVITS" / "text"


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        target = dst / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != item.stat().st_size:
                target.write_bytes(item.read_bytes())


def main() -> int:
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    TEXT.mkdir(parents=True, exist_ok=True)

    snapshot_root = Path(
        snapshot_download(
            repo_id="lj1995/GPT-SoVITS",
            allow_patterns=[
                "gsv-v2final-pretrained/*",
                "chinese-hubert-base/*",
                "chinese-roberta-wwm-ext-large/*",
            ],
        )
    )

    for name in [
        "gsv-v2final-pretrained",
        "chinese-hubert-base",
        "chinese-roberta-wwm-ext-large",
    ]:
        src = snapshot_root / name
        if not src.exists():
            raise FileNotFoundError(f"missing downloaded asset: {src}")
        copy_tree(src, PRETRAINED / name)

    g2pw_zip = Path(
        hf_hub_download(
            repo_id="XXXXRT/GPT-SoVITS-Pretrained",
            filename="G2PWModel.zip",
        )
    )
    g2pw_dst = TEXT / "G2PWModel"
    if not g2pw_dst.exists():
        with ZipFile(g2pw_zip) as archive:
            archive.extractall(TEXT)
    if not g2pw_dst.exists():
        candidates = [p for p in TEXT.iterdir() if p.is_dir() and p.name.lower().startswith("g2pw")]
        if candidates:
            candidates[0].rename(g2pw_dst)
    if not g2pw_dst.exists():
        raise FileNotFoundError("G2PWModel did not extract to GPT_SoVITS/text/G2PWModel")

    required = [
        PRETRAINED / "gsv-v2final-pretrained" / "s2G2333k.pth",
        PRETRAINED / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        PRETRAINED / "chinese-hubert-base" / "config.json",
        PRETRAINED / "chinese-roberta-wwm-ext-large" / "config.json",
        g2pw_dst,
    ]
    for path in required:
        print(path)
        if not path.exists():
            raise FileNotFoundError(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

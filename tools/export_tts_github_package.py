#!/usr/bin/env python3
"""Create a GitHub-friendly export of the Red Bow/TTS lab.

The working App directory contains virtualenvs, external model repos, build
products, and multi-GB checkpoints. This export keeps the code, reports,
reference audio, app project, and current web server state, while replacing
single files over GitHub's regular 100MB limit with split parts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/ader/Documents/App")
EXPORT = Path("/Users/ader/Documents/redbow-tts-github-export")
MAX_BLOB = 90 * 1024 * 1024

INCLUDE_PATHS = [
    ".gitignore",
    "README.md",
    "NOTES.md",
    "YOUTUBE_REFERENCES.md",
    "ZIPVOICE_MAINLINE.md",
    "ZIPVOICE_APP_NOISE_DIAGNOSIS.md",
    "LOVELENS_QWEN_DISTILLATION_REPORT.md",
    "RedBowVoice",
    "RedBowVoice.xcodeproj",
    "RedBowVoiceUITests",
    "assets",
    "voice_profiles",
    "youtube_refs",
    "reference_audio",
    "character_reference_report",
    "haibara_tts_comparison",
    "tts_mobile_reports",
    "tools",
    "distillation/conan_authorized_voice_refs_v1",
    "distillation/character_voice_collection_refs_v1",
    "distillation/taiwan_mandarin_low_r/reports",
    "distillation/taiwan_mandarin_low_r/golden_teacher",
    "distillation/taiwan_mandarin_low_r/mobile_package",
    "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1",
]

IGNORE_NAMES = {
    "__pycache__",
    ".DS_Store",
    "DerivedData",
    "build",
}


def ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in IGNORE_NAMES:
            ignored.add(name)
        if name.endswith(".pyc") or name.endswith(".pyo"):
            ignored.add(name)
    return ignored


def run(cmd: list[str], cwd: Path = EXPORT) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_path(relative: str) -> None:
    src = ROOT / relative
    dst = EXPORT / relative
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore, symlinks=True)
    else:
        shutil.copy2(src, dst)


def split_large_files() -> list[dict]:
    manifest = []
    for path in sorted(EXPORT.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        size = path.stat().st_size
        if size <= MAX_BLOB:
            continue
        rel = path.relative_to(EXPORT)
        parts_dir = EXPORT / ".large-files" / str(rel)
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_paths = []
        with path.open("rb") as source:
            index = 0
            while True:
                chunk = source.read(MAX_BLOB)
                if not chunk:
                    break
                part_path = parts_dir / f"part-{index:03d}"
                part_path.write_bytes(chunk)
                part_paths.append(str(part_path.relative_to(EXPORT)))
                index += 1
        path.unlink()
        placeholder = path.with_suffix(path.suffix + ".SPLIT.txt")
        placeholder.write_text(
            "This file exceeded GitHub's regular 100MB blob limit and was split.\n"
            "Run scripts/reassemble_large_files.py from the repository root to restore it.\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "path": str(rel),
                "size_bytes": size,
                "parts": part_paths,
                "placeholder": str(placeholder.relative_to(EXPORT)),
            }
        )
    return manifest


def write_support_files(large_manifest: list[dict]) -> None:
    (EXPORT / "scripts").mkdir(parents=True, exist_ok=True)
    (EXPORT / "docs").mkdir(parents=True, exist_ok=True)
    (EXPORT / "LARGE_FILES_MANIFEST.json").write_text(
        json.dumps(large_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (EXPORT / "scripts" / "reassemble_large_files.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "LARGE_FILES_MANIFEST.json").read_text())
for item in manifest:
    target = ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        for part in item["parts"]:
            out.write((ROOT / part).read_bytes())
    print(f"restored {target.relative_to(ROOT)}")
""",
        encoding="utf-8",
    )
    os.chmod(EXPORT / "scripts" / "reassemble_large_files.py", 0o755)

    heavy = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size >= 500 * 1024 * 1024:
            heavy.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_gb": round(size / 1024**3, 3),
                }
            )
    (EXPORT / "docs" / "heavy_artifacts_over_500mb.json").write_text(
        json.dumps(heavy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = f"""# Red Bow TTS Lab Export

Private GitHub export generated from `/Users/ader/Documents/App` on {datetime.now().isoformat(timespec="seconds")}.

This repository contains the current Red Bow voice changer app, local Cosy web
server, TTS reports, authorized reference packs, and the scripts used to prepare
and audition voices.

## What is included

- `tools/serve_haibara_cosy_web.py`: current Mac-hosted CosyVoice2 web app.
- `distillation/character_voice_collection_refs_v1`: authorized Ryotsu, Shinchan, and Misae reference packs.
- `distillation/conan_authorized_voice_refs_v1`: authorized Conan/Haibara/Agasa reference packs and reports.
- `distillation/taiwan_mandarin_low_r/reports`: historical TTS, clone, distillation, Matcha, Piper, ZipVoice reports.
- `RedBowVoice`: iOS red bow app project.
- `tts_mobile_reports`, `assets`, `reference_audio`, `youtube_refs`: supporting reports/assets.

## What is intentionally not included

The original workspace has local virtualenvs, external cloned model repos, build
outputs, and multi-GB checkpoints. Those are not suitable for normal GitHub git
storage. A list of the largest local artifacts is in:

```bash
docs/heavy_artifacts_over_500mb.json
```

## Restore split files

GitHub regular git rejects individual blobs over 100MB. Any such file in this
export was split into `.large-files/...` parts. After cloning, run:

```bash
python3 scripts/reassemble_large_files.py
```

Current split file count: {len(large_manifest)}.

## Run the web demo on a Mac

This repo does not vendor the Python virtualenv or the full CosyVoice model
weights. On a machine where the dependencies/model are installed at the same
relative paths, run:

```bash
PORT=8782 ./.venv-cosyvoice/bin/python tools/serve_haibara_cosy_web.py
```

If setting up a fresh machine, install/clone CosyVoice under `external/CosyVoice`
and place `CosyVoice2-0.5B` under `external/CosyVoice/pretrained_models/`.
"""
    (EXPORT / "README.md").write_text(readme, encoding="utf-8")

    index = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Red Bow TTS Lab Export</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans TC',sans-serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.6;background:#f7f4ee;color:#2d2a26}a{color:#9f3028}li{margin:8px 0}</style></head>
<body><h1>Red Bow TTS Lab Export</h1>
<p>這是 GitHub 匯出版入口。主要可讀報告：</p>
<ul>
<li><a href="../distillation/conan_authorized_voice_refs_v1/reports/clone_data_inventory_v1/clone_data_inventory_v1.html">Clone data inventory</a></li>
<li><a href="../distillation/conan_authorized_voice_refs_v1/reports/haibara_cosy_speaker_finetune_v1/haibara_cosy_speaker_finetune_v1_standalone.html">Haibara Cosy speaker fine-tune status</a></li>
<li><a href="../distillation/conan_authorized_voice_refs_v1/reports/conan_authorized_cosy_clone_audition_v1/conan_authorized_cosy_clone_audition_v1_standalone.html">Conan authorized Cosy clone audition</a></li>
<li><a href="../distillation/taiwan_mandarin_low_r/reports/report_index_v1/index.html">Taiwan Mandarin report index</a></li>
</ul></body></html>
"""
    (EXPORT / "docs" / "index.html").write_text(index, encoding="utf-8")

    gitignore = """.DS_Store
__pycache__/
*.pyc
.venv*/
external/
build/
DerivedData/
RunArtifacts/
tmp/
"""
    (EXPORT / ".gitignore").write_text(gitignore, encoding="utf-8")


def main() -> None:
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True)
    for relative in INCLUDE_PATHS:
        copy_path(relative)
    large_manifest = split_large_files()
    write_support_files(large_manifest)

    run(["git", "init", "-b", "main"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "Initial Red Bow TTS lab export"])
    print(EXPORT)


if __name__ == "__main__":
    main()

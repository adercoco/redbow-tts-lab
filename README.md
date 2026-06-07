# Red Bow TTS Lab Export

Private GitHub export generated from `/Users/ader/Documents/App` on 2026-06-07T22:56:28.

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

Current split file count: 1.

## Run the web demo on a Mac

This repo does not vendor the Python virtualenv or the full CosyVoice model
weights. On a machine where the dependencies/model are installed at the same
relative paths, run:

```bash
PORT=8782 ./.venv-cosyvoice/bin/python tools/serve_haibara_cosy_web.py
```

If setting up a fresh machine, install/clone CosyVoice under `external/CosyVoice`
and place `CosyVoice2-0.5B` under `external/CosyVoice/pretrained_models/`.

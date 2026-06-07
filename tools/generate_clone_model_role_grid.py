#!/usr/bin/env python3
"""Generate more Taiwan-Mandarin role samples for F5, CosyVoice2, and GPT-SoVITS."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torchaudio


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v2"
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v3"
F5_CLI = ROOT / ".venv-f5" / "bin" / "f5-tts_infer-cli"
COSY_ROOT = ROOT / "external" / "CosyVoice"
GPT_RUNNER = ROOT / "tools" / "run_gpt_sovits_role_grid.py"

REF_TEXT = "我知道你现在有点紧张，慢慢说就好，我会听完。"


@dataclass(frozen=True)
class Role:
    id: str
    label: str
    note: str

    @property
    def ref_audio(self) -> Path:
        return SOURCE / "audio" / "qwen3_voicedesign" / self.id / "soft.wav"


ROLES = [
    Role("low_r_clean", "低卷舌清亮台灣女生", "低卷舌、清亮、克制、像聰明台大女生"),
    Role("ntu_literature_warm", "台大文學院書卷氣", "溫柔、有書卷氣、不要娃娃音"),
    Role("coffee_shop_friend", "咖啡店朋友語音", "像朋友傳語音，親近、自然、尾音軟"),
    Role("bookstore_senpai", "書店學姊溫柔提醒", "小聲但清楚，親和、沒有主播腔"),
    Role("campus_radio_sister", "校園廣播學姊", "明亮、有一點笑意、像真的大學生"),
    Role("clinic_nurse_soft", "診所護理師親切聲", "讓人安心、清楚、親切但專業"),
]

TEXTS = [
    ("tw_wait", "欸，你先不要急着下结论啦，我们把时间线顺一下就好。"),
    ("tw_detail", "我刚刚有看到那个细节喔，感觉不是巧合，我们再听一次。"),
    ("tw_slow", "没关系，你慢慢讲，我在这边听，真的不用急。"),
    ("tw_weird", "等一下，这个地方有点怪耶，可是我还不想太早下判断。"),
]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def row_base(family: str, role: Role, text_id: str, text: str, output: Path) -> dict[str, object]:
    return {
        "family": family,
        "candidate_id": f"{family.lower().replace(' ', '_').replace('-', '_')}_{role.id}",
        "role_id": role.id,
        "label": role.label,
        "prompt": f"{role.note}；ref_audio=Qwen3/{role.id}/soft.wav；ref_text={REF_TEXT}",
        "text_id": text_id,
        "text": text,
        "output": str(output),
    }


def run_f5() -> list[dict[str, object]]:
    require(F5_CLI)
    rows = []
    for role in ROLES:
        require(role.ref_audio)
        audio_dir = OUT / "audio" / "f5_tts" / role.id
        audio_dir.mkdir(parents=True, exist_ok=True)
        for text_id, text in TEXTS:
            final = audio_dir / f"{text_id}.wav"
            work = audio_dir / f"_work_{text_id}"
            work.mkdir(parents=True, exist_ok=True)
            started = perf_counter()
            status = "ok"
            error = ""
            if not final.exists():
                try:
                    subprocess.run(
                        [
                            str(F5_CLI),
                            "--model",
                            "F5TTS_v1_Base",
                            "--ref_audio",
                            str(role.ref_audio),
                            "--ref_text",
                            REF_TEXT,
                            "--gen_text",
                            text,
                            "--output_dir",
                            str(work),
                            "--output_file",
                            final.name,
                            "--remove_silence",
                            "--nfe_step",
                            "32",
                        ],
                        cwd=work,
                        check=True,
                    )
                    generated = work / final.name
                    if not generated.exists():
                        raise RuntimeError(f"F5 generated no wav at {generated}")
                    shutil.copy2(generated, final)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            row = row_base("F5-TTS", role, text_id, text, final)
            row.update({"seconds": perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"F5 {role.id} {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_cosyvoice() -> list[dict[str, object]]:
    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    rows = []
    for role in ROLES:
        require(role.ref_audio)
        audio_dir = OUT / "audio" / "cosyvoice2" / role.id
        audio_dir.mkdir(parents=True, exist_ok=True)
        for text_id, text in TEXTS:
            final = audio_dir / f"{text_id}.wav"
            started = perf_counter()
            status = "ok"
            error = ""
            if not final.exists():
                try:
                    for index, item in enumerate(model.inference_zero_shot(text, REF_TEXT, str(role.ref_audio), stream=False)):
                        if index == 0:
                            torchaudio.save(str(final), item["tts_speech"], model.sample_rate)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            row = row_base("CosyVoice2", role, text_id, text, final)
            row.update({"seconds": perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"Cosy {role.id} {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_gpt_sovits() -> list[dict[str, object]]:
    plan = []
    for role in ROLES:
        require(role.ref_audio)
        for text_id, text in TEXTS:
            output = OUT / "audio" / "gpt_sovits_v2" / role.id / f"{text_id}.wav"
            plan.append(
                {
                    **row_base("GPT-SoVITS v2", role, text_id, text, output),
                    "ref_audio": str(role.ref_audio),
                    "ref_text": REF_TEXT,
                }
            )
    plan_path = OUT / "gpt_sovits_role_grid_plan.json"
    results_path = OUT / "gpt_sovits_role_grid_results.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [
            str(ROOT / ".venv-gptsovits" / "bin" / "python"),
            str(GPT_RUNNER),
            "--plan",
            str(plan_path),
            "--results",
            str(results_path),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads(results_path.read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(run_cosyvoice())
    rows.extend(run_gpt_sovits())
    rows.extend(run_f5())
    results = OUT / "clone_model_role_grid_results.json"
    results.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

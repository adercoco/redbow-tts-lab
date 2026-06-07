"""Shared definitions for no-reference Taiwan female voice-design auditions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceDesign:
    id: str
    label: str
    prompt: str


VOICE_DESIGNS = [
    VoiceDesign(
        "taiwan_soft_low_r",
        "台灣國語低卷舌",
        "台湾年轻女生，中文普通话带自然台湾国语口音，卷舌很轻，儿化音几乎没有。声音清亮温柔、漂亮耐听，有书卷气，像聪明的台大女生，克制、可爱、真实。",
    ),
    VoiceDesign(
        "taipei_daily_pretty",
        "台北日常漂亮女生",
        "二十岁出头的台北女生，声音漂亮清亮但不做作，讲话像朋友传语音。普通话有台湾腔，低卷舌，尾音柔软自然，带一点笑意，不要主播腔。",
    ),
    VoiceDesign(
        "ntu_smart_gentle",
        "台大聰明溫柔",
        "像台大文学院女大学生，声音柔和清亮，有书卷气，讲话温柔聪明。台湾国语口音自然，卷舌很轻，语气克制但可爱，不要娃娃音。",
    ),
    VoiceDesign(
        "bookstore_senpai_no_ref",
        "書店學姊",
        "台湾书店打工的年轻女生，像温柔学姐，声音小声但清楚，漂亮有亲和力。普通话低卷舌，没有儿化音，语尾自然柔软。",
    ),
    VoiceDesign(
        "soft_app_assistant",
        "溫柔 App 女聲",
        "适合手机 app 的台湾年轻女生语音，声音温柔、清楚、轻快、耐听。低卷舌台湾国语，没有儿化音，像真人助理，不要广告配音感。",
    ),
    VoiceDesign(
        "calm_detective_girl",
        "溫柔推理女生",
        "聪明温柔的台湾女生，讲话冷静、细心、声音清亮漂亮，有一点推理感但不严肃。普通话低卷舌，台湾腔自然，语尾柔软。",
    ),
]


TESTS = [
    ("tw_wait", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_weird", "这个地方我觉得有点怪耶，可是先不要太早下结论。"),
    ("tw_slow", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("tw_tail", "我刚刚有注意到喔，那句话的尾音好像不太自然。"),
]

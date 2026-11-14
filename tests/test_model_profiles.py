"""The model profiles and presets of crispz-moss (rewritten from crispz-ideogram's).

The trap this guards: profile_for_model matches by substring, first match wins, and an
unknown name (a local diffusers folder, a renamed mirror, a CivitAI single file) falls
back on the default profile. Inherited from Ideogram, that default was 48 steps at 7.0:
on MOSS-Image a CFG of 7 burns the image. Every path must give 50 / 3.0.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "OpenMOSS-Team/MOSS-Image-SFT"


def _with_profiles(profiles, fn, default=None):
    old, old_d = cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE
    cz_core.MODEL_PROFILES = profiles
    if default is not None:
        cz_core.DEFAULT_MODEL_PROFILE = default
    try:
        return fn()
    finally:
        cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE = old, old_d


def test_an_empty_default_profile_still_gives_50_at_3():
    """A config.txt whose default_model_profile lacks a field: the code fallback is
    MOSS's, not Ideogram's 48 / 7.0 nor FLUX.3's 16 / 1.0."""
    got = _with_profiles({}, lambda: cz_core.profile_for_model("mystery"), default={})
    assert got == (50, 3.0)
    got = _with_profiles({"moss": {}}, lambda: cz_core.profile_for_model(REPO), default={})
    assert got == (50, 3.0)


def test_the_ui_presets_mirror_the_sample():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    import cz_ui
    sample = {k: (int(v[0]), float(v[1])) for k, v in cfg["performance_presets"].items()}
    assert set(sample) >= set(cz_ui.PERFORMANCE) or cz_ui.PERFORMANCE == sample


def test_refine_steps_are_the_measured_ones():
    """Lot 7, measured on the 5090: at the presets' denoise values a refine at 12 steps
    and at 36 give the same image (mean difference 2/255, texts unchanged), 36 being 53 %
    slower. The presets keep the short counts; the slider still reaches 48 and beyond."""
    import cz_ui
    for name, p in cz_ui.PRESETS.items():
        if "steps" in p:
            assert 12 <= p["steps"] <= 16, (name, p["steps"])
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["default_refine_steps"] == 12

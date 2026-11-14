"""crispz-minimax engine, CPU only, no weights: the key conversion of the pruned single
file, the PDD head bank, the curve time table, the size and strength rules, the refusals.

    python tests/test_minimax_engine.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import cz_minimax as M


def _tiny_sd(hidden=8, heads=6, patch=4, rank=3, text=5):
    """A state dict with the ComfyUI names of the pruned layout, tiny shapes."""
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g)
    sd = {"adaln_t_table": r(1025, rank), "rope.inv_freq": r(16),
          "video_patch_proj.weight": r(hidden, patch), "video_patch_proj.bias": r(hidden),
          "condition_proj.weight": r(hidden, text), "condition_proj.bias": r(hidden),
          "final_layer.norm.weight": r(hidden),
          "final_layer.adaln_proj.linear.weight": r(2 * hidden, rank),
          "final_layer.adaln_proj.linear.bias": r(2 * hidden),
          "final_layer.video_out.weight": r(heads * patch, hidden),
          "final_layer.video_out.bias": r(heads * patch)}
    for pre in ("blocks.0.", "token_refiner.blocks.0."):
        sd.update({pre + "attn.qkv_proj.weight": r(3 * 16, hidden), pre + "attn.out_proj.weight": r(hidden, 16),
                   pre + "attn.q_norm.weight": r(4), pre + "attn.k_norm.weight": r(4),
                   pre + "mlp.fc1.weight": r(2 * 12, hidden), pre + "mlp.fc2.weight": r(hidden, 12),
                   pre + "norm1.weight": r(hidden), pre + "norm2.weight": r(hidden)})
    sd["blocks.0.adaln_proj.linear.weight"] = r(6 * hidden * 3, rank)
    sd["blocks.0.adaln_proj.linear.bias"] = r(6 * hidden * 3)
    return sd


def test_key_conversion_names_and_fp32_islands():
    sd = _tiny_sd()
    out = M._convert_keys(dict(sd), torch.bfloat16)
    for k in ("proj_in.weight", "context_embedder.weight", "norm_out.norm.weight", "norm_out.linear.weight",
              "time_embedder.table", "transformer_blocks.0.attn.to_qkv.weight",
              "transformer_blocks.0.attn.to_out.0.weight", "transformer_blocks.0.attn.norm_q.weight",
              "transformer_blocks.0.ff.net.0.proj.weight", "transformer_blocks.0.ff.net.2.weight",
              "transformer_blocks.0.adaln_proj.linear.weight",
              "token_refiner.refiner_blocks.0.attn.to_qkv.weight"):
        assert k in out, k
    assert "rope.inv_freq" not in out and "adaln_t_table" not in out
    assert out["proj_in.weight"].dtype == torch.float32
    assert out["transformer_blocks.0.adaln_proj.linear.weight"].dtype == torch.float32
    assert out["time_embedder.table"].dtype == torch.float32
    assert out["context_embedder.weight"].dtype == torch.bfloat16
    # the fused projections are kept as stored (no re-cut of a quantized weight)
    assert torch.equal(out["transformer_blocks.0.ff.net.0.proj.weight"].float(),
                       sd["blocks.0.mlp.fc1.weight"].to(torch.bfloat16).float())


def test_head_bank_is_head0_plus_offsets():
    sd = _tiny_sd()
    out = M._convert_keys(dict(sd), torch.bfloat16)
    rows = sd["final_layer.video_out.weight"].reshape(M.PDD_STEPS, -1, 8)
    assert torch.equal(out["pdd_proj_out.0.weight"], rows[0])
    for k in range(1, M.PDD_STEPS):
        assert torch.allclose(out[f"pdd_proj_out.{k}.weight"], rows[0] + rows[k])
    b = sd["final_layer.video_out.bias"].reshape(M.PDD_STEPS, -1)
    assert torch.allclose(out["pdd_proj_out.3.bias"], b[0] + b[3])


def test_a_dense_file_is_refused_by_name():
    sd = _tiny_sd()
    sd.pop("adaln_t_table")
    try:
        M._convert_keys(sd, torch.bfloat16)
    except ValueError as e:
        assert "pruned" in str(e)
    else:
        raise AssertionError("a file without adaln_t_table went through")


def test_curve_table_interpolates_and_clamps():
    table = torch.arange(1025, dtype=torch.float32)[:, None].repeat(1, 2)
    t = M.CurveTimeTable(table)
    got = t(torch.tensor([0.0, 1.0, 0.5, 2.0, -1.0, 0.25 + 0.5 / 1024]))
    assert torch.allclose(got[:, 0], torch.tensor([0.0, 1024.0, 512.0, 1024.0, 0.0, 256.5]))


def test_gate_first_swiglu_reads_gate_then_value():
    m = M.GateFirstSwiGLU(2, 2)
    with torch.no_grad():
        m.proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]]))
    x = torch.tensor([[1.0, 1.0]])
    gate, value = torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0])
    assert torch.allclose(m(x), (value * torch.nn.functional.silu(gate))[None])


def test_sizes_snap_to_32_and_cap_at_20_mpx():
    assert M.fit_size(2048, 2048) == (2048, 2048)
    assert M.fit_size(1000, 1000) == (992, 992)
    w, h = M.fit_size(8000, 8000)
    assert w % 32 == 0 and h % 32 == 0 and w * h <= M.MAX_PIXELS
    assert M.fit_size(None, None) == (2048, 2048)
    assert M.fit_size(6272, 2688) == (6272, 2688)


def test_strength_snaps_to_the_fixed_grid():
    assert M.strength_to_start(None) == 0 and M.strength_to_start(1.0) == 0
    assert M.strength_to_start(0.9) == 2          # sigma 0.888
    assert M.strength_to_start(0.75) == 3
    assert M.strength_to_start(0.5) == 4          # sigma 0.533
    assert M.strength_to_start(0.3) == 5 and M.strength_to_start(0.0) == 5
    assert len(M.PDD_SIGMAS) == M.PDD_STEPS + 1 == 7 and M.PDD_SIGMAS[-1] == 0.0


def test_families_are_named():
    assert M.family_from_keys(["final_layer.video_out.weight", "video_patch_proj.weight",
                               "adaln_t_table"]) == "minimax"
    assert M.family_from_keys(["final_layer.video_out.weight", "video_patch_proj.weight",
                               "audio_patch_proj.weight"]) != "minimax"       # the H3 video model
    assert M.family_from_keys(["layers.0.proj.ffn_up_gate.weight"]) == "moss"
    assert M.family_from_keys(["llm_cond_proj.weight"]) == "ideogram"
    assert M.family_from_keys([]) is None


def test_refusals_say_what_to_take():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.safetensors")
        with open(p, "wb") as f:
            f.write(b"\0" * 16)
        why = M.single_file_refusal(p)
        assert why and "GGUF" in why and M.DEFAULT_TRANSFORMER[0] in why
        assert M.is_text_encoder_file(p) is False
        g = os.path.join(d, "broken.gguf")
        with open(g, "wb") as f:
            f.write(b"not a gguf")
        assert "unreadable" in M.single_file_refusal(g)


def test_the_module_surface_the_app_reads():
    for name in ("DEFAULT_REPO", "BASE_REPO", "ENGINE_REPOS", "FAMILY", "MODEL_LABEL", "DEFAULTS", "PRECISIONS",
                 "HIDDEN_SIZE", "SCHEDULE_PEAK", "NEGATIVE_PRESET", "AUTO_NONE_MIN_VRAM_GB", "EnginePipe",
                 "LATENT_RGB", "LICENCE_NOTE", "load", "load_transformer", "load_text_encoder", "prepare_prompt"):
        assert hasattr(M, name), name
    assert M.DEFAULTS[M.FAMILY]["steps"] == 6 and M.DEFAULTS[M.FAMILY]["guidance"] == 1.0
    assert M.is_engine_repo(M.DEFAULT_REPO) and not M.is_engine_repo("OpenMOSS-Team/MOSS-Image-SFT")
    assert len(M.LATENT_RGB) == 24 and M.prepare_prompt("p", 1, 2) == ("p", 1, 2, None)
    for word in ("European Union", "United Kingdom", "Korea", "United States"):
        assert word in M.LICENCE_NOTE


def test_live_preview_shape():
    img = M.latent_preview(torch.zeros(1, 24, 8, 12), max_side=96)
    assert img is not None and img.size == (96, 64)
    assert M.latent_preview(torch.zeros(1, 128, 8, 8)) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)

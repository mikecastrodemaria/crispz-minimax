"""cz_minimax_lora on CPU: the three dialects of a MiniMax-H3 LoRA give the same result as
the delta merged into the weights, on a tiny module tree with the DiT's own names (fused
`to_qkv`, `[gate; value]` feed-forward)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
from safetensors.torch import save_file

import cz_minimax as M
import cz_minimax_lora as L

H, INNER, FFN, R = 8, 12, 10, 2


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(H, 3 * INNER, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(INNER, H, bias=False)])


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([M.GateFirstSwiGLU(H, FFN), nn.Identity(), nn.Linear(FFN, H, bias=False)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.ff = _FF()

    def forward(self, x):
        q, k, v = self.attn.to_qkv(x).chunk(3, dim=-1)
        x = x + self.attn.to_out[0](q * k + v)
        return x + self.ff.net[2](self.ff.net[0](x))


class _Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.refiner_blocks = nn.ModuleList([_Block()])


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.token_refiner = _Refiner()

    def forward(self, x):
        x = self.token_refiner.refiner_blocks[0](x)
        for b in self.transformer_blocks:
            x = b(x)
        return x


def _dit():
    torch.manual_seed(0)
    return _DiT().double().eval()


def _deltas(seed=1):
    """One low-rank pair per ComfyUI module name: {name: (down, up)}."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for pre in ("blocks.0.", "blocks.1.", "token_refiner.blocks.0."):
        for leaf, i, o in (("attn.qkv_proj", H, 3 * INNER), ("attn.out_proj", INNER, H),
                           ("mlp.fc1", H, 2 * FFN), ("mlp.fc2", FFN, H)):
            out[pre + leaf] = (torch.randn(R, i, generator=g, dtype=torch.float64),
                               torch.randn(o, R, generator=g, dtype=torch.float64))
    return out


_COMFY_TO_OURS = (("token_refiner.blocks.", "token_refiner.refiner_blocks."), (".attn.out_proj", ".attn.to_out.0"),
                  (".attn.qkv_proj", ".attn.to_qkv"), (".mlp.fc1", ".ff.net.0.proj"), (".mlp.fc2", ".ff.net.2"))


def _ours(name):
    if name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks."):]
    for a, b in _COMFY_TO_OURS:
        name = name.replace(a, b)
    return name


def _merged(deltas, scale):
    """The reference: the same deltas added into the weights."""
    dit = _dit()
    for name, (d, u) in deltas.items():
        dit.get_submodule(_ours(name)).weight.data += (u @ d) * scale
    return dit


def _save(sd, meta=None):
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    save_file({k: v.contiguous() for k, v in sd.items()}, path, metadata=meta)
    return path


def _comfy_file(deltas):
    sd = {}
    for name, (d, u) in deltas.items():
        sd[f"diffusion_model.{name}.lora_A.weight"] = d
        sd[f"diffusion_model.{name}.lora_B.weight"] = u
    return _save(sd)


def _kohya_file(deltas, alpha):
    sd = {}
    for name, (d, u) in deltas.items():
        base = "lora_unet_" + name.replace(".", "_")
        sd[base + ".lora_down.weight"] = d
        sd[base + ".lora_up.weight"] = u
        sd[base + ".alpha"] = torch.tensor(float(alpha))
    return _save(sd)


def _diffusers_file(deltas):
    """diffusers names: q, k, v separate (each its own down), feed-forward [value; gate]."""
    sd = {}
    for name, (d, u) in deltas.items():
        ours = _ours(name)
        if ours.endswith(".attn.to_qkv"):
            for i, part in enumerate("qkv"):
                key = "transformer." + ours[:-len("to_qkv")] + "to_" + part
                sd[key + ".lora_A.weight"] = d.clone()
                sd[key + ".lora_B.weight"] = u[i * INNER:(i + 1) * INNER].clone()
        elif ours.endswith(".ff.net.0.proj"):
            sd["transformer." + ours + ".lora_A.weight"] = d
            sd["transformer." + ours + ".lora_B.weight"] = torch.cat([u[FFN:], u[:FFN]], dim=0)
        else:
            sd["transformer." + ours + ".lora_A.weight"] = d
            sd["transformer." + ours + ".lora_B.weight"] = u
    return _save(sd)


def _x():
    return torch.randn(2, 5, H, generator=torch.Generator().manual_seed(7), dtype=torch.float64)


def test_the_three_dialects_equal_the_merged_weights():
    deltas = _deltas()
    want = _merged(deltas, 0.5)(_x())
    for path in (_comfy_file(deltas), _kohya_file(deltas, alpha=R), _diffusers_file(deltas)):
        try:
            dit = _dit()
            before = dit(_x())
            rep = L.apply(dit, [(path, 0.5)], fp32=[path])[0]
            got = dit(_x())
            assert rep["modules"] == 12 and not rep["unknown"] and not rep["missing"] and not rep["shape"], rep
            assert not torch.allclose(got, before)
            assert torch.allclose(got, want, atol=1e-9), path
            assert L.unload(dit) == 12
            assert torch.equal(dit(_x()), before)
        finally:
            os.remove(path)


def test_the_diffusers_dialect_is_placed_and_reordered():
    path = _diffusers_file(_deltas())
    try:
        dit = _dit()
        rep = L.apply(dit, [(path, 1.0)], fp32=[path])[0]
        assert rep["split"] == 9          # q, k, v on 3 attention modules
        assert rep["swapped"] == 3        # one feed-forward up matrix per block
    finally:
        os.remove(path)


def test_kohya_alpha_scales_the_delta():
    deltas = _deltas()
    path = _kohya_file(deltas, alpha=1.0)             # alpha / rank = 0.5
    try:
        dit = _dit()
        L.apply(dit, [(path, 1.0)], fp32=[path])
        assert torch.allclose(dit(_x()), _merged(deltas, 0.5)(_x()), atol=1e-9)
    finally:
        os.remove(path)


def test_rescale_and_two_files_stack():
    a, b = _deltas(1), _deltas(2)
    pa, pb = _comfy_file(a), _kohya_file(b, alpha=R)
    try:
        dit = _dit()
        L.apply(dit, [(pa, 1.0), (pb, 1.0)], fp32=[pa, pb])
        L.rescale(dit, [0.25, -0.5])
        ref = _dit()
        for deltas, s in ((a, 0.25), (b, -0.5)):
            for name, (d, u) in deltas.items():
                ref.get_submodule(_ours(name)).weight.data += (u @ d) * s
        assert torch.allclose(dit(_x()), ref(_x()), atol=1e-9)
        L.rescale(dit, [0.0, 0.0])
        assert torch.allclose(dit(_x()), _dit()(_x()), atol=1e-12)
    finally:
        os.remove(pa)
        os.remove(pb)


def test_keys_of_no_module_are_counted_not_dropped():
    deltas = _deltas()
    sd = {}
    for name, (d, u) in deltas.items():
        sd[f"diffusion_model.{name}.lora_A.weight"] = d
        sd[f"diffusion_model.{name}.lora_B.weight"] = u
    # the audio stream of an H3 video LoRA, and a block this DiT does not have
    sd["diffusion_model.audio_patch_proj.lora_A.weight"] = torch.zeros(R, 4, dtype=torch.float64)
    sd["diffusion_model.audio_patch_proj.lora_B.weight"] = torch.zeros(4, R, dtype=torch.float64)
    sd["diffusion_model.blocks.9.attn.out_proj.lora_A.weight"] = torch.zeros(R, INNER, dtype=torch.float64)
    sd["diffusion_model.blocks.9.attn.out_proj.lora_B.weight"] = torch.zeros(H, R, dtype=torch.float64)
    # a module of another size (a LoRA of another model with the same names)
    sd["diffusion_model.blocks.0.mlp.fc2.lora_A.weight"] = torch.zeros(R, FFN + 1, dtype=torch.float64)
    path = _save(sd)
    try:
        rep = L.apply(_dit(), [(path, 1.0)], fp32=[path])[0]
        assert rep["modules"] == 11
        assert sorted(rep["missing"]) == ["audio_patch_proj", "transformer_blocks.9.attn.to_out.0"]
        assert rep["shape"] == ["transformer_blocks.0.ff.net.2"]
    finally:
        os.remove(path)


def test_refusals_are_named():
    diffsynth = _save({"blocks.0.attn.qkv_proj.lora_A.default.weight": torch.zeros(R, H),
                       "blocks.0.attn.qkv_proj.lora_B.default.weight": torch.zeros(3 * INNER, R)})
    full = _save({"diffusion_model.blocks.0.attn.out_proj.weight": torch.zeros(H, INNER)})
    try:
        assert "DiffSynth" in L.refusal(diffsynth)
        assert "replaces whole layers" in L.refusal(full)
        try:
            L.apply(_dit(), [(diffsynth, 1.0)])
            raise AssertionError("a DiffSynth file must be refused")
        except ValueError as e:
            assert "DiffSynth" in str(e)
    finally:
        os.remove(diffsynth)
        os.remove(full)


def test_the_family_is_read_from_the_names_and_the_width():
    comfy = ["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
    kohya = ["lora_unet_blocks_0_attn_qkv_proj.lora_down.weight", "lora_unet_blocks_0_attn_qkv_proj.alpha"]
    assert M.family_from_keys(comfy) == M.FAMILY
    assert M.family_from_keys(kohya) == M.FAMILY
    dif = "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
    with_refiner = [dif, "transformer.token_refiner.refiner_blocks.0.attn.to_q.lora_A.weight"]
    assert M.family_from_keys(with_refiner) == M.FAMILY
    # diffusers names alone exist in other families: the width decides
    assert M.family_from_header({dif: {"shape": [16, 5376]}}) == M.FAMILY
    assert not L.is_h3_lora_keys([dif], {dif: [16, 3072]})
    assert not L.is_h3_lora_keys(["blocks.0.attn.qkv_proj.weight"])     # a checkpoint, not a LoRA


def test_a_metadata_alpha_serves_a_peft_file():
    deltas = _deltas()
    sd = {}
    for name, (d, u) in deltas.items():
        sd[f"diffusion_model.{name}.lora_A.weight"] = d
        sd[f"diffusion_model.{name}.lora_B.weight"] = u
    path = _save(sd, meta={"alpha": "1"})                 # alpha / rank = 0.5
    try:
        dit = _dit()
        L.apply(dit, [(path, 1.0)], fp32=[path])
        assert torch.allclose(dit(_x()), _merged(deltas, 0.5)(_x()), atol=1e-9)
    finally:
        os.remove(path)

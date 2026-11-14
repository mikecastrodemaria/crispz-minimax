"""crispz-minimax - the MiniMax-H3-Image engine (diffusers ModularPipeline, GGUF weights).

What is here:
  * a GGUF reader that keeps the K-quantized Linear weights quantized (diffusers'
    GGUFParameter / GGUFLinear: dequantized layer by layer at each forward pass) and
    turns everything else into plain tensors;
  * the "pruned" (curve form) variant of the MiniMax-H3-Image transformer, the only one
    the light files ship: the dense adaLN time projection is replaced by a rank 8
    factorization of the time embedding curve (`adaln_t_table` [1025, 8] + one
    [out, 8] projection per block). diffusers 0.43.0 only describes the dense release
    checkpoint, so the three modules that differ are swapped on its model;
  * the Qwen3-VL-32B text encoder cut to the 50 layers the model reads, from a GGUF with
    transformers key names (unsloth/MiniMax-H3-GGUF);
  * `load_pipeline`: the official `MiniMaxH3ImageBlocks` chain with those components and
    a text encoder -> transformer -> VAE offload chain (the encoder and the DiT do not
    fit together in 32 GB).

Key layout of the single files (ComfyUI layout, also the GGUFs made from it) and its
mapping to diffusers names: see `_convert_keys`. The mapping follows diffusers' own
converter for the MiniMax-H3 video model (`convert_minimax_h3_transformer_checkpoint_to_diffusers`,
Apache 2.0): `[q; k; v]` stacked rows, feed-forward fused as `[gate; value]`.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn

OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3-Image"

# The 6-step PDD grid the per-step output heads were distilled on (pdd_config.json).
PDD_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
              0.5327739119529724, 0.23200887441635132, 0.0]
PDD_STEPS = len(PDD_SIGMAS) - 1

TEXT_ENCODER_LAYERS = 50   # the DiT reads hidden_states[50]: the output of layer index 49


def _log(msg):
    print(f"[minimax] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# GGUF reading
# --------------------------------------------------------------------------------------
def read_gguf(path, device=None):
    """name -> tensor. K-quantized 2D tensors stay quantized (GGUFParameter), the rest
    (F32, F16, BF16, and anything that is not a matrix) becomes a plain torch tensor.
    Returns (tensors, metadata) where metadata holds the string KV pairs of the file
    (Abiray's files carry `config` and `comfy.gguf.orig_shape.*`).

    `device`: move each tensor there as it is read, so the whole file never sits twice
    in RAM."""
    import gguf
    import numpy as np
    from diffusers.quantizers.gguf.utils import (GGUFParameter, SUPPORTED_GGUF_QUANT_TYPES,
                                                 dequantize_gguf_tensor)

    reader = gguf.GGUFReader(path)
    meta = {}
    orig_shapes = {}
    for key, field in reader.fields.items():
        try:
            if key.startswith("comfy.gguf.orig_shape."):
                orig_shapes[key[len("comfy.gguf.orig_shape."):]] = tuple(
                    int(field.parts[i][0]) for i in field.data)
            elif field.types and field.types[0] == gguf.GGUFValueType.STRING:
                meta[key] = bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
        except Exception:
            pass

    Q = gguf.GGMLQuantizationType
    plain = {Q.F32: torch.float32, Q.F16: torch.float16, Q.BF16: torch.bfloat16}
    out = {}
    for t in reader.tensors:
        shape = tuple(int(x) for x in reversed(t.shape))
        if t.name in orig_shapes:
            shape = orig_shapes[t.name]
        qt = t.tensor_type
        raw = torch.from_numpy(np.array(t.data, copy=True))
        if qt in plain:
            if raw.dtype != plain[qt]:
                raw = raw.view(torch.uint8).view(plain[qt])
            ten = raw.reshape(shape)
        else:
            if qt not in SUPPORTED_GGUF_QUANT_TYPES:
                raise ValueError(f"{t.name}: GGUF quantization {qt!r} is not supported")
            ten = GGUFParameter(raw, quant_type=qt)
            if len(shape) != 2:
                # a quantized vector or conv kernel: no GGUFLinear for it, dequantize now
                ten = dequantize_gguf_tensor(ten).reshape(shape)
        if device is not None:
            ten = ten.to(device)
        out[t.name] = ten
    return out, meta


def _is_quant(t):
    return hasattr(t, "quant_type")


def _plain(t, dtype):
    """A plain tensor of `dtype` out of a tensor that may be GGUF-quantized."""
    if _is_quant(t):
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
        t = dequantize_gguf_tensor(t)
    return t.to(dtype)


def _to_gguf_linears(model, state_dict, compute_dtype):
    """Replace every nn.Linear whose weight is quantized in `state_dict` by a GGUFLinear."""
    from diffusers.quantizers.gguf.utils import _replace_with_gguf_linear
    _replace_with_gguf_linear(model, compute_dtype, state_dict)


def _assign_state(model, state_dict, what):
    """Put `state_dict` into `model` by assignment. nn.Module.load_state_dict cannot do it:
    a quantized weight is a byte matrix, whose shape is not the Linear's."""
    target = model.state_dict()
    missing = sorted(set(target) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(target))
    if missing or unexpected:
        raise RuntimeError(f"{what} keys do not match: missing {missing[:6]} ({len(missing)}), "
                           f"unexpected {unexpected[:6]} ({len(unexpected)})")
    for key, val in state_dict.items():
        path, _, leaf = key.rpartition(".")
        mod = model.get_submodule(path) if path else model
        if _is_quant(val):
            if tuple(val.quant_shape) != tuple(target[key].shape):
                raise RuntimeError(f"{what}: {key} is {tuple(val.quant_shape)}, expected {tuple(target[key].shape)}")
        elif tuple(val.shape) != tuple(target[key].shape) and val.numel() == target[key].numel():
            # GGUF holds at most 4 dimensions: a 5D kernel (the vision tower's 3D patch
            # embedding) is stored with its two leading dimensions merged
            val = val.reshape(target[key].shape)
        elif tuple(val.shape) != tuple(target[key].shape):
            raise RuntimeError(f"{what}: {key} is {tuple(val.shape)}, expected {tuple(target[key].shape)}")
        if leaf in mod._buffers:
            mod._buffers[leaf] = val
        else:
            mod._parameters[leaf] = val if isinstance(val, nn.Parameter) else nn.Parameter(val, requires_grad=False)


# --------------------------------------------------------------------------------------
# The pruned (curve form) transformer
# --------------------------------------------------------------------------------------
class CurveTimeTable(nn.Module):
    """Stands in for `time_embedder`: `t` in [0, 1] -> the interpolated row of the
    time-embedding curve, sampled on `table.shape[0]` evenly spaced points. Out of range
    values clamp to the ends of the curve."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table.to(torch.float32), persistent=True)

    def forward(self, t):
        table = self.table
        pos = t.to(table.device, torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


class _PassThrough(nn.Module):
    """Stands in for `time_proj`: the curve table reads `t` itself, not its sinusoid."""

    def forward(self, t):
        return t


class CurveAdaLN(nn.Module):
    """Block modulation from the curve coordinates: no activation (the curve already is
    the activated embedding, factorized), projection kept in float32, result cast to the
    block stack's dtype."""

    def __init__(self, rank, hidden_size, modalities, out_dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.linear = nn.Linear(rank, 6 * hidden_size * modalities, bias=True)

    def forward(self, temb):
        x = self.linear(temb.to(self.linear.weight.dtype)).to(self.out_dtype)
        return x.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class CurveAdaLNOut(nn.Module):
    """Final norm, shift / scale from the curve coordinates."""

    def __init__(self, rank, hidden_size, eps, out_dtype):
        super().__init__()
        self.out_dtype = out_dtype
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(rank, 2 * hidden_size, bias=True)

    def forward(self, hidden_states, temb, timestep_indices):
        mod = self.linear(temb.to(self.linear.weight.dtype)).to(self.out_dtype)
        shift, scale = mod.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices)


class GateFirstSwiGLU(nn.Module):
    """SwiGLU whose fused projection is stored `[gate; value]` (the checkpoint's order;
    diffusers' SwiGLU reads `[value; gate]`). Keeping the stored order means a quantized
    weight is used as it is, without re-cutting its rows."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=False)

    def forward(self, hidden_states):
        gate, value = self.proj(hidden_states).chunk(2, dim=-1)
        return value * nn.functional.silu(gate)


_RENAMES = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("video_patch_proj.", "proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
    (".attn.qkv_proj.", ".attn.to_qkv."),
    (".mlp.fc1.", ".ff.net.0.proj."),
    (".mlp.fc2.", ".ff.net.2."),
)


def _convert_keys(sd, compute_dtype):
    """ComfyUI single-file names -> the names of the model built by `build_transformer`.
    The fused `qkv` and the `[gate; value]` feed-forward are kept as stored."""
    out = {}
    table = None
    head_w = head_b = None
    for key, val in sd.items():
        if key == "rope.inv_freq":
            continue                      # recomputed from the config
        if key == "adaln_t_table":
            table = _plain(val, torch.float32)
            continue
        if key == "final_layer.video_out.weight":
            head_w = _plain(val, torch.float32)
            continue
        if key == "final_layer.video_out.bias":
            head_b = _plain(val, torch.float32)
            continue
        new = key
        if new.startswith("blocks."):
            new = "transformer_blocks." + new[len("blocks."):]
        for a, b in _RENAMES:
            new = new.replace(a, b)
        if _is_quant(val):
            out[new] = val
        elif "adaln_proj.linear." in new or new.startswith("norm_out.linear.") or new.startswith("proj_in."):
            out[new] = val.to(torch.float32)          # the fp32 islands
        else:
            out[new] = val.to(compute_dtype)
    if table is None:
        raise ValueError("no `adaln_t_table` in this file: it is not a pruned (curve form) "
                         "MiniMax-H3-Image checkpoint. The dense 66 GB release is not handled here.")
    if head_w is None:
        raise ValueError("no `final_layer.video_out` in this file")
    out["time_embedder.table"] = table
    # The PDD head bank: block 0 is a full head, the later blocks are offsets from it.
    # The step from sigma[k] to sigma[k+1] uses head 0 + offset k (k >= 1).
    patch_dim = head_w.shape[0] // PDD_STEPS
    rows = head_w.reshape(PDD_STEPS, patch_dim, head_w.shape[1])
    brows = head_b.reshape(PDD_STEPS, patch_dim)
    for k in range(PDD_STEPS):
        out[f"pdd_proj_out.{k}.weight"] = (rows[0] if k == 0 else rows[0] + rows[k]).clone()
        out[f"pdd_proj_out.{k}.bias"] = (brows[0] if k == 0 else brows[0] + brows[k]).clone()
    return out


def build_transformer(state_dict, compute_dtype=torch.bfloat16):
    """The diffusers MiniMaxH3ImageTransformer2DModel, reshaped for the curve form and
    loaded from a converted state dict. Linear weights that are quantized in the state
    dict stay quantized."""
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3ImageTransformer2DModel

    rank = state_dict["time_embedder.table"].shape[1]
    with init_empty_weights():
        model = MiniMaxH3ImageTransformer2DModel(time_embed_dim=rank, time_embed_hidden_dim=rank)
        cfg = model.config
        hidden = cfg.hidden_size
        inner = cfg.num_attention_heads * cfg.attention_head_dim
        model.time_proj = _PassThrough()
        model.time_embedder = CurveTimeTable(torch.empty_like(state_dict["time_embedder.table"]))
        model.norm_out = CurveAdaLNOut(rank, hidden, cfg.final_norm_eps, compute_dtype)
        blocks = list(model.transformer_blocks) + list(model.token_refiner.refiner_blocks)
        for blk in blocks:
            attn = blk.attn
            del attn.to_q, attn.to_k, attn.to_v
            attn.to_qkv = nn.Linear(hidden, 3 * inner, bias=False)
            attn.fused_projections = True
            blk.ff.net[0] = GateFirstSwiGLU(hidden, cfg.ffn_dim)
        for blk in model.transformer_blocks:
            modalities = state_dict["transformer_blocks.0.adaln_proj.linear.weight"].shape[0] // (6 * hidden)
            blk.adaln_proj = CurveAdaLN(rank, hidden, modalities, compute_dtype)
    _to_gguf_linears(model, state_dict, compute_dtype)
    _assign_state(model, state_dict, "transformer")
    model.requires_grad_(False)
    model.eval()
    return model


def load_transformer(path, compute_dtype=torch.bfloat16, precision=None, single_file=None,
                     token=None):
    """`path`: the GGUF file. Called by the app as load_transformer(repo, dtype, precision,
    single_file=path): the file wins over the repo, which only lends the configs."""
    t0 = time.time()
    path = single_file or path
    if not str(path).lower().endswith(".gguf"):
        raise ValueError("only GGUF transformers are handled for now")
    sd, meta = read_gguf(path)
    n_quant = sum(1 for v in sd.values() if _is_quant(v))
    conv = _convert_keys(sd, compute_dtype)
    del sd
    model = build_transformer(conv, compute_dtype)
    cfg = meta.get("config")
    if cfg:
        try:
            sig = json.loads(cfg)["transformer"]["pdd_sigmas"]
            if [round(float(s), 6) for s in sig] != [round(s, 6) for s in PDD_SIGMAS]:
                _log(f"WARNING: the file's PDD sigmas differ from the release grid: {sig}")
        except Exception:
            pass
    _log(f"transformer: {os.path.basename(str(path))}, {n_quant} quantized matrices, "
         f"loaded in {time.time() - t0:.1f}s")
    return model


# --------------------------------------------------------------------------------------
# The text encoder (Qwen3-VL-32B, 50 layers, GGUF with transformers key names)
# --------------------------------------------------------------------------------------
def _text_encoder_config(repo=OFFICIAL_REPO):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    path = hf_hub_download(repo, "text_encoder/config.json")
    cfg = AutoConfig.from_pretrained(os.path.dirname(path))
    cfg.text_config.num_hidden_layers = TEXT_ENCODER_LAYERS
    return cfg


def load_text_encoder(path, compute_dtype=torch.bfloat16, precision=None, token=None,
                      config_repo=None, repo=OFFICIAL_REPO):
    """Qwen3VLForConditionalGeneration with 50 decoder layers and no final norm, so that
    `hidden_states[50]` is the output of the 50th layer, the state the DiT reads. The
    language-model head is dropped (never used)."""
    from accelerate import init_empty_weights
    from transformers import Qwen3VLForConditionalGeneration

    t0 = time.time()
    repo = config_repo or repo
    sd, _ = read_gguf(path)
    cfg = _text_encoder_config(repo)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)
    model.lm_head = nn.Identity()
    model.model.language_model.norm = nn.Identity()

    target = set(model.state_dict().keys())
    conv = {}
    for key, val in sd.items():
        cands = [key, "model." + key]
        if key.startswith("model."):
            cands.append("model.language_model." + key[len("model."):])
        new = next((c for c in cands if c in target), None)
        if new is None:
            continue
        conv[new] = val if _is_quant(val) else val.to(compute_dtype)
    del sd
    _to_gguf_linears(model, conv, compute_dtype)
    _assign_state(model, conv, "text encoder")
    # The pipeline refuses a stack of exactly 50 layers because its last hidden state is
    # normally post-norm. Here the norm is removed, so hidden_states[50] IS the pre-norm
    # output of layer 50: declare one more layer than there is to pass that check.
    model.config.text_config.num_hidden_layers = TEXT_ENCODER_LAYERS + 1
    model.requires_grad_(False)
    model.eval()
    _log(f"text encoder: {os.path.basename(str(path))}, loaded in {time.time() - t0:.1f}s")
    return model


# --------------------------------------------------------------------------------------
# What the rest of the app reads from the engine module
# --------------------------------------------------------------------------------------
DEFAULT_REPO = OFFICIAL_REPO      # configs, tokenizer, processor, scheduler and VAE
BASE_REPO = OFFICIAL_REPO
ENGINE_REPOS = (OFFICIAL_REPO,)
FAMILY = "minimax"
MODEL_LABEL = "MiniMax-H3-Image"
DEFAULTS = {"minimax": {"steps": PDD_STEPS, "guidance": 1.0, "size": 2048}}
# One way to hold the weights for now: GGUF, quantized on the card.
PRECISIONS = ("gguf",)
HIDDEN_SIZE = 5120                # the text encoder's width (what a replacement must match)
SCHEDULE_PEAK = 1.0               # no CFG: the guidance is baked into the weights
NEGATIVE_PRESET = ""              # no negative prompt on this model
# Under this much VRAM the text encoder and the DiT take the card in turn.
AUTO_NONE_MIN_VRAM_GB = 46

GRID = 32                         # 16x VAE, 2x2 patches
MIN_SIDE, MAX_SIDE = 256, 6272
MAX_PIXELS = 20 * 1024 * 1024     # the release card's working range ends around 20 Mpx
SMALL_PIXELS = 1024 * 1024        # under this the model is far from its 4 to 16 Mpx range
EDIT_MAX_REFS = 16
INPAINT_WITH_REFERENCE = True     # an inpaint also hands the source to the model as <Picture 1>

# The files fetched when none is set and none is found in ./checkpoints.
DEFAULT_TRANSFORMER = ("unsloth/MiniMax-H3-Image-GGUF", "minimax_h3_image_pruned-UD-Q5_K_XL.gguf")
DEFAULT_TEXT_ENCODER = ("unsloth/MiniMax-H3-GGUF", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf")

LICENCE_NOTE = ("MiniMax-H3-Image Community License: rights are granted worldwide EXCEPT in the "
                "European Union, the United Kingdom, the Republic of Korea and the United States "
                "(separate authorization: https://platform.minimax.io/license). You are "
                "responsible for checking that you may use the model where you are.")

# Other crispz families, named so that their files are refused with the app that runs them.
_MOSS_MARKERS = (".video_embedder.", ".text_embedder.", ".shared_block_modulation.",
                 ".final_linear_video.", ".proj.ffn_up_gate.", ".proj.ffn_down.")
_IDEOGRAM_MARKERS = (".llm_cond_proj.", ".llm_cond_norm.", ".embed_image_indicator.",
                     ".unconditional_transformer.")
_TEXT_ENCODER_MARKERS = ("model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight")


def gguf_tensor_names(path):
    """The tensor names of a GGUF file (header only, the data is not read)."""
    import gguf
    return [t.name for t in gguf.GGUFReader(path).tensors]


def family_from_keys(keys):
    """'minimax' for a MiniMax-H3-Image transformer file, else what the family markers of
    the other crispz engines say (so that their files are refused by name)."""
    keys = set(keys or ())
    if "final_layer.video_out.weight" in keys and "video_patch_proj.weight" in keys \
            and not any(k.startswith("audio_patch_proj") for k in keys):
        return FAMILY
    dotted = ["." + k for k in keys if isinstance(k, str)]
    if any(m in k for k in dotted for m in _MOSS_MARKERS):
        return "moss"
    if any(m in k for k in dotted for m in _IDEOGRAM_MARKERS):
        return "ideogram"
    try:
        import cz_flux3
        return cz_flux3.family_from_keys(keys)
    except Exception:
        return None


def family_from_header(hdr):
    return family_from_keys(list((hdr or {}).keys()))


def is_engine_repo(repo):
    """True for the official repo or a local folder laid out like it."""
    r = str(repo or "").strip().rstrip("/\\")
    if r in ENGINE_REPOS:
        return True
    try:
        idx = os.path.join(r, "modular_model_index.json")
        if os.path.isfile(idx):
            with open(idx, encoding="utf-8") as f:
                return "MiniMaxH3Image" in str(json.load(f).get("_class_name", ""))
    except Exception:
        pass
    return False


def is_base_repo(repo):
    return False          # one checkpoint, distilled; there is no undistilled "base"


def variant_of(repo):
    return FAMILY


def variant_from_keys(keys):
    return None


def _bfl_quant_of(path):
    return None


def is_text_encoder_file(path):
    """True when `path` is a GGUF of the Qwen3-VL-32B encoder with transformers names."""
    try:
        if not str(path).lower().endswith(".gguf"):
            return False
        names = set(gguf_tensor_names(path))
        return all(m in names for m in _TEXT_ENCODER_MARKERS)
    except Exception:
        return False


def single_file_refusal(path):
    """Why this file cannot be the transformer, or None when it can."""
    name = os.path.basename(str(path))
    if not str(path).lower().endswith(".gguf"):
        return (f"{name}: only GGUF transformers are loaded for now (safetensors fp8, int8 "
                f"ConvRot and nvfp4 are not). Take a GGUF of the same model, for instance from "
                f"{DEFAULT_TRANSFORMER[0]}.")
    try:
        names = set(gguf_tensor_names(path))
    except Exception as e:
        return f"{name}: unreadable GGUF ({type(e).__name__}: {e})"
    if all(m in names for m in _TEXT_ENCODER_MARKERS):
        return f"{name} is a text encoder, not a transformer: choose it in Text encoder."
    if "final_layer.video_out.weight" not in names:
        return f"{name}: not a MiniMax-H3-Image transformer."
    if "adaln_t_table" not in names:
        return (f"{name}: this is the dense release layout, only the pruned (curve form) "
                f"files are handled.")
    if any(n.startswith("audio_patch_proj") for n in names):
        return f"{name}: this is the MiniMax-H3 VIDEO model, not MiniMax-H3-Image."
    return None


def edit_ref_max_edge():
    return 2048           # the pipeline's own rule (reference_max_short_edge), not a setting


def fit_size(width, height):
    """The size the model runs: multiples of 32, sides within bounds, 20 Mpx at most."""
    w, h = int(width or DEFAULTS[FAMILY]["size"]), int(height or DEFAULTS[FAMILY]["size"])
    if w * h > MAX_PIXELS:
        k = (MAX_PIXELS / float(w * h)) ** 0.5
        w, h = int(w * k), int(h * k)
    w = min(max(int(round(w / GRID)) * GRID, MIN_SIDE), MAX_SIDE)
    h = min(max(int(round(h / GRID)) * GRID, MIN_SIDE), MAX_SIDE)
    return w, h


def prepare_prompt(prompt, width=None, height=None, fit=False, place=None):
    """No layout language on this model: the prompt and the size go through unchanged."""
    return prompt, width, height, None


def strength_to_start(strength):
    """Which step of the fixed 6-step grid an img2img of this strength starts at. The grid
    cannot be re-cut (one output head per step), so the strength snaps to the nearest of
    its sigmas: 1.0, 0.97, 0.89, 0.75, 0.53, 0.23."""
    s = 1.0 if strength is None else float(min(max(float(strength), 0.0), 1.0))
    return min(range(PDD_STEPS), key=lambda k: abs(PDD_SIGMAS[k] - s))


def _find_local(kind):
    """A file of this kind ('transformer' / 'text_encoder') under ./checkpoints, else None."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    found = []
    for base, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".gguf"):
                found.append(os.path.join(base, f))
    for path in sorted(found):
        try:
            if kind == "text_encoder":
                if is_text_encoder_file(path):
                    return path
            elif single_file_refusal(path) is None:
                return path
        except Exception:
            continue
    return None


def default_file(kind, token=None):
    """The file to load when none is set: one found in ./checkpoints, else the default
    downloaded into the Hugging Face cache (said in the log, with the licence note)."""
    path = _find_local(kind)
    if path:
        return path
    from huggingface_hub import hf_hub_download
    repo, name = DEFAULT_TEXT_ENCODER if kind == "text_encoder" else DEFAULT_TRANSFORMER
    _log(f"no {kind.replace('_', ' ')} file set or found in ./checkpoints: fetching {repo}/{name} "
         f"into the Hugging Face cache (15 to 18 GB)")
    _log(LICENCE_NOTE)
    return hf_hub_download(repo, name, token=token)


# --------------------------------------------------------------------------------------
# Live preview
# --------------------------------------------------------------------------------------
# Normalized latents (24 channels, one per 16x16 pixels) -> RGB in [-1, 1]: a least-squares
# fit on six renders (tools/minimax_fit_preview.py), mean error 7.5 on 255.
LATENT_RGB = [
    [0.05682, -0.11963, -0.13233], [-0.07727, 0.03948, -0.03811], [0.49400, 0.38194, 0.21002],
    [0.14443, 0.02061, -0.35107], [0.04266, 0.15657, -0.16204], [0.06919, 0.08296, -0.11452],
    [-0.00794, -0.02957, -0.00393], [0.00851, -0.00877, -0.02126], [-0.00256, -0.00827, -0.00077],
    [0.24179, 0.08177, 0.16398], [0.02470, 0.03265, 0.06386], [-0.06669, -0.02995, 0.03216],
    [0.00238, -0.00943, 0.02870], [-0.01355, 0.01240, -0.00778], [-0.01929, -0.00289, 0.02597],
    [0.01796, 0.00368, -0.02605], [-0.00121, 0.01329, -0.00337], [-0.00823, -0.00158, 0.01552],
    [0.02033, 0.01874, -0.00362], [0.01579, -0.01368, 0.00648], [-0.00061, 0.00801, -0.00395],
    [-0.01021, -0.00595, -0.00096], [-0.01196, 0.01602, 0.00136], [0.00462, 0.00640, 0.00809],
]
LATENT_RGB_BIAS = [0.12912, 0.08993, 0.02948]


def latent_preview(latents, max_side=512):
    """(B, 24, h, w) normalized latents -> a small PIL image, or None when the shape is not
    this model's. Runs where the latents are; only the uint8 result crosses back."""
    from PIL import Image
    z = latents[0] if latents.dim() == 4 else latents
    if z.dim() != 3 or z.shape[0] != len(LATENT_RGB):
        return None
    z = z.detach().float()
    m = torch.tensor(LATENT_RGB, dtype=z.dtype, device=z.device)
    b = torch.tensor(LATENT_RGB_BIAS, dtype=z.dtype, device=z.device)
    rgb = (z.permute(1, 2, 0) @ m + b).clamp(-1.0, 1.0).add(1.0).mul(127.5).round().byte().cpu().numpy()
    img = Image.fromarray(rgb, mode="RGB")
    side = max(img.size)
    if side < max_side:
        k = max_side / float(side)
        img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.BILINEAR)
    return img


# --------------------------------------------------------------------------------------
# The block chains
# --------------------------------------------------------------------------------------
def _make_step_hook():
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks

    class CrispzStepHook(ModularPipelineBlocks):
        model_name = "minimax-h3-image"

        @property
        def description(self):
            return "crispz: per-step hook (inpaint blending, preview callback, interrupt)."

        @property
        def inputs(self):
            return []

        @property
        def intermediate_outputs(self):
            return []

        @torch.no_grad()
        def __call__(self, components, block_state, i, t):
            owner = getattr(self, "owner", None)
            if owner is not None:
                owner._on_step(block_state, i, t)
            return components, block_state

    return CrispzStepHook()


def _make_start_block():
    """After `set_timesteps`: drops the first steps of the plan when the call starts from an
    image (img2img / inpaint). The transformer is told the offset so that each remaining
    step still uses ITS output head."""
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam

    class CrispzStart(ModularPipelineBlocks):
        model_name = "minimax-h3-image"

        @property
        def description(self):
            return "crispz: start the fixed schedule at a later step (img2img / inpaint)."

        @property
        def inputs(self):
            return [InputParam(name="timesteps", required=True),
                    InputParam(name="row_timestep_plan", required=True)]

        @property
        def intermediate_outputs(self):
            return [OutputParam("timesteps"), OutputParam("row_timestep_plan")]

        @torch.no_grad()
        def __call__(self, components, state):
            owner = getattr(self, "owner", None)
            k = int(getattr(owner, "_start", 0) or 0) if owner is not None else 0
            if k:
                bs = self.get_block_state(state)
                bs.timesteps = bs.timesteps[k:]
                bs.row_timestep_plan = list(bs.row_timestep_plan)[k:]
                self.set_block_state(state, bs)
            return components, state

    return CrispzStart()


def _walk(blocks, fn):
    subs = getattr(blocks, "sub_blocks", None)
    if not subs:
        return
    fn(blocks, subs)
    for child in list(subs.values()):
        _walk(child, fn)


def build_blocks():
    """(main chain, text chain, crispz blocks). The main chain is the official one minus
    the text encoder (crispz encodes and caches the conditioning itself, so that the 18 GB
    encoder leaves the card before anything else happens), plus the two crispz blocks.
    The text chain is the official reference resize + text encoder."""
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3_image import MiniMaxH3ImageBlocks
    from diffusers.modular_pipelines.modular_pipeline import LoopSequentialPipelineBlocks

    main = MiniMaxH3ImageBlocks()
    text = SequentialPipelineBlocks.from_blocks_dict({
        "before_encode": main.sub_blocks["before_encode"].__class__(),
        "text_encoder": main.sub_blocks["text_encoder"].__class__(),
    })
    main.sub_blocks.pop("text_encoder")
    crispz = []

    def patch(block, subs):
        if isinstance(block, LoopSequentialPipelineBlocks):
            if "crispz_hook" not in subs:
                hook = _make_step_hook()
                subs.insert("crispz_hook", hook, len(subs))
                crispz.append(hook)
        elif "set_timesteps" in subs and "crispz_start" not in subs:
            start = _make_start_block()
            subs.insert("crispz_start", start, list(subs.keys()).index("set_timesteps") + 1)
            crispz.append(start)
    _walk(main, patch)
    return main, text, crispz


def _offset_forward(transformer):
    """Makes `transformer.forward` add `transformer._cz_pdd_offset` to the step index it is
    given (the loop counts from 0 even when the schedule starts later)."""
    import functools
    if getattr(transformer, "_cz_offset_installed", False):
        return
    inner = transformer.forward

    @functools.wraps(inner)
    def forward(*args, **kwargs):
        if "pdd_step_index" in kwargs:
            kwargs["pdd_step_index"] = int(kwargs["pdd_step_index"]) + int(
                getattr(transformer, "_cz_pdd_offset", 0) or 0)
        return inner(*args, **kwargs)
    transformer.forward = forward
    transformer._cz_pdd_offset = 0
    transformer._cz_offset_installed = True


class _Interrupted(Exception):
    """Raised by the step hook when `_interrupt` is set: unwinds the modular loop."""


def _quiet_interrupt(active):
    """The modular loop logs an error when a block raises: silenced for our own stop."""
    try:
        import cz_flux3
        if active is None:
            cz_flux3._install_quiet_interrupt()
        else:
            cz_flux3._QuietInterrupt.active = bool(active)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# The adapter the app calls
# --------------------------------------------------------------------------------------
class MiniMaxPipe:
    """A DiffusionPipeline-shaped face on the modular MiniMax-H3-Image chain, with the call
    convention the crispz engines share:

      pipe(prompt=..., width=..., height=..., generator=..., callback_on_step_end=cb)
      pipe(prompt=..., image=[refs])                      edit with references (native)
      pipe(prompt=..., image=init, strength=0.5)          img2img (emulated)
      pipe(prompt=..., image=init, mask_image=m, ...)     inpaint (emulated)

    Fixed by the model, whatever is asked: 6 steps, no CFG, no negative prompt (each said
    once in the log when a call asks otherwise)."""

    def __init__(self, mp, text_pipe, crispz_blocks, repo=DEFAULT_REPO):
        self._mp = mp
        self._text_pipe = text_pipe
        self.repo = repo
        self.variant = FAMILY
        self._interrupt = False
        self._all_hooks = []
        self._offload_device = None
        self._cfg = {"is_distilled": True}
        self._ctx = None
        self._start = 0
        self._last_x0 = None
        self._cz_run_steps = None
        self._said = set()
        self._embeds = {}
        self.released = False
        self.applied_loras = []
        # init_pipeline works on a COPY of the blocks: the crispz blocks are looked up in
        # the pipeline's own chain, not in the list they were built into.
        found = []

        def collect(_block, subs):
            for name, child in subs.items():
                if name in ("crispz_hook", "crispz_start"):
                    found.append(child)
        _walk(getattr(mp, "_blocks", None) or mp.blocks, collect)
        for b in found or crispz_blocks:
            b.owner = self
        self._crispz_blocks = len(found)
        _offset_forward(mp.transformer)
        _quiet_interrupt(None)

    # -- plumbing ----------------------------------------------------------------------
    def __getattr__(self, name):
        if name.startswith("__") or name == "_mp":
            raise AttributeError(name)
        if name in ("text_encoder", "tokenizer", "processor"):
            return getattr(self.__dict__["_text_pipe"], name)
        return getattr(self.__dict__["_mp"], name)

    @property
    def config(self):
        import types
        return types.SimpleNamespace(**self._cfg)

    def register_to_config(self, **kw):
        self._cfg.update(kw)

    def register_modules(self, **kw):
        """The transformer (or another component) hot swap."""
        text = {k: v for k, v in kw.items() if k in ("text_encoder", "tokenizer", "processor")}
        rest = {k: v for k, v in kw.items() if k not in text}
        if rest:
            self._mp.update_components(**rest)
        if text:
            self._text_pipe.update_components(**text)
            self._embeds.clear()
        if "transformer" in kw:
            _offset_forward(kw["transformer"])

    def _say_once(self, key, msg):
        if key not in self._said:
            self._said.add(key)
            _log(msg)

    def _modules(self):
        return [m for m in (self._text_pipe.text_encoder, self._mp.transformer, self._mp.vae)
                if isinstance(m, nn.Module)]

    @property
    def _execution_device(self):
        if self._offload_device is not None:
            return torch.device(self._offload_device)
        return self._mp._execution_device

    # -- placement ---------------------------------------------------------------------
    def to(self, *args, **kwargs):
        self.remove_all_hooks()
        for m in self._modules():
            m.to(*args, **kwargs)
        return self

    def remove_all_hooks(self):
        try:
            from accelerate.hooks import remove_hook_from_module
            for m in self._modules():
                remove_hook_from_module(m, recurse=True)
        except Exception:
            pass
        self._all_hooks = []
        self._offload_device = None

    def enable_model_cpu_offload(self, gpu_id=None, device="cuda", **_ignored):
        """Text encoder, transformer and VAE take the card in turn; the others wait in RAM."""
        from accelerate import cpu_offload_with_hook
        dev = torch.device(device if gpu_id is None else f"{device}:{gpu_id}")
        self.remove_all_hooks()
        hooks, prev = [], None
        for m in self._modules():
            m.to("cpu")
            _, prev = cpu_offload_with_hook(m, dev, prev_module_hook=prev)
            hooks.append(prev)
        self._all_hooks = hooks
        self._offload_device = dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def enable_sequential_cpu_offload(self, gpu_id=None, device="cuda", **_ignored):
        self._say_once("seq", "sequential offload is not available with GGUF weights: using the "
                              "model offload (one model on the card at a time) instead")
        self.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def maybe_free_model_hooks(self):
        for h in self._all_hooks or []:
            try:
                h.offload()
            except Exception:
                pass

    def release(self):
        """Drops what the adapter holds on the card (free_vram)."""
        self.remove_all_hooks()
        for m in self._modules():
            try:
                m.to("cpu")
            except Exception:
                pass
        self._embeds.clear()
        self._last_x0 = None
        self._ctx = None
        self.released = True

    # -- not on this model ---------------------------------------------------------------
    def set_loras(self, wanted):
        if wanted:
            raise RuntimeError("LoRAs are not supported on MiniMax-H3-Image yet")
        self.applied_loras = []

    def edit_loras_applied(self):
        return []

    def load_lora_weights(self, *a, **k):
        raise RuntimeError("LoRAs are not supported on MiniMax-H3-Image yet")

    def magic_prompt(self, *a, **k):
        raise RuntimeError("the local magic prompt is not available on this engine: its text "
                           "encoder ships without a language head. Use Ollama.")

    def describe_image(self, *a, **k):
        raise RuntimeError("the local Describe is not available on this engine: its text "
                           "encoder ships without a language head. Use Ollama.")

    # -- text --------------------------------------------------------------------------
    @staticmethod
    def _fingerprint(image):
        import hashlib
        small = image.convert("RGB").resize((48, 48))
        return (image.size, hashlib.sha1(small.tobytes()).hexdigest())

    @torch.no_grad()
    def _conditioning(self, prompt, refs):
        """(prompt_embeds, text_token_tags) of a prompt and its references, remembered: the
        same prompt again (a batch, the detailer's passes) does not bring the 18 GB encoder
        back onto the card. The encoder leaves the card before returning."""
        prompt = prompt if (isinstance(prompt, str) and prompt.strip()) else " "
        key = (prompt, tuple(self._fingerprint(r) for r in refs), id(self._text_pipe.text_encoder))
        hit = self._embeds.get(key)
        if hit is None:
            t0 = time.time()
            kw = {"prompt": prompt}
            if refs:
                kw["image"] = list(refs)
            out = self._text_pipe(output=["prompt_embeds", "text_token_tags"], **kw)
            hit = (out["prompt_embeds"].detach().to("cpu"), out["text_token_tags"].detach().to("cpu"))
            if len(self._embeds) >= 8:
                self._embeds.pop(next(iter(self._embeds)))
            self._embeds[key] = hit
            if self._all_hooks:
                self._all_hooks[0].offload()        # the encoder goes back to RAM now
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _log(f"text: {hit[0].shape[1]} tokens encoded in {time.time() - t0:.1f}s")
        return hit

    # -- latents -----------------------------------------------------------------------
    def _norm_stats(self, device):
        cfg = self._mp.vae.config
        mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1)
        return mean, std

    @torch.no_grad()
    def _encode_image(self, image, width, height):
        """Normalized latents (1, 24, h/16, w/16) of `image` at width x height: the inverse
        of the pipeline's decode (ImageNet-normalized pixels, posterior mode)."""
        import numpy as np
        from PIL import Image
        dev = self._execution_device
        vae = self._mp.vae
        img = image.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        px = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        pm = torch.tensor(self._mp.pixel_mean, device=dev).view(1, -1, 1, 1)
        ps = torch.tensor(self._mp.pixel_std, device=dev).view(1, -1, 1, 1)
        px = ((px - pm) / ps).to(vae.dtype)
        post = vae.encode(px, return_dict=False)[0]
        lat = post.mode().float() if hasattr(post, "mode") else post.float()
        mean, std = self._norm_stats(lat.device)
        return (lat - mean) / std

    @torch.no_grad()
    def _decode(self, latents, width, height, output_type="pil"):
        dev = self._execution_device
        vae = self._mp.vae
        mean, std = self._norm_stats(dev)
        lat = latents.to(dev).float() * std + mean
        img = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
        pm = torch.tensor(self._mp.pixel_mean, device=dev).view(1, -1, 1, 1)
        ps = torch.tensor(self._mp.pixel_std, device=dev).view(1, -1, 1, 1)
        img = (img.float() * ps + pm).clamp(0, 1)
        return self._mp.image_processor.postprocess(img, output_type=output_type)

    def _rows(self, latents):
        """(1, 24, h, w) -> the packed rows the loop works on."""
        from diffusers.modular_pipelines.minimax_h3_image.before_denoise import patchify_video_latents
        return patchify_video_latents(latents[:, :, None], self._mp.patch_size)

    def _unrows(self, rows, lat_h, lat_w):
        """Packed rows of the generated image -> (1, 24, h, w)."""
        _, ph, pw = self._mp.patch_size
        c = self._mp.vae_latent_channels
        x = rows.reshape(-1, lat_h // ph, lat_w // pw, c, ph, pw).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(-1, c, lat_h, lat_w)

    @staticmethod
    def _mask_rows(mask_image, width, height, device):
        """(rows, 1): 1 = repaint, 0 = keep. A 32 px token is repainted as soon as ANY of
        its pixels is (max-pool), so a thin stroke never vanishes below the grid."""
        import numpy as np
        from PIL import Image
        m = mask_image.convert("L")
        if m.size != (width, height):
            m = m.resize((width, height), Image.NEAREST)
        t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None]
        t = torch.nn.functional.max_pool2d(t, kernel_size=GRID, stride=GRID)
        return (t.reshape(-1, 1) > 0.5).float().to(device)

    @staticmethod
    def _is_full_mask(mask_image):
        try:
            return mask_image.convert("L").getextrema()[0] >= 250
        except Exception:
            return False

    # -- the step hook -----------------------------------------------------------------
    def _on_step(self, block_state, i, t):
        ctx = self._ctx or {}
        k = self._start + i                               # index on the full grid
        nxt = PDD_SIGMAS[min(k + 1, PDD_STEPS)]
        nref = int(getattr(block_state, "num_reference_rows", 0) or 0)
        lat = block_state.latents
        if ctx.get("mask") is not None:
            known = (1.0 - nxt) * ctx["x0_rows"] + nxt * ctx["noise_rows"]
            m = ctx["mask"]
            lat[nref:] = m * lat[nref:] + (1.0 - m) * known.to(lat.dtype)
            block_state.latents = lat
        # What the image would be if the run stopped here: x0 = x + sigma * velocity.
        x0 = lat[nref:].float()
        try:
            x0 = x0 + nxt * block_state.noise_pred[0, nref:].float()
        except Exception:
            pass
        x0 = self._unrows(x0, ctx.get("lat_h"), ctx.get("lat_w"))
        self._last_x0 = x0
        cb = ctx.get("callback")
        if cb is not None:
            cb(self, i, t, {"latents": x0})
        if self._interrupt:
            _quiet_interrupt(True)
            raise _Interrupted()

    # -- the call ----------------------------------------------------------------------
    def __call__(self, prompt=None, image=None, mask_image=None, strength=None,
                 width=None, height=None, num_inference_steps=None, generator=None,
                 guidance_scale=None, num_images_per_prompt=1, negative_prompt=None,
                 prompt_embeds=None, callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=None, output_type="pil",
                 return_dict=True, reference_init=None, **_ignored):
        import types
        self._interrupt = False
        if isinstance(prompt, (list, tuple)):
            imgs = []
            for j, p in enumerate(prompt):
                g = generator[j] if isinstance(generator, (list, tuple)) else generator
                imgs += self(prompt=p, image=image, mask_image=mask_image, strength=strength,
                             width=width, height=height, generator=g,
                             callback_on_step_end=callback_on_step_end,
                             output_type=output_type).images
            return types.SimpleNamespace(images=imgs)

        if num_inference_steps not in (None, PDD_STEPS, 0):
            self._say_once(("steps", int(num_inference_steps)),
                           f"steps {int(num_inference_steps)} ignored: MiniMax-H3-Image is distilled "
                           f"to exactly {PDD_STEPS} steps (one output head per step)")
        if guidance_scale is not None and abs(float(guidance_scale) - 1.0) > 1e-6:
            self._say_once(("cfg", float(guidance_scale)),
                           f"guidance {float(guidance_scale):g} ignored: MiniMax-H3-Image has no CFG "
                           f"(the guidance is baked into the weights)")
        if isinstance(negative_prompt, str) and negative_prompt.strip():
            self._say_once("neg", "negative prompt ignored: MiniMax-H3-Image has none. Say what "
                                  "you want in the prompt instead.")

        imgs_in = image if isinstance(image, (list, tuple)) else ([image] if image is not None else [])
        imgs_in = [r for r in imgs_in if r is not None]
        emulated = strength is not None or mask_image is not None
        refs = []
        init = None
        if emulated:
            if not imgs_in:
                raise ValueError("img2img / inpaint need an image to start from")
            init = imgs_in[0]
            width, height = width or init.size[0], height or init.size[1]
            # Inpaint: the model also SEES the whole source as <Picture 1> (its native
            # editing input), so what it repaints agrees with what is around the mask.
            # Without it the masked area is drawn from the prompt alone.
            if reference_init is None:
                reference_init = mask_image is not None and INPAINT_WITH_REFERENCE
            if reference_init and mask_image is not None and not self._is_full_mask(mask_image):
                refs = [init]
        elif imgs_in:
            refs = list(imgs_in)
            if len(refs) > EDIT_MAX_REFS:
                self._say_once("refs", f"{len(refs)} references given, the model takes "
                                       f"{EDIT_MAX_REFS}: the first {EDIT_MAX_REFS} are used")
                refs = refs[:EDIT_MAX_REFS]
            width, height = width or refs[0].size[0], height or refs[0].size[1]
        w, h = fit_size(width, height)
        if (width and height) and (w, h) != (int(width), int(height)):
            self._say_once(("size", int(width), int(height)),
                           f"size {width}x{height} -> {w}x{h} (MiniMax-H3-Image: multiples of "
                           f"{GRID}, 20 Mpx at most)")
        if w * h < SMALL_PIXELS:
            self._say_once(("small", w, h),
                           f"{w}x{h} is under 1 Mpx: MiniMax-H3-Image is made for 4 to 16 Mpx, "
                           f"quality drops at small sizes")

        # Same seed, same noise on every machine: the release draws its noise on the CPU.
        gen = generator[0] if isinstance(generator, (list, tuple)) else generator
        if gen is not None and getattr(gen, "device", None) is not None and gen.device.type != "cpu":
            gen = torch.Generator("cpu").manual_seed(int(gen.initial_seed()))

        # 1. The text first; the encoder has left the card when this returns.
        embeds, tags = self._conditioning(prompt, refs)
        dev = self._execution_device
        kw = {"prompt_embeds": embeds.to(dev), "text_token_tags": tags,
              "width": w, "height": h, "generator": gen, "output_type": output_type,
              "min_generation_area": 1024}
        if refs:
            kw["image"] = refs
        ctx = {"callback": callback_on_step_end, "lat_h": h // 16, "lat_w": w // 16}

        # 2. img2img / inpaint: start later on the grid, from the noised image.
        start = 0
        if emulated:
            start = strength_to_start(strength)
            sigma0 = PDD_SIGMAS[start]
            x0 = self._encode_image(init, w, h)
            noise = torch.randn(x0.shape, generator=gen, dtype=torch.float32, device="cpu").to(x0.device)
            kw["latents"] = (1.0 - sigma0) * x0 + sigma0 * noise
            kind = "img2img"
            if mask_image is not None and not self._is_full_mask(mask_image):
                kind = "inpaint"
                ctx.update(mask=self._mask_rows(mask_image, w, h, x0.device),
                           x0_rows=self._rows(x0), noise_rows=self._rows(noise))
            asked = 1.0 if strength is None else float(strength)
            self._say_once((kind, round(asked, 2)),
                           f"{kind} (emulated): strength {asked:.2f} -> sigma {sigma0:.2f}, "
                           f"{PDD_STEPS - start} of {PDD_STEPS} steps (the grid is fixed)")
        self._start = start
        self._mp.transformer._cz_pdd_offset = start
        self._cz_run_steps = PDD_STEPS - start
        self._ctx = ctx
        self._last_x0 = None
        try:
            images = self._mp(output="images", **kw)
        except _Interrupted:
            images = [] if self._last_x0 is None else self._decode(self._last_x0, w, h, output_type)
        finally:
            _quiet_interrupt(False)
            self._ctx = None
            self._start = 0
            self._mp.transformer._cz_pdd_offset = 0
            self._cz_run_steps = None
            self.maybe_free_model_hooks()
        if not isinstance(images, list):
            images = list(images) if output_type == "pil" else images
        return types.SimpleNamespace(images=images) if return_dict else (images,)


EnginePipe = MiniMaxPipe


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_vae(repo=OFFICIAL_REPO, dtype=torch.bfloat16, token=None):
    from diffusers import AutoencoderKLMiniMaxH3Image
    return AutoencoderKLMiniMaxH3Image.from_pretrained(
        repo, subfolder="vae_diffusers", torch_dtype=dtype, token=token).eval()


def load(repo=DEFAULT_REPO, dtype=torch.bfloat16, precision=None, transformer=None,
         text_encoder=None, token=None):
    """A MiniMaxPipe on the CPU. `transformer` / `text_encoder`: loaded modules, file
    paths, or None for the default files (found in ./checkpoints, else downloaded)."""
    from diffusers import MiniMaxH3Scheduler
    from transformers import AutoTokenizer, Qwen3VLProcessor

    main_blocks, text_blocks, crispz = build_blocks()
    mp = main_blocks.init_pipeline(repo)
    text_pipe = text_blocks.init_pipeline()
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", token=token)
    processor = Qwen3VLProcessor.from_pretrained(repo, subfolder="processor", token=token)
    scheduler = MiniMaxH3Scheduler.from_pretrained(repo, subfolder="scheduler", token=token)
    if not isinstance(text_encoder, nn.Module):
        text_encoder = load_text_encoder(text_encoder or default_file("text_encoder", token),
                                         dtype, repo=repo)
    if not isinstance(transformer, nn.Module):
        transformer = load_transformer(transformer or default_file("transformer", token), dtype)
    vae = load_vae(repo, dtype, token)
    mp.update_components(scheduler=scheduler, transformer=transformer, vae=vae)
    text_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer, processor=processor)
    return MiniMaxPipe(mp, text_pipe, crispz, repo=repo)

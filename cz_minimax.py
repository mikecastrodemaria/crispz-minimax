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


def load_transformer(path, compute_dtype=torch.bfloat16):
    t0 = time.time()
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


def load_text_encoder(path, compute_dtype=torch.bfloat16, repo=OFFICIAL_REPO):
    """Qwen3VLForConditionalGeneration with 50 decoder layers and no final norm, so that
    `hidden_states[50]` is the output of the 50th layer, the state the DiT reads. The
    language-model head is dropped (never used)."""
    from accelerate import init_empty_weights
    from transformers import Qwen3VLForConditionalGeneration

    t0 = time.time()
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
# The pipeline
# --------------------------------------------------------------------------------------
def load_vae(repo=OFFICIAL_REPO, dtype=torch.bfloat16):
    from diffusers import AutoencoderKLMiniMaxH3Image
    return AutoencoderKLMiniMaxH3Image.from_pretrained(repo, subfolder="vae_diffusers", torch_dtype=dtype).eval()


def load_pipeline(transformer_path, text_encoder_path, device="cuda", repo=OFFICIAL_REPO,
                  compute_dtype=torch.bfloat16, vae_dtype=torch.bfloat16, offload=True):
    """The official block chain with the GGUF components. With `offload`, the text encoder,
    the transformer and the VAE take the GPU in turn (accelerate hooks: a model goes back
    to RAM when the next one is called)."""
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3_image import MiniMaxH3ImageBlocks
    from transformers import AutoTokenizer, Qwen3VLProcessor

    blocks = MiniMaxH3ImageBlocks()
    mp = blocks.init_pipeline(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    processor = Qwen3VLProcessor.from_pretrained(repo, subfolder="processor")
    scheduler = MiniMaxH3Scheduler.from_pretrained(repo, subfolder="scheduler")
    text_encoder = load_text_encoder(text_encoder_path, compute_dtype, repo)
    transformer = load_transformer(transformer_path, compute_dtype)
    vae = load_vae(repo, vae_dtype)
    mp.update_components(text_encoder=text_encoder, tokenizer=tokenizer, processor=processor,
                         scheduler=scheduler, transformer=transformer, vae=vae)
    hooks = []
    if offload:
        from accelerate import cpu_offload_with_hook
        hook = None
        for model in (text_encoder, transformer, vae):
            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            hooks.append(hook)
    else:
        for model in (text_encoder, transformer, vae):
            model.to(device)
    return mp, hooks


def offload_all(hooks):
    for h in hooks or []:
        try:
            h.offload()
        except Exception:
            pass

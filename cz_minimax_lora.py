"""crispz-minimax - the MiniMax-H3-Image LoRA loader.

diffusers 0.43.0 has no LoRA mixin for MiniMaxH3ImageTransformer2DModel, and the DiT of
this app is the pruned one, its big matrices GGUF-quantized on the card: nothing can be
merged into them. LoRAs are applied as FORWARD HOOKS on the DiT's Linear modules (the
technique of crispz-moss): the base weights stay as they are, the weight slider re-scales
live, unloading removes the hooks. The LoRA tensors are module buffers, so they follow the
DiT between the card and the RAM.

Dialects read (headers read on 2026-11-22):
  comfy     diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight / .lora_B.weight
            (ai-toolkit; ComfyUI's module names, the names of the single files)
  kohya     lora_unet_blocks_N_attn_qkv_proj.lora_down.weight / .lora_up.weight / .alpha
  diffusers transformer.transformer_blocks.N.attn.to_q.lora_A.weight / .lora_B.weight,
            q, k and v SEPARATE, the feed-forward in diffusers' [value; gate] order

The DiT built by cz_minimax keeps the checkpoint's layout: one fused `attn.to_qkv`
([q; k; v] rows) and a `[gate; value]` feed-forward. So:
  - a comfy / kohya `qkv_proj` and `mlp.fc1` LoRA applies as it is;
  - a diffusers `to_q` / `to_k` / `to_v` LoRA applies on its third of the fused output
    (a row slice, no zero-padded matrix);
  - a diffusers `ff.net.0.proj` LoRA has the two halves of its up matrix swapped.
Keys that match no module are COUNTED AND SAID, never dropped in silence (the audio
stream of an H3 video LoRA, for instance). DiffSynth files (`.lora_A.default.weight`,
q / k / v interleaved per head) are refused by name: their layout is not handled.
"""
import json
import os
import re

import torch

PREFIXES = ("diffusion_model.", "transformer.", "base_model.model.", "model.")
LOW_RANK = ((".lora_down.weight", ".lora_up.weight"),
            (".lora_A.weight", ".lora_B.weight"),
            (".lora.down.weight", ".lora.up.weight"))
# ComfyUI module name -> the module of the DiT built by cz_minimax.build_transformer
_COMFY_RENAMES = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    (".attn.out_proj", ".attn.to_out.0"),
    (".attn.qkv_proj", ".attn.to_qkv"),
    (".mlp.fc1", ".ff.net.0.proj"),
    (".mlp.fc2", ".ff.net.2"),
)
_KOHYA_RE = re.compile(r"^lora_unet_((?:token_refiner_)?blocks)_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)$")
_KOHYA_LEAF = {"attn_qkv_proj": "attn.qkv_proj", "attn_out_proj": "attn.out_proj",
               "mlp_fc1": "mlp.fc1", "mlp_fc2": "mlp.fc2"}
_SPLIT_QKV_RE = re.compile(r"^(.*\.attn)\.to_([qkv])$")
_DIFFUSERS_FF_RE = re.compile(r"^(?:transformer_blocks|token_refiner\.refiner_blocks)\.\d+\.ff\.net\.0\.proj$")


def _log(msg):
    try:
        from cz_core import _log as core_log
        core_log(msg)
    except Exception:
        print(f"[crispz] {msg}")


def strip_prefix(key):
    for p in PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _module_of(name):
    """(module of this app's DiT, dialect) for a LoRA module name without its suffix."""
    m = _KOHYA_RE.match(name)
    if m:
        root = "token_refiner.blocks" if m.group(1).startswith("token") else "blocks"
        name, dialect = f"{root}.{m.group(2)}.{_KOHYA_LEAF[m.group(3)]}", "kohya"
    else:
        name = strip_prefix(name)
        dialect = "comfy"
    if name.startswith("blocks.") or name.startswith("token_refiner.blocks."):
        if name.startswith("blocks."):
            name = "transformer_blocks." + name[len("blocks."):]
        for a, b in _COMFY_RENAMES:
            name = name.replace(a, b)
        return name, dialect
    return name, "diffusers"


def classify(key):
    """(kind, module, dialect): kind in down / up / alpha / diffsynth / full / unknown."""
    if ".lora_A.default." in key or ".lora_B.default." in key:
        return "diffsynth", key, "diffsynth"
    for d, u in LOW_RANK:
        if key.endswith(d):
            return ("down",) + _module_of(key[:-len(d)])
        if key.endswith(u):
            return ("up",) + _module_of(key[:-len(u)])
    if key.endswith(".alpha"):
        return ("alpha",) + _module_of(key[:-len(".alpha")])
    if key.endswith(".weight") or key.endswith(".bias"):
        return "full", strip_prefix(key).rsplit(".", 1)[0], None
    return "unknown", key, None


def _global_alpha(meta):
    """An alpha that holds for the whole file, from its metadata (peft files carry no
    `.alpha` tensor): `alpha`, or diffusers' `lora_adapter_metadata` JSON."""
    meta = meta or {}
    for key in ("alpha", "lora_alpha", "ss_network_alpha"):
        try:
            v = meta.get(key)
            if v not in (None, ""):
                return float(v)
        except (TypeError, ValueError):
            pass
    for key in ("lora_adapter_metadata", "transformer.lora_adapter_metadata"):
        raw = meta.get(key)
        if not raw:
            continue
        try:
            cfg = json.loads(raw)
        except Exception:
            continue
        for k in ("lora_alpha", "transformer.lora_alpha"):
            if isinstance(cfg, dict) and isinstance(cfg.get(k), (int, float)):
                return float(cfg[k])
    return None


class LoraFile:
    """What a LoRA file holds, in the DiT's own module names.
    items: module -> list of {"down", "up", "alpha", "rows": (start, stop) or None}."""

    def __init__(self, path):
        self.path = path
        self.items = {}
        self.unknown = []
        self.full = []
        self.diffsynth = 0
        self.meta = {}
        self.dialects = set()
        self.split = 0
        self.swapped = 0


def read_header(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys()), (f.metadata() or {})


def inspect_file(path):
    """The file's layout from its header alone (no tensor read)."""
    keys, meta = read_header(path)
    lf = LoraFile(path)
    lf.meta = meta
    for k in keys:
        kind, module, dialect = classify(k)
        if kind == "unknown":
            lf.unknown.append(k)
        elif kind == "diffsynth":
            lf.diffsynth += 1
        elif kind == "full":
            lf.full.append(module)
        else:
            lf.dialects.add(dialect)
            lf.items.setdefault(module, {})[kind] = True
    return lf


def refusal(path):
    """Why this MiniMax-family file cannot be applied, or None."""
    try:
        lf = inspect_file(path)
    except Exception as e:
        return f"unreadable LoRA ({type(e).__name__}: {e})"
    if lf.diffsynth:
        return ("DiffSynth layout (`.lora_A.default.weight`, q / k / v interleaved per "
                "head): not handled by this app. Take the ComfyUI or diffusers file of "
                "the same LoRA")
    if lf.full:
        what = ", ".join(sorted(set(lf.full))[:3])
        return f"replaces whole layers ({what}): not a LoRA"
    if not any("down" in it and "up" in it for it in lf.items.values()):
        return "no LoRA tensors found (lora_A / lora_B or lora_down / lora_up)"
    return None


def read_file(path):
    """The tensors of a LoRA file, grouped per DiT module of this app: the separate
    q / k / v of the diffusers dialect placed on their third of the fused qkv, the
    feed-forward up matrix of that dialect reordered to [gate; value]."""
    from safetensors.torch import load_file
    sd = load_file(path)
    _keys, meta = read_header(path)
    lf = LoraFile(path)
    lf.meta = meta
    raw, alphas = {}, {}
    for k, t in sd.items():
        kind, module, dialect = classify(k)
        if kind == "unknown":
            lf.unknown.append(k)
        elif kind == "diffsynth":
            lf.diffsynth += 1
        elif kind == "full":
            lf.full.append(module)
        elif kind == "alpha":
            alphas[module] = float(t.float().item()) if t.numel() == 1 else None
        else:
            lf.dialects.add(dialect)
            it = raw.setdefault(module, {})
            it[kind] = t
            it["dialect"] = dialect
    g_alpha = _global_alpha(meta)
    for module, it in raw.items():
        if "down" not in it or "up" not in it:
            lf.unknown.append(module)
            continue
        a = alphas.get(module)
        it["alpha"] = g_alpha if a is None else a
        it["rows"] = None
        m = _SPLIT_QKV_RE.match(module)
        if m:
            rows = it["up"].shape[0]
            i = "qkv".index(m.group(2))
            it["rows"] = (i * rows, (i + 1) * rows)
            module = m.group(1) + ".to_qkv"
            lf.split += 1
        elif it["dialect"] == "diffusers" and _DIFFUSERS_FF_RE.match(module):
            half = it["up"].shape[0] // 2
            it["up"] = torch.cat([it["up"][half:], it["up"][:half]], dim=0).contiguous()
            lf.swapped += 1
        lf.items.setdefault(module, []).append(it)
    return lf


# --- Application ----------------------------------------------------------------------
def _hook(mod, args, out):
    # The tensors are read from the module's buffers BY NAME at every call: a
    # module.to(device) replaces the buffers with copies (the DiT parked in RAM while the
    # text encoder works, then back on the card).
    x = args[0]
    for dn, un, s, rows in getattr(mod, "_cz_lr", ()):
        if s == 0:
            continue
        d, u = mod._buffers[dn], mod._buffers[un]
        delta = ((x.to(d.dtype) @ d.t()) @ u.t()).to(out.dtype) * s
        if rows is None:
            out = out + delta
        else:
            out = out.clone()
            out[..., rows[0]:rows[1]] += delta
    return out


def unload(transformer):
    """Removes every LoRA hook and buffer. Returns how many modules were touched."""
    n = 0
    for mod in transformer.modules():
        h = getattr(mod, "_cz_lora_hook", None)
        if h is not None:
            h.remove()
            del mod._cz_lora_hook
            n += 1
        for attr in ("_cz_lr", "_cz_scales"):
            if hasattr(mod, attr):
                delattr(mod, attr)
        for name in [b for b in list(mod._buffers) if b.startswith("_cz_lora_")]:
            del mod._buffers[name]
    return n


def _buf(mod, name, t, keep_dtype=False):
    """Registers `t` as a buffer of `mod`, on the module's device, in the DiT's compute
    dtype (a GGUF weight is a byte matrix: its own dtype says nothing) or, with
    keep_dtype, in the tensor's own dtype (the matmul then runs in it and is cast)."""
    ref = getattr(mod, "weight", None)
    dev = ref.device if ref is not None else torch.device("cpu")
    if keep_dtype and t.is_floating_point():
        dt = t.dtype
    else:
        dt = getattr(mod, "compute_dtype", None)
        if dt is None:
            dt = ref.dtype if ref is not None and ref.is_floating_point() else torch.bfloat16
    mod.register_buffer(name, t.detach().to(dev, dt), persistent=False)
    return name


def _out_features(mod):
    return int(getattr(mod, "out_features", 0) or 0)


def apply(transformer, loras, fp32=()):
    """Applies [(path, weight), ...] on the DiT (the previous set removed first).
    Returns one summary per file. Raises on an unreadable or refused file."""
    unload(transformer)
    reports = []
    fp32 = {os.path.abspath(str(p)) for p in (fp32 or ())}
    for li, (path, weight) in enumerate(loras):
        keep = os.path.abspath(str(path)) in fp32
        why = refusal(path)
        if why:
            raise ValueError(why)
        lf = read_file(path)
        rep = {"path": path, "weight": float(weight), "modules": 0, "unknown": list(lf.unknown),
               "split": lf.split, "swapped": lf.swapped, "missing": [], "shape": []}
        for module, entries in lf.items.items():
            try:
                mod = transformer.get_submodule(module)
            except AttributeError:
                mod = None
            if mod is None or not isinstance(mod, torch.nn.Linear):
                rep["missing"].append(module)
                continue
            touched = False
            for it in entries:
                rows = it["rows"]
                want_out = (rows[1] - rows[0]) if rows else _out_features(mod)
                if it["down"].shape[1] != mod.in_features or it["up"].shape[0] != want_out \
                        or (rows and rows[1] > _out_features(mod)):
                    rep["shape"].append(module)
                    continue
                r = it["down"].shape[0]
                base = (float(it["alpha"]) / r) if it.get("alpha") else 1.0
                idx = len(getattr(mod, "_cz_lr", ()))
                dn = _buf(mod, f"_cz_lora_{li}_{idx}_down", it["down"], keep)
                un = _buf(mod, f"_cz_lora_{li}_{idx}_up", it["up"], keep)
                mod._cz_lr = list(getattr(mod, "_cz_lr", ())) + [(dn, un, base * float(weight), rows)]
                mod._cz_scales = list(getattr(mod, "_cz_scales", ())) + [(idx, li, base)]
                touched = True
            if touched:
                rep["modules"] += 1
                if getattr(mod, "_cz_lora_hook", None) is None:
                    mod._cz_lora_hook = mod.register_forward_hook(_hook)
        reports.append(rep)
        name = os.path.basename(path)
        msg = (f"LoRA {name} @ {float(weight):g}: {rep['modules']} module(s)"
               + (f", {rep['split']} separate q / k / v placed on the fused qkv" if rep["split"] else "")
               + (f", {rep['swapped']} feed-forward up matrices reordered" if rep["swapped"] else "")
               + (", tensors kept in their own dtype (fp32)" if keep else ""))
        skipped = rep["unknown"] + rep["missing"] + rep["shape"]
        if skipped:
            msg += (f"; NOT applied: {len(rep['unknown'])} unknown key(s), "
                    f"{len(rep['missing'])} module(s) not in this DiT, "
                    f"{len(rep['shape'])} of another size (first: {skipped[0]})")
        _log(msg)
        if rep["modules"] == 0:
            _log(f"LoRA {name}: nothing applied, the render will look exactly as without it")
    return reports


def rescale(transformer, weights):
    """New weights for the SAME files, in order: the scales change, nothing reloads."""
    for mod in transformer.modules():
        scales = getattr(mod, "_cz_scales", None)
        if not scales:
            continue
        for idx, li, base in scales:
            w = float(weights[li]) if li < len(weights) else 0.0
            d, u, _s, rows = mod._cz_lr[idx]
            mod._cz_lr[idx] = (d, u, base * w, rows)
    return True


def applied_modules(transformer):
    return sum(1 for m in transformer.modules() if getattr(m, "_cz_lora_hook", None) is not None)


# --- Family, from the tensor names (and widths when the header gives them) -------------
HIDDEN = 5376


def is_h3_lora_keys(keys, shapes=None):
    """True when the tensor names are those of a MiniMax-H3 (image or video) LoRA.
    comfy and kohya names are the model's own; the diffusers names (`transformer_blocks.N
    .attn.to_q`) exist in other families, so they need the token refiner beside them or,
    when `shapes` (name -> shape) is given, the model's width of 5376."""
    keys = [k for k in (keys or ()) if isinstance(k, str)]
    if not any(".lora_" in k or ".lora." in k for k in keys):
        return False
    for k in keys:
        if k.startswith("lora_unet_blocks_") and ("_attn_qkv_proj." in k or "_mlp_fc1." in k):
            return True
        s = strip_prefix(k)
        if s.startswith("blocks.") and (".attn.qkv_proj." in s or ".mlp.fc1." in s):
            return True
    dif = [k for k in keys if strip_prefix(k).startswith("transformer_blocks.")
           and ".attn.to_q.lora_A" in k]
    if not dif:
        return False
    if any(strip_prefix(k).startswith("token_refiner.refiner_blocks.") for k in keys):
        return True
    if shapes:
        shp = shapes.get(dif[0])
        return bool(shp) and int(shp[-1]) == HIDDEN
    return False

"""First render of the MiniMax-H3-Image engine: loads, generates, times, reports peaks.

    python tools/minimax_smoke.py [--size 2048x2048] [--seed 1234] [--out out/smoke.png]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import cz_minimax as M

p = argparse.ArgumentParser()
p.add_argument("--dit", default="checkpoints/minimax/minimax_h3_image_pruned-UD-Q5_K_XL.gguf")
p.add_argument("--te", default="checkpoints/minimax/qwen3vl_32b_minimax_h3-Q4_K_M.gguf")
p.add_argument("--size", default="2048x2048")
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--prompt", default=None)
p.add_argument("--image", action="append")
p.add_argument("--out", default="out/smoke.png")
p.add_argument("--strength", type=float, default=None)
p.add_argument("--mask", default=None)
p.add_argument("--init", default=None)
p.add_argument("--repeat", type=int, default=1)
a = p.parse_args()

prompt = a.prompt or ("A weathered lighthouse keeper in a yellow oilcloth coat stands on a wet stone pier at dusk, "
    "holding a brass lantern whose warm light falls across his lined face and grey beard. Behind him a white "
    "lighthouse with a red band rises against a stormy violet sky, its beam cutting through drifting sea spray. "
    "Waves break against black rocks on the left, and a small fishing boat with a green hull is tied to an iron "
    "ring in the foreground. A hand-painted wooden sign on the pier reads \"HARBOUR 7\" in white letters. "
    "Cinematic photograph, shallow depth of field, rich detail in the fabric, the stone and the water.")
if os.path.isfile(prompt):
    prompt = open(prompt, encoding="utf-8").read()
w, h = (int(x) for x in a.size.lower().split("x"))

t0 = time.time()
pipe = M.load(transformer=a.dit, text_encoder=a.te)
pipe.enable_model_cpu_offload()
print(f"load total {time.time()-t0:.1f}s", flush=True)

from PIL import Image
kw = dict(prompt=prompt, height=h, width=w, generator=torch.Generator("cuda").manual_seed(a.seed))
if a.image:
    kw["image"] = [Image.open(x).convert("RGB") for x in a.image]
if a.init:
    kw["image"] = Image.open(a.init).convert("RGB")
    kw["strength"] = 1.0 if a.strength is None else a.strength
    if a.mask:
        kw["mask_image"] = Image.open(a.mask).convert("L")
seen = []
def cb(p, i, t, d):
    seen.append(tuple(d["latents"].shape))
for r in range(a.repeat):
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        images = pipe(callback_on_step_end=cb, **kw).images
    dt = time.time() - t1
    print(f"render {w}x{h} #{r+1}: {dt:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
          f"{len(seen)} callbacks {seen[-1] if seen else None}", flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
images[0].save(a.out)
print("saved", a.out, images[0].size)

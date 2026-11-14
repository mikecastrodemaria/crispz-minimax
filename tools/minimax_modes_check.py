"""img2img strengths, inpaint and edit with references on the real engine, one load.

    python tools/minimax_modes_check.py [--src out/adapter.png] [--out out/modes]
Writes one image per case and a contact sheet; prints time and peak VRAM per case.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image, ImageDraw
import cz_minimax as M

p = argparse.ArgumentParser()
p.add_argument("--src", default="out/adapter.png")
p.add_argument("--out", default="out/modes")
p.add_argument("--cases", default="i2i,inpaint,edit,small")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
src = Image.open(a.src).convert("RGB")
W, H = src.size
BASE = ("A weathered lighthouse keeper in a yellow oilcloth coat stands on a wet stone pier at dusk, holding a "
        "brass lantern. Behind him a white lighthouse with a red band under a stormy violet sky. Cinematic photograph.")
pipe = M.load()
pipe.enable_model_cpu_offload()
rows = []

def run(name, **kw):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    img = pipe(generator=torch.Generator("cpu").manual_seed(7), **kw).images[0]
    print(f"{name}: {time.time()-t:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, {img.size}", flush=True)
    img.save(os.path.join(a.out, name + ".png"))
    rows.append((name, img))

cases = a.cases.split(",")
if "i2i" in cases:
    for s in (0.25, 0.5, 0.75, 0.9):
        run(f"i2i_{int(s*100):03d}", prompt=BASE.replace("yellow", "bright red"), image=src, strength=s, width=W, height=H)
if "inpaint" in cases:
    m = Image.new("L", src.size, 0)
    ImageDraw.Draw(m).rectangle((int(W*.60), int(H*.16), int(W*.86), int(H*.34)), fill=255)
    m.save(os.path.join(a.out, "mask.png"))
    for s in (0.75, 1.0):
        run(f"inpaint_{int(s*100):03d}", prompt=BASE.replace("stands", "wearing a navy blue knitted beanie stands"),
            image=src, mask_image=m, strength=s, width=W, height=H)
if "inpaintref" in cases:
    m = Image.new("L", src.size, 0)
    ImageDraw.Draw(m).rectangle((int(W*.60), int(H*.16), int(W*.86), int(H*.34)), fill=255)
    P1 = BASE.replace("stands", "wearing a navy blue knitted beanie stands")
    P2 = "Edit <Picture 1>: the man now wears a navy blue knitted beanie instead of his hat. Everything else is unchanged."
    run("inp_ref_plain_100", prompt=P1, image=src, mask_image=m, strength=1.0, width=W, height=H)
    run("inp_ref_edit_100", prompt=P2, image=src, mask_image=m, strength=1.0, width=W, height=H)
    run("inp_ref_edit_075", prompt=P2, image=src, mask_image=m, strength=0.75, width=W, height=H)
    run("inp_noref_edit_100", prompt=P2, image=src, mask_image=m, strength=1.0, width=W, height=H, reference_init=False)
if "edit" in cases:
    run("edit_1ref", prompt="Edit <Picture 1>: replace the yellow oilcloth coat with a bright red one and make it night "
        "with a starry sky. Keep the man's face, the lantern, the lighthouse and the composition unchanged.",
        image=[src], width=W, height=H)
if "small" in cases:
    run("t2i_1024", prompt=BASE, width=1024, height=1024)
    run("t2i_1536x1024", prompt=BASE, width=1536, height=1024)
th = 512
sheet = Image.new("RGB", (th * (len(rows) + 1), th + 28), "white")
sheet.paste(src.resize((th, th)), (0, 28)); d = ImageDraw.Draw(sheet); d.text((4, 6), "source", fill="black")
for i, (n, im) in enumerate(rows):
    t = im.copy(); t.thumbnail((th, th)); sheet.paste(t, (th * (i + 1), 28)); d.text((th * (i + 1) + 4, 6), n, fill="black")
sheet.save(os.path.join(a.out, "sheet.jpg"), quality=88)
print("sheet", sheet.size)

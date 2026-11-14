"""The app's own entry points on the real engine, one load: edit with references,
inpaint, upscale / img2img refine, outpaint. Times each, writes the results.

    python tools/minimax_app_check.py [--src out/adapter.png] [--out out/app]
"""
import argparse, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw
import cz_pipeline as P

p = argparse.ArgumentParser()
p.add_argument("--src", default="out/adapter.png")
p.add_argument("--out", default="out/app")
p.add_argument("--cases", default="edit,inpaint,refine,reframe,gen4k")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
src = Image.open(a.src).convert("RGB")
W, H = src.size
ok = True

def case(name, fn):
    global ok
    t = time.time()
    try:
        img = fn()
        img = img[0] if isinstance(img, (list, tuple)) else img
        img.save(os.path.join(a.out, name + ".png"))
        print(f"CASE {name}: OK {time.time()-t:.1f}s {img.size}", flush=True)
    except Exception as e:
        ok = False
        print(f"CASE {name}: FAIL {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

cases = a.cases.split(",")
if "edit" in cases:
    case("edit", lambda: P.generate_omni([src], "Edit <Picture 1>: make it a snowy winter day, snow on the pier and "
         "on the rocks, the man keeps his yellow coat and his lantern. Keep the composition unchanged.", "", W, H, 6, 5))
if "inpaint" in cases:
    m = Image.new("L", src.size, 0)
    ImageDraw.Draw(m).rectangle((int(W*.60), int(H*.16), int(W*.86), int(H*.34)), fill=255)
    import inspect
    print("inpaint_run", inspect.signature(P.inpaint_run), flush=True)
    case("inpaint", lambda: P.inpaint_run(src, m, "the man wears a navy blue knitted beanie", 6, 1.0, 5))
if "refine" in cases:
    import inspect
    print("_refine_whole", inspect.signature(P._refine_whole), flush=True)
    small = src.resize((1024, 1024), Image.LANCZOS).resize((2048, 2048), Image.BICUBIC)
    case("refine_030", lambda: P._refine_whole(P.get_pipe("img2img"), small, 0.3, 6, "a weathered lighthouse keeper in a yellow coat on a stone pier at dusk, detailed photograph", 5))
if "outpaint" in cases:
    base = src.resize((1536, 1536), Image.LANCZOS)
    case("outpaint_lr", lambda: P.outpaint_directions(base, None, ["left", "right"],
         "a stormy sea coast with black rocks and a stone pier, dusk, cinematic photograph", 6, 5))
    case("reframe_16_9", lambda: P.reframe(base, 16, 9, "contain",
         "a stormy sea coast with black rocks and a stone pier, dusk, cinematic photograph", 6, 5))
if "gen16" in cases:
    case("gen_4096", lambda: P.generate("An aerial photograph of a terraced rice valley at sunrise, mist between the "
         "hills, a small village with red roofs, a winding river, extremely detailed.", 4096, 4096, 6, 3, ""))
    case("gen_4096_again", lambda: P.generate("An aerial photograph of a terraced rice valley at sunrise, mist between the "
         "hills, a small village with red roofs, a winding river, extremely detailed.", 4096, 4096, 6, 4, ""))
if "gen4k" in cases:
    case("gen_3072", lambda: P.generate("A macro photograph of a jewel beetle on a wet green leaf, iridescent shell, water droplets, shallow depth of field.", 3072, 3072, 6, 11, ""))
print("ALL OK" if ok else "SOME FAILED")

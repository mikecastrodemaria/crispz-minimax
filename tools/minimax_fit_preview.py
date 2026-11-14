"""Fits the 24-channel latent -> RGB matrix of the live preview (least squares on a few
renders). Prints the matrix and bias to paste into cz_minimax.LATENT_RGB.

    python tools/minimax_fit_preview.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
import cz_minimax as M

PROMPTS = [
    "A red fox on a mossy log in a misty pine forest at dawn, photorealistic.",
    "A neon-lit street in Tokyo at night in the rain, pink and cyan signs, reflections on wet asphalt.",
    "A bright studio product shot of a yellow rubber duck, a green apple and a blue ceramic mug on a white table.",
    "A flat vector poster with large orange, purple and teal geometric shapes on a cream background.",
    "A portrait of an old woman with white hair and a black coat in front of a red brick wall, overcast light.",
    "A desert canyon at noon under a deep blue sky, orange rock, a small green cactus.",
]
pipe = M.load(); pipe.enable_model_cpu_offload()
X, Y = [], []
for i, p in enumerate(PROMPTS):
    keep = {}
    def cb(_p, k, t, d):
        keep["x0"] = d["latents"].detach().float().cpu()
    img = pipe(prompt=p, width=1536, height=1024, generator=torch.Generator("cpu").manual_seed(i),
               callback_on_step_end=cb).images[0]
    z = keep["x0"][0]                                   # (24, h, w)
    h, w = z.shape[1:]
    small = np.asarray(img.resize((w, h), Image.BOX), dtype=np.float32) / 127.5 - 1.0
    X.append(z.permute(1, 2, 0).reshape(-1, 24).numpy()); Y.append(small.reshape(-1, 3))
X = np.concatenate(X); Y = np.concatenate(Y)
A = np.concatenate([X, np.ones((X.shape[0], 1), np.float32)], axis=1)
W, *_ = np.linalg.lstsq(A, Y, rcond=None)
err = np.abs(A @ W - Y).mean() * 127.5
print("mean abs error on 255:", round(float(err), 2))
print("LATENT_RGB = [")
for r in W[:24]:
    print("    [%.5f, %.5f, %.5f]," % tuple(r))
print("]")
print("LATENT_RGB_BIAS = [%.5f, %.5f, %.5f]" % tuple(W[24]))

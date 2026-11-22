# crispz-minimax

The **MiniMax-H3-Image** studio of the crispz family: text-to-image and editing with up
to 16 reference images in one checkpoint, up to 16 megapixels without an upscaler, in 6
steps, Fooocus-style, **100% local**. A fork of
[crispz-moss](https://github.com/mikecastrodemaria/crispz-moss).

> **0.1.0, 2026-11-14.** First working version: text to image, editing with references,
> emulated img2img / inpaint / outpaint, measured on an RTX 5090. Built in one pass and
> not yet used day to day: expect rough edges. The measurements and the known limits are
> below, the differences with crispz-moss in `FORK.md`.

## Licence: read this first

**Where you live decides whether you may use this model.** MiniMax-H3-Image is released
under the
[MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)
(read on 2026-11-14). This is a summary, not legal advice: the licence text is what
counts.

- **Territory.** The licence grants rights only inside its "Applicable Territory":
  worldwide **except the European Union, the United Kingdom, the Republic of Korea and
  the United States of America** (sections I.3, I.5 and II). Section III.4 says the model
  and its outputs may not be used, reproduced, modified, distributed or displayed
  outside that territory. If you are in one of those places, the licence does not cover
  you: MiniMax asks you to request a separate authorization at
  [platform.minimax.io/license](https://platform.minimax.io/license).
- **Revenue and hosting.** A separate, prior written authorization is also required if
  the annualized revenue of you or your affiliates exceeds 1 million USD, or if you
  provide a hosted service to third parties (section IV).
- **Acceptable use.** The licence comes with an Acceptable Use Policy.

What this means for crispz-minimax:

- **This repository contains code only.** It ships no MiniMax weights and never
  redistributes them. The app downloads the files you choose from Hugging Face, on your
  machine, under your account and your responsibility.
- This code is published on GitHub and can be cloned from anywhere. Being able to
  install it does not mean the model licence applies to you: **each user is responsible
  for checking that they are allowed to use MiniMax-H3-Image where they are.**
- The quantized files the app can load (GGUF and others, from Comfy-Org, unsloth,
  Abiray and so on) are derivatives of MiniMax-H3-Image and carry the same licence and
  the same territory.

| Component | Licence |
|---|---|
| MiniMax-H3-Image weights and every quantized derivative | MiniMax-H3-Image Community License Agreement (territory above) |
| Text encoder (Qwen3-VL-32B, shipped by MiniMax in `MiniMaxAI/MiniMax-H3`) | the MiniMax-H3 licence (`MiniMaxAI/MiniMax-H3`, `LICENSE` and `docs/QA-about-License.md`); read it, it was not analysed here |
| diffusers (pipeline code) | Apache 2.0 |
| This app's code | CC BY-NC 4.0 ([LICENSE.txt](LICENSE.txt)) |

**Nothing leaves the machine.** There is no cloud function at all. Ollama, when used, is
a local server (its URL is logged at every call).

## What it does

| | |
|---|---|
| Text to image | 4, 9 and 16 megapixel presets (the model's range), 1 Mpx drafts; any size on a 32 px grid up to 20 Mpx |
| Editing with references | native: Ref 1 to Ref 4 of the Edit tab are `<Picture 1>` ... `<Picture 4>` in the prompt |
| img2img, upscale + refine, detailer | emulated: the render starts later on the model's fixed 6-step grid |
| Inpaint, outpaint, reframe | emulated, with the source image also given to the model as `<Picture 1>` |
| Around it | ESRGAN upscale, styles, wildcards, X/Y/Z grid, queue, presets, live preview, Improve prompt and Vision Mix (Ollama), Remove BG, Face Swap, Vectorize (SVG), the crispz protocol v1 CLI |

Fixed by the model, so not offered: steps (always 6), guidance (no CFG), sampler and
schedule, negative prompt. Asking for another value is ignored and said in the log.

**LoRAs** are applied by the app's own loader (diffusers has none for this model), as
hooks on the quantized DiT: the ComfyUI / ai-toolkit, kohya and diffusers file layouts
load, the weight slider re-scales live. MiniMax-H3 video LoRAs have the same layer names
and sizes and load too (their audio keys are skipped and said); how well they carry over
is not measured. DiffSynth files are refused by name. A step-distillation LoRA (pDMD
2 steps) loads, but the app still runs the model's 6 steps.

Not in this version: the safetensors
builds (fp8, int8 ConvRot, nvfp4), the dense 66 GB release checkpoint, the MOSS-Image
layout editor and character catalog of crispz-moss (their code is still in the repository,
hidden and unused).

## Requirements

- Windows or Linux, an NVIDIA GPU. Measured on an RTX 5090 (32 GB) with 64 GB of RAM.
  The peak of a render is the text encoder's 17.7 GB, so a 24 GB card should work; it
  has not been tried. Smaller cards need smaller GGUF files and have not been tried
  either.
- About 48 GB of RAM: the text encoder (18 GB) and the transformer (15.5 GB) wait in RAM
  while the other one is on the card.
- About 36 GB of disk for the default model files, plus 6 GB for the environment.
- Python 3.11 to 3.13.

## Install

```
git clone https://github.com/mikecastrodemaria/crispz-minimax
cd crispz-minimax
install.bat        (Windows)      ./install.sh   (Linux)
run.bat                           ./run.sh
```

The first render downloads what is missing into the Hugging Face cache:

| File | From | Size |
|---|---|---|
| Transformer `minimax_h3_image_pruned-UD-Q5_K_XL.gguf` | `unsloth/MiniMax-H3-Image-GGUF` | 15.5 GB |
| Text encoder `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | `unsloth/MiniMax-H3-GGUF` | 18.2 GB |
| VAE, tokenizer, configs | `MiniMaxAI/MiniMax-H3-Image` | 2.4 GB |

To use other files, put them in `checkpoints/` (or the folder set in Models): any GGUF
of the **pruned** transformer (unsloth, Abiray: same layout) appears in the Checkpoints
menu, any GGUF of the text encoder with transformers tensor names in the Text encoder
menu. A file found there is used before anything is downloaded. Files the app cannot
load are listed with the reason.

## Measured on an RTX 5090

GGUF `UD-Q5_K_XL` transformer, `Q4_K_M` text encoder, bf16 VAE, offload `model` (one
model on the card at a time), 2026-11-14. Times are for one image, the models already
loaded (loading takes about 50 s).

| | Time | Peak VRAM |
|---|---|---|
| 2048 x 2048, new prompt | 30 to 33 s | 17.7 GB |
| 2048 x 2048, same prompt again | 15 s | 15.8 GB |
| 3072 x 3072, new prompt | 41 s | |
| 4096 x 4096 | 60 to 70 s | 17.7 GB |
| 1536 x 1024 draft, same prompt | 9 s | 15.7 GB |
| Edit, one reference, 2048 x 2048 | 54 s | 18.6 GB |
| Inpaint 2048 x 2048 (source as reference) | 40 to 50 s | 18.7 GB |
| img2img 2048 x 2048, strength 0.5 | 12 s | 17.3 GB |
| Reframe to 16:9, 2720 x 1536 | 45 s | |

A new prompt costs 9 to 13 s of text encoding (20 s with a reference image): the 18 GB
encoder comes onto the card, encodes, and leaves. The same prompt again skips it.

## Getting good images

- Write a **long, concrete description in English** of the whole image: subject,
  setting, composition, light, materials, colours. The model follows it closely. The
  Improve prompt button (Ollama) is set up to write that kind of prompt.
- Text in the image: give it word for word between double quotes.
- Sizes: the model is made for 4 to 16 Mpx. The 1 Mpx drafts work but are below its
  range.
- No negative prompt: say what you want, not what you do not want.
- A quantized file changes the composition a little, like another seed would: judge a
  GGUF on texture and text, not on the pose.

## Editing with references

Put the image to edit in **Ref 1** of the Edit tab, up to three more references after
it, and write an instruction that names them: *Edit `<Picture 1>`: replace the coat with
the one of `<Picture 2>`, keep the face and the background unchanged.* The result takes
Ref 1's size unless *Output size from Settings* is ticked. The model accepts up to 16
references; the tab has four slots for now.

## img2img, inpaint, outpaint: what "emulated" means here

The model has no mask input and no strength. The app starts the render later on its
fixed grid, from the source image noised to that step, so **the strength snaps to five
values**: about 0.97, 0.89, 0.75, 0.53 and 0.23 (4, 3, 2 or 1 steps left for the last
four). The log says which one was used.

Inpaint and outpaint also hand the source to the model as `<Picture 1>`. Without that
the masked area was painted from the prompt alone and did not agree with the rest (a
hat with no head under it, in our test); with it, strength 1.0 gives a clean result.
Known limit: a directional outpaint can leave a visible change of tone where the new
band meets the picture; Reframe (contain) gave clean results on the same image.

## Command line

`README_CLI.md` describes the CLI and the crispz protocol v1. Quick render:

```
cli.bat --cli --txt2img --prompt "..." --gen-width 2048 --gen-height 2048 --seed 42 -o out/image.png
```

## Tests

```
.venv\Scripts\python tools\run_tests.py
```

52 of 53 pass (`test_vector.py` fails the same way in crispz-moss, on a vtracer
detail). They run without a GPU and without the model files. With the files and a GPU:
`tools/minimax_smoke.py` (one render), `tools/minimax_modes_check.py` (img2img, inpaint,
edit) and `tools/minimax_app_check.py` (the app's own functions).

## Credits

- MiniMax for MiniMax-H3-Image; Comfy-Org for the pruned single-file layout; unsloth
  and Abiray for the GGUF files; Hugging Face diffusers for the pipeline code.
- crispz-moss, crispz-ideogram, crispz-flux3, crispz-klein and Fooocus for everything
  around the engine.

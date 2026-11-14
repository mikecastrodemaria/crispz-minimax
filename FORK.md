# crispz-minimax : fork MiniMax-H3-Image de crispz-moss

- Base : `wip/moss-engine` de crispz-moss (a7b9876, 1.0.0), avec ses 496 commits ;
  branche locale renommée `main`.
- Remotes : `origin` = crispz-minimax, `moss` = crispz-moss (pour reprendre les
  correctifs communs).
- Pourquoi cette base et pas crispz-flux3 : l'édition par références, l'img2img et
  l'inpaint émulés, l'offload « encodeur seul » avec le DiT garé pendant l'encodage, le
  chargeur de fichiers uniques ComfyUI. Le détail est dans `CLAUDE.md` §3.

## Ce qui change par rapport à crispz-moss (prévu, rien n'est encore écrit)

| | crispz-moss | crispz-minimax |
|---|---|---|
| Modèle | MOSS-Image (5B, bf16, Apache 2.0) | MiniMax-H3-Image (33,1B, 20,1B en variante pruned, licence communautaire à territoire limité) |
| Moteur | `cz_moss.py` sur le pipeline OpenMOSS vendorisé | `cz_minimax.py` sur `MiniMaxH3ImageModularPipeline` (diffusers 0.43.0) et une classe pruned maison |
| Poids | bf16, fp8, int8 ConvRot | GGUF (unsloth, Abiray), quantifié en VRAM |
| Encodeur | Qwen3.5-9B | Qwen3-VL-32B, couche 50, quantifié |
| VAE | `AutoencoderKLFlux2` (32 canaux) | `AutoencoderKLMiniMaxH3Image` (24 canaux, décodeur ViT) |
| Steps / guidance | 50 steps, CFG 3 | 6 steps fixes, guidance 1 |
| Prompt négatif | réel | aucun |
| Tailles | 1 à 2 Mpx | 4 à 16 Mpx |
| Édition par références | LoRA Moss-Image-Edit | native, 1 à 16 images, `<Picture N>` |
| Layout JSON à boîtes | oui | retiré |
| LoRAs | chargeur maison | hors périmètre v0.1 |

Les notes de crispz-moss sont gardées dans `FORK-moss.md` et `FORK-moss-CLAUDE.md`,
celles des forks précédents dans `FORK-ideogram*.md`, `FORK-flux3*.md`, `FORK-klein.md`.

## Mesures

Aucune pour l'instant : voir `CLAUDE.md` §9.

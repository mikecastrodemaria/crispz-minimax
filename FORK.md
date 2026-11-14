# crispz-minimax : fork MiniMax-H3-Image de crispz-moss

- Base : `wip/moss-engine` de crispz-moss (a7b9876, 1.0.0), avec ses 496 commits ;
  branche locale renommée `main`.
- Remotes : `origin` = crispz-minimax, `moss` = crispz-moss (pour reprendre les
  correctifs communs).
- Pourquoi cette base et pas crispz-flux3 : l'édition par références, l'img2img et
  l'inpaint émulés, l'offload « encodeur seul » avec le DiT garé pendant l'encodage, le
  chargeur de fichiers uniques ComfyUI. Le détail est dans `CLAUDE.md` §3.

## Ce qui change par rapport à crispz-moss

| | crispz-moss | crispz-minimax |
|---|---|---|
| Modèle | MOSS-Image (5B, bf16, Apache 2.0) | MiniMax-H3-Image (33,1B, 20,1B en variante pruned, licence communautaire à territoire limité) |
| Moteur | `cz_moss.py` sur le pipeline OpenMOSS vendorisé | `cz_minimax.py` sur `MiniMaxH3ImageModularPipeline` (diffusers 0.43.0) et une classe pruned maison |
| Poids | bf16, fp8, int8 ConvRot | GGUF (unsloth ; Abiray non essayé), quantifié en VRAM |
| Encodeur | Qwen3.5-9B | Qwen3-VL-32B, couche 50, quantifié |
| VAE | `AutoencoderKLFlux2` (32 canaux) | `AutoencoderKLMiniMaxH3Image` (24 canaux, décodeur ViT) |
| Steps / guidance | 50 steps, CFG 3 | 6 steps fixes, guidance 1 |
| Prompt négatif | réel | aucun |
| Tailles | 1 à 2 Mpx | 4 à 16 Mpx |
| Édition par références | LoRA Moss-Image-Edit | native, `<Picture N>` (4 emplacements dans l'onglet, 16 dans le moteur) |
| img2img / inpaint | émulés (fin du planning, mélange de latents) | émulés : départ sur la grille PDD (5 forces), inpaint avec la source en référence |
| Layout JSON à boîtes | oui | masqué (code encore présent) |
| LoRAs | chargeur maison | hors périmètre v0.1 |

Les notes de crispz-moss sont gardées dans `FORK-moss.md` et `FORK-moss-CLAUDE.md`,
celles des forks précédents dans `FORK-ideogram*.md`, `FORK-flux3*.md`, `FORK-klein.md`.

## Mesures (RTX 5090, GGUF UD-Q5_K_XL, encodeur Q4_K_M, 14/11/2026)

| | |
|---|---|
| Chargement | 47 à 55 s |
| 2048 x 2048 | 30 à 33 s avec un nouveau prompt, 15 s avec le même, pic 17,7 Go |
| 4096 x 4096 | 60 à 70 s, pic 17,7 Go |
| Édition 1 référence, 2048 x 2048 | 54 s, pic 18,6 Go |
| Inpaint 2048 x 2048 | 40 à 50 s |
| img2img 2048 x 2048, force 0,5 | 12 s |

Le détail, les choix faits en route et ce qui n'est pas vérifié : `CLAUDE.md` §9.

## Pièges à connaître

- La licence du modèle exclut l'UE, le Royaume-Uni, la Corée du Sud et les États-Unis
  (`README.md`, « Licence »). Le dépôt ne contient aucun poids.
- Seuls les GGUF de la variante pruned se chargent ; les safetensors (fp8, int8, nvfp4)
  sont refusés par nom.
- La force d'un img2img ne prend que cinq valeurs (la grille de 6 steps est fixe).
- Un inpaint sans la source en référence donne un résultat incohérent : ne pas
  désactiver `INPAINT_WITH_REFERENCE` sans raison.
- Ne jamais lancer `pip install -r requirements.txt` dans le `.venv` installé (Pillow 13
  rétrogradé par gradio).

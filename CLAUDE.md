# crispz-minimax : cahier des charges pour Claude Code

`CLAUDE.md` du repo `mikecastrodemaria/crispz-minimax`. Claude Code le lit à l'ouverture du
projet. Rédigé le 14/11/2026 depuis crispz-moss (`wip/moss-engine`, a7b9876, 1.0.0), à
partir des fiches et des fichiers de configuration du modèle, des en-têtes de poids lus à
distance (GGUF et safetensors) et du code de diffusers 0.43.0. Aucun poids n'a été
téléchargé et rien n'a tourné : tout ce qui suit est lu, pas mesuré. Même méthode que les
forks précédents : lots courts, validation de Mike entre chaque lot, mesures sur la 5090.

Notes des forks précédents : `FORK-moss.md` et `FORK-moss-CLAUDE.md` (journal complet de
crispz-moss, pièges d'environnement), `FORK-ideogram*.md`, `FORK-flux3*.md`,
`FORK-klein.md`.

## 1. Objectif

Un studio local autour de **MiniMax-H3-Image** (MiniMax, novembre 2026) : txt2img et
édition par références (jusqu'à 16 images) dans un seul checkpoint, jusqu'à 16 Mpx sans
upscaler, en 6 steps sans CFG. La machine de référence est celle de Mike : RTX 5090
32 Go, 63 Go de RAM, Windows 11.

Ce que le modèle apporte à la famille : la très haute résolution native, le suivi
d'instructions, l'édition multi-références native. Ce qu'il coûte : un DiT de 33,1B
(20,1B dans sa variante « pruned ») et un encodeur Qwen3-VL-32B, qui ne tiennent pas
ensemble dans 32 Go. **Le lot 1 dit si une image sort en un temps acceptable sur la
5090, avant tout autre travail.**

## 2. Le modèle : ce qui est lu (14/11/2026)

| | MiniMax-H3-Image |
|---|---|
| Poids officiels | `MiniMaxAI/MiniMax-H3-Image` : DiT bf16 66,3 Go, VAE 2,34 Go ; encodeur dans `MiniMaxAI/MiniMax-H3` (`text_encoder/`, 66,7 Go) |
| Tâches | `t2i` et `edit` (`model_index.json`). Aucune entrée masque, aucune force de denoise |
| DiT | single-stream, 50 blocs + 2 blocs de token refiner, dim 5376, 56 têtes (head_dim 128), FFN 14336 (gated, fc1 en 2 x 14336), RoPE 3D, `text_dim` 5120 |
| Sortie | PDD : 6 têtes de sortie, une par step (`pdd_proj_out.0..5`, ou `final_layer.video_out` [576, 5376] = 6 x 96 dans les fichiers ComfyUI) |
| Planning | 6 steps fixes, sigmas `1.0, 0.96752, 0.88755, 0.75, 0.53277, 0.23201, 0.0`, Euler en vitesse, guidance 1, **aucun prompt négatif** |
| Encodeur | `Qwen3VLForConditionalGeneration` 32B (64 couches, hidden 5120), le DiT lit les états cachés de la **couche 50** ; tour vision utilisée pour les références |
| VAE | `AutoencoderKLMiniMaxH3Image` : encodeur CNN, décodeur ViT 2D de 24 couches, /16, **24 canaux**, `latents_mean` / `latents_std` dans la config |
| Latents | patch 2 x 2 : un token couvre 32 x 32 pixels (4096 x 4096 = 16 384 tokens) |
| Tailles | plage utile officielle 4 à 20 Mpx, optimum vers 16 Mpx ; `min_generation_area` par défaut 2048 x 2048 (en dessous, rendu plus grand au même ratio puis réduit en LANCZOS) |
| Références | 1 à 16 images, nommées `<Picture 1>`, `<Picture 2>` dans le prompt, dans l'ordre fourni ; encodées par l'encodeur texte ET par le VAE |
| Prompts | description anglaise longue et détaillée de toute l'image, 4096 tokens max ; expandeur officiel optionnel `MiniMaxAI/MiniMax-H3-Image-PE` (17,9 Go, NVFP4) |
| Bruit | `generator_device="cpu"` conseillé (même graine, même bruit partout) |

### diffusers

- 0.42.1 (la pile de crispz-moss) n'a que le H3 vidéo (`MiniMaxH3Transformer3DModel`).
- **0.43.0** (PyPI, 12/11/2026) ajoute `MiniMaxH3ImageModularPipeline`,
  `MiniMaxH3ImageBlocks`, `MiniMaxH3ImageTransformer2DModel`,
  `AutoencoderKLMiniMaxH3Image`, `MiniMaxH3Scheduler`. Blocs : `before_encode`,
  `text_encoder`, `vae_encoder`, `denoise`, `decode` ; branche `image_conditioned`
  quand `image` est fourni.
- `MiniMaxH3ImageTransformer2DModel` est `ModelMixin, ConfigMixin, AttentionMixin` :
  **pas de `from_single_file`, pas de mixin LoRA**. Il ne décrit que le checkpoint
  complet (embedding de temps dense, `pdd_proj_out` en `ModuleList`).
- Le chargeur de fichier unique du H3 vidéo refuse explicitement les checkpoints pruned
  (`adaln_t_table`).

### La variante « pruned » (celle de tous les fichiers légers)

Travail de Comfy-Org (`minimax_h3_image_pruned_bf16.safetensors`, 40,2 Go), repris par
tous les quantifieurs. La projection dense de modulation adaLN est remplacée par une
factorisation de rang 8 de la courbe de temps :
- `adaln_t_table` [1025, 8] (fp32) à la place de l'embedding de temps ;
- `blocks.N.adaln_proj.linear.weight` [96768, 8] + biais [96768] par bloc ;
- `final_layer.adaln_proj.linear.weight` [10752, 8].
Écart annoncé par unsloth aux 6 steps : erreur relative 6,5e-4 à 7,5e-4. 20,11 milliards
de paramètres, 528 tenseurs. Clés ComfyUI : `blocks.N.attn.qkv_proj` [21504, 5376]
(q, k, v empilés), `attn.out_proj`, `mlp.fc1` [28672, 5376], `mlp.fc2`,
`condition_proj`, `video_patch_proj` [5376, 96], `token_refiner.blocks.N.*`,
`rope.inv_freq`.

**Non lu** : comment le temps devient un index de `adaln_t_table` (1025 lignes). À lire
dans `comfy/ldm/minimax/` et dans le chargeur d'unsloth avant d'écrire la classe.

### Fichiers disponibles (en-têtes lus)

| Dépôt | Fichiers | Remarque |
|---|---|---|
| `unsloth/MiniMax-H3-Image-GGUF` | Q2_K 7,75 Go à Q8_0 22,03 Go, et `UD-*_XL` | mesures LPIPS contre bf16 ; pas de config embarquée ; pensé pour stable-diffusion.cpp |
| `Abiray/MiniMax-H3-Image-GGUF` | Q2_K à Q5_K_M (14,1 Go) | couches sensibles en fp32, planning embarqué (`config`) |
| `unsloth/MiniMax-H3-Image-FP8` | INT8-ConvRot et FP8, 20,27 Go | **clés diffusers** (`transformer_blocks.N.attn.to_q`, `pdd_proj_out.N`, `time_embedder.table`, `context_embedder`), sérialisation `Int8Tensor` |
| `Comfy-Org/MiniMax-H3-Image` | bf16 40,2 Go, int8 ConvRot 21 Go, w4a8 12,5 Go ; encodeur int8 ConvRot 27,1 Go, nvfp4 AWQ 15,7 Go ; VAE fp16 1,17 Go | format ComfyUI |
| `LSDJesus/MiniMax-H3-Image-fp8-nvfp4` | fp8 scaled 21 Go, nvfp4 12,5 Go | format ComfyUI (`comfy_quant`) |
| MLX (`VisualInference`, `cguldogan`) | Apple Silicon seulement | hors sujet ; confirme l'encodeur coupé à 50 couches |

Les GGUF d'unsloth et d'Abiray ont les mêmes 528 tenseurs, mêmes noms, mêmes formes : un
seul chargeur.

LPIPS contre le DiT bf16 à 1024 x 1024 (unsloth, plus bas = plus proche) : Q4_K_M 0,195,
UD-Q4_K_XL 0,172, Q5_K_M 0,157, UD-Q5_K_XL 0,154, Q6_K 0,134, Q8_0 0,083. À 2048 x 2048 :
INT8-ConvRot 0,140, FP8 0,255. Pic VRAM annoncé (DiT + décodage, 1024, A100) : 15,8 Gio
en Q5_K_M, 17,6 en Q6_K, 23,0 en Q8_0.

Encodeur GGUF : `unsloth/MiniMax-H3-GGUF` Q4_K_M 18,2 Go (Q2_K_M 13,1 Go), Abiray Q4_K_M
14,6 Go et Q3_K_M 12 Go. Selon unsloth, passer l'encodeur de bf16 à Q4_K_M déplace à lui
seul l'image de LPIPS 0,17 : l'encodeur compte autant que le DiT.

## 3. Base : crispz-moss

Fork de `wip/moss-engine` (a7b9876, 496 commits, 3 commits devant `main`), branche
locale renommée `main`. Remotes : `origin` = crispz-minimax (dépôt GitHub à créer par
Mike, rien n'est poussé), `moss` = crispz-moss.

Pourquoi : onglet d'édition par références, img2img / inpaint / outpaint émulés pour un
modèle sans masque natif, offload « encodeur seul » avec le DiT garé pendant l'encodage
(`moss_encode_swap`), chargeur de fichiers uniques ComfyUI (ConvRot, `qkv` fusionné),
protocole v1, passerelles, garde CivitAI par famille. `cz_flux3.py` est encore là et
sert de modèle pour câbler un ModularPipeline (retrait de blocs, hook par step).
crispz-flux3 aurait donné les références natives et le GGUF, mais sans l'émulation ni le
swap, et avec 86 commits de retard.

Ce qu'on garde : l'app, le détailleur, les tuiles, l'ESRGAN, la queue, le XYZ, le
protocole v1, `cz_handoff`, `cz_magic` (magic prompt local / Ollama, à réorienter vers
des descriptions longues en anglais), les tests. Ce qu'on remplace : le moteur
(`cz_moss.py` devient `cz_minimax.py`), les profils, les presets de steps (6, fixe), les
tailles. Ce qu'on retire, lot par lot et pas avant que le moteur tourne : le layout JSON
à boîtes (`cz_caption`, `cz_caption_ui`, `cz_design`), le catalogue de personnages
(`cz_catalog`), `cz_moss_lora.py`, `third_party/moss_image`.

## 4. Règles de travail

- Commits au nom de Mike (`Mike Castro Demaria`, même adresse que sur crispz-moss),
  auteur et committer, sans aucune ligne d'attribution à Claude.
- Ni tiret cadratin ni emoji, nulle part.
- Étape par étape : à la fin de chaque lot, s'arrêter, résumer, attendre la validation.
- Aucune fonction n'envoie de données hors de la machine ; Ollama compris.
- Toute dégradation est annoncée dans le log (`_log`), jamais silencieuse.
- Jamais de `trust_remote_code`. Le code repris d'ailleurs est vendorisé dans
  `third_party/` avec licence, commit d'origine et hash.
- **Ne pas copier de code ComfyUI** (GPL-3) : crispz est en CC BY-NC 4.0. Le lire pour
  comprendre, réécrire à partir de diffusers (Apache 2.0). Vérifier la licence du
  chargeur d'unsloth avant d'en reprendre une ligne.
- Ne jamais lancer `pip install -r requirements.txt` dans un `.venv` déjà installé.
- Un `.venv` neuf pour ce fork (diffusers 0.43.0) ; ne pas toucher à celui de crispz-moss.

## 5. Lots

### Lot 1 : une image sur la 5090, mesurée

1. `.venv` neuf, `diffusers==0.43.0` épinglé, le reste de la pile de crispz-moss ; les
   tests existants passent ou les écarts sont listés.
2. `cz_minimax.py` : classe pruned dérivée de `MiniMaxH3ImageTransformer2DModel`
   (table de temps de rang 8, 6 têtes), mapping des clés ComfyUI vers diffusers
   (`qkv_proj` scindé en q, k, v ; `fc1` / `fc2` vers `ff.net.0.proj` / `ff.net.2`),
   chargement GGUF en restant quantifié en VRAM (`GGUFParameter` / linéaires GGUF de
   diffusers, déjà utilisés par crispz-flux3).
3. Encodeur : trancher par la mesure entre (a) nf4 bitsandbytes depuis les poids
   officiels coupés à 50 couches, (b) le GGUF Q4_K_M, (c) le nvfp4 AWQ de Comfy-Org.
   Critères : VRAM, temps d'encodage, écart d'image.
4. VAE officiel (`vae_diffusers/`, 2,34 Go), décodage ViT : mesurer le pic à 4 et
   16 Mpx, prévoir des tuiles si besoin.
5. Swap encodeur / DiT, texte encodé avant tout VAE.
6. Fichiers de départ : DiT `UD-Q5_K_XL` (15,47 Go) ou `Q6_K` (16,6 Go) d'unsloth ;
   essayer `Q8_0` ensuite.

Acceptation : une image 2048 x 2048 puis une 4096 x 4096, mêmes prompt et graine que
l'exemple officiel, lisibles et sans grille ; temps et pics VRAM / RAM notés au §9.
Si le lot échoue (encodeur trop lent, pic hors carte), on s'arrête et on en parle.

### Lot 2 : interface txt2img

Presets de taille (4 à 16 Mpx et les ratios officiels), steps et guidance verrouillés
(6 et 1, dit dans l'interface), négatif retiré, aperçu live (matrice latent vers RGB à
refaire pour ce VAE 24 canaux), profils et garde VRAM, retrait du layout JSON.

### Lot 3 : édition par références

Onglet Edit branché sur la branche `image_conditioned` du pipeline, 1 à 16 références,
insertion de `<Picture N>` dans le prompt, taille de sortie au ratio de la référence.

### Lot 4 : img2img, inpaint, outpaint

Aucun n'est natif. Deux voies à comparer : l'édition par instruction avec l'image en
référence, et l'émulation de crispz-moss (SDEdit, mélange de latents). Avec 6 steps
fixes et une tête par step, le denoise émulé ne peut partir que d'une des sigmas du
planning (0,97 / 0,89 / 0,75 / 0,53 / 0,23) : c'est une déduction, à prouver sur images.

### Lot 5 : magic prompt, outils, passerelles

Magic prompt local orienté descriptions longues ; protocole v1 ; passerelles vers la
famille. L'expandeur officiel PE (17,9 Go) est hors périmètre tant que le magic prompt
local suffit.

### Lot 6 : tests, doc, livraison

README, `FORK.md`, lanceur Pinokio, e2e GPU, avis de licence (§6).

## 6. Licence : à afficher dans l'app et le README

MiniMax-H3-Image Community License Agreement. À dire tel quel, sans l'interpréter :
- La licence ne s'applique que dans son « territoire applicable » : le monde entier
  **sauf l'Union européenne, le Royaume-Uni, la Corée du Sud et les États-Unis**
  (sections I.3, I.5, II, III.4). Dans ces pays, il faut une autorisation séparée de
  MiniMax (platform.minimax.io/license).
- Autorisation écrite aussi au-delà de 1 M USD de revenu annuel, ou pour un service
  hébergé (section IV).
- Elle comporte une politique d'usage acceptable.

Décision de Mike (14/11/2026) : le code est publié sur GitHub, les utilisateurs sont
partout, chacun répond de sa licence. Conséquences :
- crispz-minimax **ne redistribue aucun poids** ; l'utilisateur les télécharge lui-même
  depuis Hugging Face.
- L'avis de territoire est affiché au premier lancement, dans le README et dans le
  message Pinokio, avec le lien vers le texte de la licence.
- Le code de crispz-minimax reste en CC BY-NC 4.0 (`LICENSE.txt`).

## 7. Hors périmètre v0.1

LoRAs (aucun mixin dans diffusers 0.43.0, aucun écosystème repéré), le H3 vidéo / audio
(« omni »), le checkpoint complet de 66 Go, l'expandeur PE, toute fonction cloud MiniMax.

## 8. Risques connus avant de commencer

- **Encodeur 32B** : le plus gros risque. Sur une carte 24 Go, unsloth constate que
  l'essentiel du temps est l'encodeur. Sur 32 Go il ne tient pas non plus à côté du DiT.
- Classe pruned : l'index de la table de temps n'est pas lu (§2).
- Linéaires GGUF de diffusers : déquantification à chaque passe, vitesse à mesurer sur
  20B de paramètres.
- Décodeur VAE en ViT à 16 Mpx (65 536 tokens de latents) : pic inconnu.
- Le modèle est sensible : bf16 ne reproduit pas fp32, une quantification change la
  composition comme une autre graine. Juger sur le texte et la texture, pas sur la pose.
- La fiche d'Abiray annonce 1024 x 1024 ; la fiche officielle place la plage utile à
  partir de 4 Mpx. À vérifier sur images.
- RAM : 63 Go pour garer un encodeur de 13 à 18 Go et un DiT de 15 à 22 Go.
- diffusers 0.43.0 a deux jours : régressions possibles sur le reste de l'app.

## 9. État au 14/11/2026

Les six lots ont été faits d'une traite le 14/11/2026, à la demande de Mike (« fais
tous les lots en autonomie, on regarde à la fin l'usabilité »), sans validation entre
les lots. Version 0.1.0. Ce qui suit est mesuré sur la 5090 sauf mention contraire.

### Ce qui a été tranché en route

- **Table de temps de la variante pruned** : `t` dans [0, 1] donne une position
  `t * 1024` dans `adaln_t_table`, interpolée linéairement entre deux lignes ; la
  projection adaLN prend ces 8 coordonnées sans activation. Lu dans
  `comfy/ldm/minimax/model.py` (GPL-3), réécrit dans `cz_minimax.CurveTimeTable` et
  `CurveAdaLN`, rien n'est copié.
- **Têtes PDD** : dans les fichiers uniques, `final_layer.video_out` [576, 5376] est une
  banque de 6 blocs, le bloc 0 est une tête complète et les blocs 1 à 5 des écarts.
  Tête du step k = bloc 0 + bloc k. Vérifié contre `pdd_fused_heads.safetensors` du
  dépôt officiel : écart relatif de l'ordre de 1e-10 sur les six.
- **Clés** : mapping repris du convertisseur diffusers du H3 vidéo (Apache 2.0) : `qkv`
  empilé `[q; k; v]`, FFN fusionnée `[gate; value]`. Les deux sont gardées telles
  quelles (`attn.fused_projections`, `GateFirstSwiGLU`) pour ne pas recouper un poids
  quantifié.
- **Encodeur** : le GGUF d'unsloth (`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, 18,2 Go) a
  les noms de tenseurs de transformers et déjà 50 couches, tour vision comprise. Chargé
  dans un `Qwen3VLForConditionalGeneration` à 50 couches sans norme finale ni `lm_head`.
  Les pistes nf4 et nvfp4 AWQ n'ont pas été essayées : celle-ci marchait du premier coup.
- **GGUF en VRAM** : `GGUFParameter` / `GGUFLinear` de diffusers, 208 matrices du DiT
  et les linéaires de l'encodeur restent quantifiés sur la carte.
  `nn.Module.load_state_dict` refuse ces poids (forme en octets) : affectation directe
  (`_assign_state`).
- **Chaîne de blocs** : l'officielle moins l'encodeur de texte (encodé et mis en cache
  par l'adaptateur, l'encodeur quitte la carte avant le reste), plus deux blocs crispz :
  un crochet de fin de step (aperçu, mélange de l'inpaint, arrêt) et un bloc qui coupe
  le début du planning (img2img). `init_pipeline` copie les blocs : les crochets sont
  retrouvés dans la chaîne du pipeline, pas dans la liste d'origine.
- **img2img** : départ au step k de la grille PDD, tête k (décalage `_cz_pdd_offset`
  sur le transformer). La force s'arrondit à la sigma la plus proche (1,0 / 0,97 /
  0,89 / 0,75 / 0,53 / 0,23). Lu sur images : cohérent de 0,25 à 0,9.
- **Inpaint** : le mélange de latents seul échoue (force 1,0 : un bonnet sans tête, un
  bloc blanc avec un prompt d'instruction). Avec la source donnée aussi en référence
  (`INPAINT_WITH_REFERENCE`), le résultat est propre à force 1,0, avec un prompt
  descriptif comme avec une instruction. C'est le réglage par défaut.
- **Édition par références** : native, excellente sur les essais (manteau rouge et
  nuit, neige), composition conservée.
- **Tailles** : rendu à la taille demandée (`min_generation_area=1024`), pas de
  passage forcé à 4 Mpx ; 1024 x 1024 sort propre, sous la plage du modèle.
- **Bruit** : toujours tiré sur le CPU (même graine, même image partout) ; un
  générateur CUDA de l'app est converti par sa graine.
- **Offload** : `model` forcé sous 46 Go de VRAM (encodeur puis DiT), `sequential`
  ramené à `model`. Pas de tuiles VAE : le décodage 16 Mpx tient entier.

### Mesures (GGUF UD-Q5_K_XL, encodeur Q4_K_M, VAE bf16, offload model)

| | Temps | Pic VRAM |
|---|---|---|
| Chargement | 47 à 55 s | |
| 2048 x 2048, nouveau prompt | 30 à 33 s | 17,7 Go |
| 2048 x 2048, même prompt | 15 s | 15,8 Go |
| 3072 x 3072, nouveau prompt | 41 s | |
| 4096 x 4096 | 60 à 70 s | 17,7 Go |
| 1536 x 1024, même prompt | 9 s | 15,7 Go |
| Édition 1 référence, 2048 x 2048 | 54 s | 18,6 Go |
| Inpaint 2048 x 2048, source en référence | 40 à 50 s | 18,7 Go |
| img2img 2048 x 2048, force 0,5 | 12 s | 17,3 Go |
| Recadrage 16:9, 2720 x 1536 | 45 s | |
| Encodage du texte | 9 à 13 s (20 s avec une référence) | |

Le pic est celui de l'encodeur, pas du débruitage ni du décodage. Pic de RAM non mesuré.

### Ce qui n'est pas fait, ou pas vérifié

- **Usage réel** : l'interface a été lancée et une image générée par le navigateur
  (1024 x 1024). Les onglets Edit, Inpaint / Outpaint et Upscale n'ont PAS été cliqués :
  leurs fonctions ont été appelées par script (`tools/minimax_app_check.py`). Détailleur,
  Face Swap, X/Y/Z, file d'attente, presets, protocole v1 en vrai : non essayés.
- **Aperçu live** : la matrice 24 canaux est calée (erreur moyenne 7,5 sur 255) et le
  rappel reçoit bien les latents, mais l'aperçu n'a pas été regardé dans l'interface.
- **Outpaint directionnel** : remplit, mais laisse un raccord de teinte visible aux
  jonctions sur l'essai. Le recadrage « contain » était propre sur la même image.
- **Ménage** : `cz_moss.py`, `cz_moss_lora.py`, `third_party/moss_image`, `cz_caption*`,
  `cz_design`, `cz_catalog`, `cz_handoff`, `cz_magic`, `cz_flux3.py` sont toujours là.
  Le panneau de layout, le catalogue, les onglets de passerelle, la section LoRA et les
  LoRAs d'édition sont masqués, pas retirés. `cz_flux3` sert encore (marqueurs de
  famille, filtre de log de l'arrêt). Beaucoup de commentaires parlent encore de MOSS,
  FLUX.3 ou klein ; les noms `_MOSS_*`, `ZIMAGE_*`, `FLUX3_*` restent.
- **Tests** : 52 sur 53 (`test_vector` échoue pareil dans crispz-moss). 19 fichiers de
  tests propres à MOSS ou FLUX retirés, quelques tests à deux dépôts retirés, un test
  unitaire du moteur ajouté. Aucun test e2e GPU dans la suite : les trois scripts de
  `tools/` en tiennent lieu.
- **Fichiers non essayés** : les GGUF d'Abiray (même structure, lecture de
  `comfy.gguf.orig_shape` écrite mais jamais exercée), Q6_K, Q8_0, tout autre encodeur.
- **Magic prompt local et Describe local** : indisponibles (l'encodeur n'a pas de tête
  de langage). Improve prompt (Ollama) est orienté descriptions longues en anglais.
- **16 références** : le moteur les accepte, l'onglet n'a que 4 emplacements et le
  protocole `MAX_REFS = 4`.
- **Licence de l'encodeur** (`MiniMaxAI/MiniMax-H3`) : non lue.
- **Lanceur Pinokio** : écrit dans `D:\Github\crispz-minimax.pinokio` d'après celui de
  crispz-flux3, jamais exécuté. `install.bat` / `install.sh` non rejoués : le `.venv` a
  été monté à la main avec uv, de la même façon (lock sans Pillow, puis Pillow 13).
- `README_CLI.md`, `config_modification_tutorial.txt`, `SECURITY.md` : noms remplacés,
  contenu non relu.
- **Avis de licence** : écrit dans le README, la note du dépôt (onglet Models) et le log
  au premier chargement de chaque session. Pas de fenêtre au premier lancement.

## 10. Sources

- https://huggingface.co/MiniMaxAI/MiniMax-H3-Image (fiche, configs, LICENSE)
- https://huggingface.co/Comfy-Org/MiniMax-H3-Image
- https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF
- https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF
- https://huggingface.co/unsloth/MiniMax-H3-Image-FP8 (NOTICE : description du pruning)
- https://huggingface.co/unsloth/MiniMax-H3-GGUF (encodeur)
- https://huggingface.co/LSDJesus/MiniMax-H3-Image-fp8-nvfp4
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image
- wheel `diffusers-0.43.0` (`modular_pipelines/minimax_h3_image/`,
  `models/transformers/transformer_minimax_h3_image.py`)

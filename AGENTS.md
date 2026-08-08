# AGENTS.md — GRID: Multi-Behavior Multimodal Generative Recommendation

Read this top-to-bottom and you should understand the whole project: what it is, the
end-to-end pipeline, where every piece lives, what has been customized in this fork, how to
run it, and the gotchas. Keep it updated as the project moves.

---

## 1. What this project is

**GRID** (Generative Recommendation with semantic IDs) is Snap Research's open framework
(arXiv:2507.22224, `README.md`). It does generative recommendation in 3 stages:
LLM/VLM embeddings → quantize items into hierarchical **semantic IDs (SIDs)** → train a
**TIGER**-style T5 encoder-decoder that autoregressively generates the next item's SID tokens.

**This fork** (`erichkingruc`) extends GRID to **Multi-Behavior + Multimodal (MMMB)**
recommendation on the **Alibaba Tmall** dataset:
- Items are quantized from **both text and image** embeddings via a dual-modality MD-RQ-VAE
  (shared + text-specific + image-specific codebooks).
- Each interaction carries a **behavior type** (click / collect / cart / alipay). The TIGER
  sequence interleaves a behavior token with each item's SID levels.
- Goal: predict the next item *and* behavior, with emphasis on the rare `buy` (alipay) behavior.

Reference (design only, NOT imported): `/scratch/yw8866/MMMB-Genrec`.
This GRID checkout is a **git repo** (`/scratch/yw8866/GRID/.git`); `rec-tmall` is too.

---

## 2. End-to-end pipeline (data → trained model)

```
Tmall raw logs + product file  (/scratch/yw8866/rec-tmall/*.txt, ~57 GB of logs)
  │
  │ (a) download_image.py            download product images (squashFS archives -> image_sqf/*.sqf)
  │     image_sqf/extract_sqf_ids.py -> all_ids.txt  (items that actually have an image)
  │
  │ (b) valid_data/trim_data.py      keep only log/product/review rows whose item_id has an image
  │                                  -> valid_data/<same filenames>
  │
  ├─ (c1) text_embedding/build_text_embeddings.py   SigLIP2 text encoder on product names (Chinese)
  │                                  -> siglip2-so400m-patch14-384.npy + _ids.pkl
  ├─ (c2) image_embedding/build_embeddings.py       SigLIP2 image encoder on product images
  │                                  -> siglip2-so400m-patch14-384.npy + _ids.pkl  (same vector space)
  │
  │ (d) GRID experiment=md_rqvae_train_flat   dual-modality residual-VQ over text+image embeds
  │     -> 3 codebooks: shared(L0) / text(L1) / image(L2);  checkpoint .ckpt
  │     experiment=md_rqvae_inference_flat    -> semantic_ids.pt  (per-item SID tuples)
  │
  │ (e) src/data/build_tmall_tfrecords_mb.py  group interactions per user, attach behavior ids,
  │     split 80/10/10 by user_id%100         -> tfrecords_mb*/{training,evaluation,testing}
  │
  └─ (f) GRID experiment=tiger_mb_train_flat  train multi-behavior TIGER on the TFRecords,
        injecting MD-RQ-VAE codebook geometry into the embedding table (codebook_init_path).
```

SigLIP2 embedding_dim = **1152** (so400m-patch14-384); MD-RQ-VAE latent_dim = **512**.
The codebook checkpoint used downstream lives at
`/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt`
and SIDs at `/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt`.
Codebook sizes in use: **shared=2048, text=1024, image=1024** (the `2048_1024` run name).

GRID is also capable of the simpler single-modality SID pipelines (kept, not the focus here):
`rkmeans_*`, `rqvae_*`, `rvq_*`, `rqvae_siglip_*`, `sem_embeds_inference_*`, `tiger_train_flat`
(single-behavior), `tiger_inference_flat`.

---

## 3. Repository layout

```
GRID/
├── src/
│   ├── train.py          Hydra entrypoint (also monkey-patches torch.save to pickle proto 5
│   │                     for >4 GiB objects; rootutils sets project root via .project-root)
│   ├── inference.py      Hydra entrypoint for embedding/SID generation + tiger inference
│   ├── data/
│   │   ├── build_tmall_tfrecords.py        single-behavior TFRecord builder
│   │   ├── build_tmall_tfrecords_mb.py     ★ multi-behavior builder + pv rebalancing (see §5)
│   │   └── loading/
│   │       ├── components/
│   │       │   ├── collate_functions.py    ★ collate_with_behavior_sid_causal_duplicate (train),
│   │       │   │                             collate_with_behavior_interleave (val/test)
│   │       │   ├── iterators.py             TFRecordIterator
│   │       │   ├── pre_processing.py        map_sparse_id_to_semantic_id, tensor conversion
│   │       │   ├── label_function.py        NextKTokenMasking (next_k = 1+H)
│   │       │   ├── dataloading.py           UnboundedSequenceIterable
│   │       │   └── interfaces.py            SequentialModelInputData / ...LabelData, dataset configs
│   │       └── datamodules/sequence_datamodule.py   Lightning DataModule
│   ├── models/
│   │   ├── modules/
│   │   │   ├── base_module.py               BaseModule (LightningModule; stores self.model)
│   │   │   ├── huggingface/transformer_base_module.py  TransformerBaseModule (self.encoder=...)
│   │   │   └── semantic_id/tiger_generation_model.py   ★ all TIGER model classes (see §6)
│   │   └── components/network_blocks/       MLP, embedding aggregator, normalize layer
│   ├── components/
│   │   ├── eval_metrics.py                  ★ NDCG, Recall, *RetrievalEvaluator,
│   │   │                                      MultiBehaviorSIDRetrievalEvaluator (see §7)
│   │   ├── loss_functions.py, optimizer.py, scheduler.py, training_loop_functions.py
│   │   └── quantization_strategies.py, clustering_initializers.py, distance_functions.py
│   └── utils/
│       ├── codebook_embedding_init.py       ★ orthogonal up-projection of MD-RQ-VAE codebooks
│       ├── custom_hydra_resolvers.py         math_eval, extract_fields_from_list_of_dicts, etc.
│       ├── restart_job*.py                   Slurm requeue / checkpoint-resume callbacks
│       └── utils.py, file_utils.py, instantiators.py, masking_utils.py, ...
├── configs/                                  Hydra config tree (see §8)
├── sbatchfiles/                              ★ Slurm launchers (see §9)
├── expoutfile/                               Slurm stdout (*.out)
├── logs/, outputs/                           Lightning/Hydra run outputs
├── requirements.txt, README.md, notices.txt
└── AGENTS.md                                 this file
```

★ = files actively customized in this fork.

External data root: `/scratch/yw8866/rec-tmall/` — raw logs, embeddings, SIDs, all TFRecord
variants, and the standalone analysis scripts (`analyze_tfrecords_mb.py`,
`debug_behavior_strings.py`, `analyze.py`).

---

## 4. Data: behaviors and TFRecord format

Raw Tmall logs: `\x01`-separated, fields `item_id, user_id (u<digits>), behavior, timestamp`.
Raw behavior strings (lowercase) → behavior id used everywhere downstream:

| raw string | id | semantic | raw share |
|-----------|----|----------|-----------|
| `click`   | 0  | pv (page view) | ~93.6% |
| `collect` | 1  | fav (favorite) | ~2.4% |
| `cart`    | 2  | add-to-cart    | ~3.0% |
| `alipay`  | 3  | buy (purchase) | ~0.9% |

TFRecord row = one user (built by `build_tmall_tfrecords_mb.py`):
`user_id` (int64 scalar), `sequence_data` (int64 item ids, time-sorted),
`behavior_data` (int64 behavior ids, parallel to items). Users split deterministically by
`user_id % 100` → 80% training / 10% evaluation / 10% testing. Length filter [min,max]=[5,200].

In the TIGER sequence each item becomes `1+H` tokens (H=`num_hierarchies`=3, stride=4):
`[behavior, SID_L0(shared), SID_L1(text), SID_L2(image)]`. The collate fn produces causal
sliding-window subsequences for training; `NextKTokenMasking(next_k=stride)` sets the targets.

### ⚠ Behavior-map bug (FOUND & FIXED this stage)
`BEHAVIOR_MAP` used `{pv,fav,cart,buy}` but raw strings are `{click,collect,cart,alipay}` —
only `cart` matched; everything else fell through to `UNKNOWN_BEHAVIOR=0`, so **fav and buy
were 0 rows** and "users with ≥1 buy" was 0. All prior `tfrecords_mb` were buy-less.
Fix: correct map + `UNKNOWN_BEHAVIOR=-1` (unknown rows dropped & reported); the builder now
prints a behavior histogram in Phase 1 (`<-- EMPTY` flag) to self-verify. **Any dataset built
before this fix must be rebuilt.**

---

## 5. pv/click rebalancing (build_tmall_tfrecords_mb.py)

Even with the fix, pv ≈ 93.6%, so the loss is dominated by "predict a click." Rebalancing
options (all **TRAINING-split only**; eval/test stay natural for honest evaluation):

- `--augment_mode user_drop` — greedily drop the most pv-dominated whole training users until
  the training pv share reaches `--target_pv_ratio` (default 0.80). Sequences stay 100% real;
  loses some buy/cart events; retains least data. Output: `tfrecords_mb_userdrop`.
- `--augment_mode event_downsample` — thin pv events *within* each training sequence
  (keep ratio `k = t(1-p0)/(p0(1-t))`, computed from real data; keep 100% of cart/collect/
  alipay) until pv share hits target. Keeps most data; sequences become synthetic. Output:
  `tfrecords_mb_seqdownsample`.
- `--augment_mode none` — natural distribution. Output: `tfrecords_mb`.

**Val/test handling (decided design):** randomly **subsample val/test (distribution-
preserving → stays ~93.6% pv)** to restore the 8:1:1 split ratio. The target count comes from
a **mode-independent reference** `n_train_ref` (= the user_drop greedy keep-count, computed in
both modes) and a fixed RNG stream (`seed+1`). Consequence: **val/test are IDENTICAL across
user_drop and event_downsample** — a shared eval set for a fair comparison. The builder prints
the post-transformation behavior ratio per split and the final train:val:test counts.

Old/superseded flags `--augment`, `--drop_pv_only_prob` were replaced by `--augment_mode`.
(`build_tfrecords_mb_aug.sbatch` still uses the old flags — legacy, kept by the user.)

⚠ Builder uses `mkdir(exist_ok=True)` and restarts file indices at 0: it overwrites
same-named files but does NOT delete orphans. **Delete the split dirs before rebuilding.**

---

## 6. Model architecture (tiger_generation_model.py)

Class hierarchy (Lightning):
```
BaseModule (base_module.py; self.model = hf model)
 └ TransformerBaseModule (transformer_base_module.py; self.encoder = hf model, self.decoder)
    └ SemanticIDGenerativeRecommender        (line 28)
       └ SemanticIDEncoderDecoder             (line 463; sep_token, SID embedding tables, T5 enc/dec)
          └ SemanticIDMultiBehaviorEncoderDecoder  (line 1214) ★ the MB model used here
Helpers: SemanticIDEncoderModule (1036, wraps T5EncoderModel, deletes its embed_tokens/shared),
         SemanticIDDecoderModule (960), T5MultiLayerFF (1084), BMTVEmbeddingWrapper (1119).
```

**T5 specifics that matter:** encoder is `transformers.T5EncoderModel` fed via `inputs_embeds`
(its own `embed_tokens`/`shared` tables are *deleted* — all embeddings are built in our code).
T5 has **no absolute positional embeddings**; its only positional signal is the **relative
attention bias** added inside each self-attention layer. So any extra structure must be added
to `inputs_embeds`. Encoder/decoder are 4 layers, d_model=128, 6 heads, d_ff=1024 (config).

`SemanticIDMultiBehaviorEncoderDecoder`:
- Embedding table layout: SID levels concatenated `[L0 | L1 | L2]` then behavior tokens.
  `sid_level_offsets` = cumulative SID sizes; `behavior_offset` = sum of SID sizes.
- `_mb_offset_pattern`: (1+H)-periodic offset so a flat id sequence indexes the right table block.
- Decoder heads: `decoder_mlp[h]` per SID level (sizes from codebook) + `behavior_mlp`
  (predicts behavior). So `num_hierarchies+1` predictions per item.
- `encoder_forward_pass` branches on the feature switches below.

### Optional encoder feature switches (config + model; default OFF, need num_hierarchies==3)
- **`use_item_pe`** — on the *standard flat* path, add learned **item-level PE**
  (`item_pos_embedding`: all 4 tokens of item k share index k → 0,0,0,0,1,1,1,1,…) +
  **modality TE** (`modality_type_embedding(stride=4)`: within-item slot → 0,1,2,3,0,1,2,3,…)
  to `inputs_embeds`. Sequence length & T5 layers unchanged. **T5-compatible**: lives in
  embedding space, complements (does not collide with) T5's relative attention bias.
  `item_pe_max_seq_len` must cover N=#items (not L=#tokens).
- **`use_bmtv`** — Behavior-Modulated Triple-View encoder (`BMTVEmbeddingWrapper`): replaces the
  flat lookup with a `(B, 3N, d)` sequence. Three strictly-orthogonal views per item
  (V_shared=e_f, V_text=e_t, V_img=e_i), modulated by a 3-way `Softmax(Linear(e_B))` behavior
  gate, interleaved with shared item-PE + 3-way modality-TE. Decoder/T5 layers unchanged.
- `use_item_pe` and `use_bmtv` are **mutually exclusive** (ValueError at init).
- Both **nullify `sep_token`** via `register_parameter('sep_token', None)` because that path
  bypasses it and DDP crashes on any registered parameter that never receives a gradient.

History: an earlier **dual-view** `BMDVEmbeddingWrapper` / `use_bmdv` was replaced by the
triple-view `BMTV`. Do not reintroduce the `bmdv` names.

### Loss-rebalancing switches (config + model; default OFF, independent)
Total training loss = `behavior_CE + Σ_h SID_CE_h`, all summed equally. Two switches reshape it
to fight the pv imbalance (no covariate shift — all data/histories kept):
- **`focal_behavior`** (+ `focal_gamma` default 2.0, + optional `focal_alpha` per-class weights
  `[pv,fav,cart,buy]`) — replace plain CE on the behavior head with focal loss
  `FL=-α_c·(1-p_t)^γ·log(p_t)` (`FocalLoss` in `components/loss_functions.py`). Focuses gradient on
  hard/rare behaviors. Affects t3 behavior P/R/F1 + joint. ⚠ γ=2 **overcorrects** (collapses pv —
  see §12); lower γ is the primary knob, α can't rescue pv at high γ.
- **`behavior_weighted_sid`** (+ `sid_behavior_weights=[pv,fav,cart,buy]`, default `[1,3,3,8]`) —
  weight each item's SID loss by its TARGET behavior (`fut_ids[:,0]`), reduced as a weighted
  mean `Σ(w·CE)/Σw`. Pushes the model to nail SIDs for rare-behavior (buy) items → targets t1.
  Stored as a registered buffer (DDP-safe). Only affects the training loss, not eval/generation.

### Two-View Semantic Alignment + Retrieval (config + model; default OFF, need H==3)
Leverages the MD-RQ-VAE **addable** property (`e_f+e_t ≈ text latent`, `e_f+e_i ≈ image latent`,
preserved into d_model by the orthogonal `W_up`). Two heads `align_head_text/image` on the decoder
**item-summary hidden state** (position 0, after BOS) predict the next item's `(v_t, v_i)`.
- **`use_semantic_align`** (+ `semantic_align_weight`, `align_loss_type`, `align_temperature`) —
  TRAIN two heads on a **stop-grad** `_sid_two_view(fut_ids[:,1:])` target, added in `model_step`.
  `align_loss_type='mse'` (regress query→GT view) or `'contrastive'` (InfoNCE, `_contrastive_align`:
  query must RANK its GT view above in-batch negatives, cosine/temperature, false-negative SID mask).
  Contrastive matches the cosine rerank and trains discriminative (not shared-code) structure.
- **`use_semantic_rerank`** (+ `rerank_weight`, default 0.5) — INFER: in `generate_multibehavior`,
  reorder the candidate pool by cosine of predicted `(q_t,q_i)` to each candidate's `(v_t,v_i)`;
  `_blend_rerank` per-sample min-max normalizes and blends with the decoder score (w=0 baseline,
  w=1 pure semantic). Rerank helps recall@10 only if pool>10 → widen `top_k_for_generation` (e.g.
  50); the evaluator takes top-K by the blended score itself.
- Motivation/expected effect: **aggregate/pv lever** (converts pv near-misses to hits; §12c). NOT a
  buy lever (buy SID already 96% solved). Helper `_sid_two_view` shared with the probe.
- ⚠ Rerank uses the trained heads, so **train with `use_semantic_align=true` first**. Using rerank
  in a *training* run without align leaves the heads gradient-less → DDP unused-param crash; for a
  checkpoint-only eval use `train=false`.

### Codebook init (utils/codebook_embedding_init.py)
`build_codebook_init_embedding` loads the 3 MD-RQ-VAE centroid matrices (shared/text/image) and
**orthogonally up-projects** them (`W_up ∈ R^{d_vae×d_model}`, `nn.init.orthogonal_`) into the
embedding table so projected cosine similarities match the VAE latent space; behavior tokens
get random init. Pass `codebook_init_path=<.ckpt>`; `codebook_sizes` is validated against sizes
detected in the checkpoint. ⚠ Uses `torch.load(..., weights_only=False)` — required because
PyTorch 2.6 flipped the default to `True`, which breaks loading the trusted Lightning ckpt. Keep it.

---

## 7. Evaluation (components/eval_metrics.py)

`MultiBehaviorSIDRetrievalEvaluator` reports **Recall@{5,10}** and **NDCG@{5,10}** over three
tracks (top-k generation, k from `top_k_for_generation`, default 10):
- **t1** — buy-only: samples where GT behavior == buy (`buy_behavior_id=3`); SID HR/NDCG.
- **t2** — behavior-conditioned: decoder conditioned on GT behavior; SID HR/NDCG over all behaviors.
- **t3** — free generation: joint behavior+SID match, plus behavior-prediction metrics on the
  best-beam predicted behavior: `t3_behavior_accuracy` and per-behavior one-vs-rest
  `t3_behavior_{precision,recall,f1}_{pv,fav,cart,buy}` (`BehaviorClassMetric`, DDP-safe
  TP/FP/FN gather; behavior names default `{0:pv,1:fav,2:cart,3:buy}`, overridable via the
  evaluator's `behavior_names` arg).
Checkpoint monitor is `val/t2_recall_5` (mode max). Consider `t1_recall_5` if selecting for buy.

**Semantic near-miss probe** (eval-only diagnostic, switch `probe_sid_distance`, default off;
`model._update_sid_probe`). Among MISS cases (GT SID not in free-gen top-k), logs per behavior +
overall: `probe_{group}_miss_{pred_sim,rand_sim,frac}` — cosine similarity of the best-beam
predicted SID to GT in the `e_f+e_t / e_f+e_i` reconstruction space, vs a random in-batch item, and
the miss fraction. `pred_sim ≫ rand_sim` ⇒ near-miss (retrieval could help); `≈` ⇒ random. Runs on
the natural test set (~93% pv) regardless of training rebalancing — read per-behavior groups, not
`overall` (pv-dominated). Requires num_hierarchies==3. Findings in §12.

vs MMMB-Genrec: same metrics & K. MMMB matches the *entire generated sequence* (behavior+SIDs)
and has Target/Behavior_specific/Behavior_item modes; we score the SID portion and break
behavior out into the t1/t2/t3 tracks. Recall denominators differ (MMMB assumes 1 GT).

---

## 8. Configs (Hydra)

`configs/train.yaml` / `configs/inference.yaml` are the roots; `experiment=<name>` selects a
full `# @package _global_` override (everything: data_loading, model, trainer, eval, …).
Shared groups: `callbacks/`, `trainer/` (ddp.yaml), `logger/` (csv, + wandb inline),
`paths/`, `extras/`, `hydra/`. Custom resolvers (`math_eval`, `extract_fields_from_list_of_dicts`,
`create_map_from_list_of_dicts`) registered in `utils/custom_hydra_resolvers.py`.

Key experiment configs:
- `md_rqvae_train_flat.yaml` / `md_rqvae_inference_flat.yaml` — dual-modality SID learning/gen.
- `tiger_mb_train_flat.yaml` ★ — the multi-behavior training run. Notable knobs:
  `data_dir`, `semantic_id_path`, `num_hierarchies` (=3), `num_behaviors` (=4),
  `codebook_init_path`, `codebook_sizes` [2048,1024,1024], and the feature switches
  `use_item_pe` / `item_pe_max_seq_len`, `use_bmtv` / `bmtv_max_seq_len`.
  `sequence_length = (1+H)*40` (40 items). Train collate =
  `collate_with_behavior_sid_causal_duplicate`; val/test = `collate_with_behavior_interleave`.
  Trainer: DDP, precision 32-true, accumulate_grad_batches 4, val_check_interval 5000.

---

## 9. How to run (Slurm + Singularity)

Container: `singularity exec --nv --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro
/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif bash -c "source /ext3/env.sh
&& conda activate py310 && <cmd>"`. Stdout → `expoutfile/`. Slurm account
`torch_pr_69_tandon_advanced`, A100 GPUs. (Builders/analysis scripts don't need `--nv`.)

`sbatchfiles/`:
| file | what |
|------|------|
| `build_tfrecords_mb.sbatch`               | natural MB build → `tfrecords_mb` |
| `build_tfrecords_mb_userdrop.sbatch`      | `user_drop`, target_pv 0.80 → `tfrecords_mb_userdrop` |
| `build_tfrecords_mb_seqdownsample.sbatch` | `event_downsample`, target_pv 0.80 → `tfrecords_mb_seqdownsample` |
| `build_tfrecords_mb_aug.sbatch`           | legacy (old `--augment`/`--drop_pv_only_prob` flags) |
| `analyze_tfrecords_mb.sbatch`             | behavior distribution over a built dataset |
| `debug_behavior_strings.sbatch`           | sample raw logs → distinct behavior strings |
| `train_tiger_mb.sbatch`                   | training (edit `data_dir=`, `use_item_pe=`, etc.) |

Train example:
```
python src/train.py experiment=tiger_mb_train_flat \
  codebook_init_path=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt \
  data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
  semantic_id_path=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt \
  use_item_pe=true num_hierarchies=3
```
W&B project `MMMBGRec`. `restart_job` callback supports Slurm requeue + checkpoint resume.

---

## 10. Conventions & gotchas

- **DDP unused-parameter crash**: any registered `nn.Parameter` that never receives a gradient
  ("Expected to have finished reduction in the prior iteration…") kills training. Nullify
  bypassed params with `register_parameter(name, None)` (done for `sep_token`).
- **Don't shell-scan the raw logs** (~57 GB) — write a bounded Python script + sbatch (user
  preference). Examples in `rec-tmall/`: `debug_behavior_strings.py` (line-capped),
  `analyze_tfrecords_mb.py` (TFRecord histogram).
- **torch.save / load**: `train.py` forces pickle protocol 5; loaders of trusted local ckpts
  must pass `weights_only=False` (PyTorch 2.6 default flip).
- **Rebuilding TFRecords overwrites in place but leaves orphans** — `rm -rf` the split dirs first.
- **num_hierarchies must be 3** for `use_item_pe` / `use_bmtv` (shared/text/image SID levels).
- TensorFlow is only in the `py310` conda env (used by builders/analysis), not base Python.

---

## 11. Verification status (keep current)

- [x] Behavior-map bug fixed; `debug_behavior_strings` confirmed raw strings
      `click/cart/collect/alipay` (93.63 / 2.99 / 2.44 / 0.93 %).
- [x] Builder runs end-to-end; `user_drop` produced 80% pv training (confirmed by user).
- [x] `use_item_pe` and `use_bmtv` (and codebook init, weights_only fix) implemented.
- [ ] Rebuilt `tfrecords_mb` (post behavior-fix) + `tfrecords_mb_seqdownsample` not yet
      confirmed run after the latest val/test-parity change.
- [x] Behavior-imbalance interventions run end-to-end (data rebalancing, scale-up, focal,
      behavior-weighted SID) — see §12.
- [x] SID-collision / codebook-utilization diagnostic run — codebooks healthy (≈100% active all
      levels), collision 36.2% (acceptable, and the metric matches SID tuples not item ids so
      collisions don't cap it). Ruled out as the ceiling. See §12.
- [x] Semantic near-miss probe run on user_drop — misses are near-misses, conversions already solved
      on the SID side; behavior head is the bottleneck. See §12.
- [x] Behavior-focal γ/α frontier mapped (γ=1 best buy_F1 0.341; cart/fav unlearnable; joint flat) — §12d.
- [ ] Two-View semantic align + rerank **implemented** (§12e) — NOT yet run. Next experiment.
- [ ] use_item_pe / use_bmtv not re-evaluated on fixed data.

---

## 12. Experiment log — behavior-imbalance interventions

Test-set results (top-k=10). Bold = best in column. ⬇ loss better; others higher better.
pv≈93.6% / fav≈2.4% / cart≈3.0% / buy≈0.9% of interactions.

| # | run | Loss⬇ | t1_NDCG (buy) | t1_Rec | t2_Rec | t3_BehAcc | buy_P | buy_R | buy_F1 | cart_R | fav_R | pv_R | joint_Rec |
|---|-----|------|------|------|------|------|------|------|------|------|------|------|------|
| 1 | Original (natural)          | 12.661 | 0.886 | 0.963 | 0.150 | **0.864** | 0.388 | 0.141 | 0.207 | 0.174 | 0.297 | **0.907** | 0.186 |
| 2 | Seq Down (event_downsample) | 12.861 | 0.861 | 0.938 | 0.146 | 0.832 | 0.269 | **0.345** | **0.302** | 0.242 | 0.255 | 0.869 | 0.183 |
| 3 | User Drop                   | 13.098 | 0.888 | 0.962 | 0.156 | 0.626 | **0.450** | 0.170 | 0.247 | 0.435 | 0.572 | 0.640 | 0.193 |
| 4 | User Drop + Scale (d256/L8)  | 13.455 | 0.883 | 0.951 | 0.153 | 0.643 | 0.359 | 0.193 | 0.251 | 0.313 | 0.563 | 0.661 | 0.182 |
| 5 | User Drop + Behavior Focal   | 12.792 | **0.895** | **0.973** | **0.157** | 0.173 | 0.235 | 0.298 | 0.263 | **0.766** | **0.687** | 0.145 | **0.195** |
| 6 | User Drop + Beh + SID(focal/wtd) | **11.186** | 0.893 | 0.962 | **0.157** | 0.267 | 0.216 | 0.242 | 0.228 | 0.638 | 0.679 | 0.249 | 0.188 |

Key reads:
- **t1 (buy SID retrieval) is already ~0.96 everywhere** and barely moves — the model is good at
  SID-given-behavior. **joint_Rec is stuck at ~0.18–0.20 across ALL runs** → the ceiling is NOT
  the loss/data balance; it's elsewhere (SID quality / collisions, or inherent difficulty). This
  is why the codebook diagnostic (§11) is the next priority.
- **Scale-up (#4) did not help** — worst loss, no metric win. Confirms the model wasn't capacity-bound.
- **Behavior focal (#5, γ=2, no α) overcorrects**: minority recall soars (cart 0.77, fav 0.69,
  buy 0.30) but pv recall collapses 0.91→0.145 and behavior accuracy collapses 0.86→0.17 (because
  pv is 93% of samples). Best t1 and best joint, but the behavior head is now badly miscalibrated.
- **Best buy F1 came from data rebalancing (#2 Seq Down, 0.302), not focal** — and #2 keeps pv
  recall (0.87) and behavior accuracy (0.83) intact. Best balance for buy so far.
- **Best buy precision: User Drop (#3, 0.45).** Trade-off: precision (user_drop) vs recall (seq_down).
- There is **no free lunch on behavior**: every method trades the pv majority for the rare classes,
  because the model must commit to one behavior per generation and pv is 93%.

Standing decision: current focal at γ=2 is **too aggressive**. Treat event_downsample (#2) as the
working baseline for buy.

### 12b. SID diagnostics (semantic_ids.pt, 5.41M items, widths [2048,1024,1024])

**Codebook health:** levels ≈100% active (2046/2048, 1024/1024, 1024/1024) — no collapse.
**Collisions:** 36.2% of items share a SID with another (4.16M unique tuples; largest group 152).
Acceptable — and since the evaluator matches **SID tuples, not item ids**, collisions don't cap the
metric (they only add ~36% training label noise). So the joint_Rec ceiling is **not** a SID-space
problem.

### 12c. Semantic near-miss probe (user_drop checkpoint, natural ~93% pv test)

| group | hit rate (1−miss) | pred_sim | rand_sim | ratio |
|-------|------|------|------|------|
| buy   | **96.2%** | 0.131 | 0.044 | 3.0× |
| cart  | **96.8%** | 0.220 | 0.081 | 2.7× |
| fav   | **94.5%** | 0.221 | 0.076 | 2.9× |
| pv    | 14.6% | 0.220 | 0.089 | 2.5× |
| overall | 19.6% | 0.220 | 0.089 | 2.5× |

Two conclusions, both important:
1. **Conversions are already solved on the SID side.** Free-gen top-k contains the GT item's SID
   for buy/cart/fav **94–97%** of the time (this is measured *without* conditioning on the correct
   behavior — behavior token is stripped, grouped by GT behavior). The 0.19 joint ceiling is purely
   **pv** being unpredictable (15% hit), and pv is ~93% of test. So SID/item prediction is **not**
   the bottleneck for what matters.
2. **Misses are near-misses, but modest.** `pred_sim` is 2.5–3× `rand_sim` everywhere → the model's
   wrong guesses are semantically related (not random), so the SID/encoder representations are
   healthy. But absolute `pred_sim ≈ 0.2` is modest → semantic retrieval/SRD would give a **moderate
   pv-only** bump, and ~nothing for buy (already 96% hit). SRD **deprioritized**.

**The real bottleneck = the behavior head.** The model predicts the right *item* for buy 96% of the
time but mislabels the *behavior* (defaults to pv), which is what tanks t3_buy_R and joint_buy. SID
and behavior are nearly **decoupled** in the model's predictions. → invest in **calibrated behavior
focal** (`focal_gamma` ↓ from 2.0, optional `focal_alpha`), not encoder/SID changes.

### 12d. Behavior-focal γ sweep (user_drop data)

| γ | buy_F1 | buy_R | buy_P | pv_R | BehAcc | joint_R@10 |
|---|--------|-------|-------|------|--------|-----------|
| 0 (plain CE) | 0.207 | 0.141 | 0.388 | 0.907 | 0.864 | 0.186 |
| **1** | **0.341** | **0.390** | 0.304 | 0.346 | 0.361 | 0.195 |
| 2 | 0.263 | 0.298 | 0.235 | 0.145 | 0.173 | 0.195 |

γ=1 **dominates** γ=2 (better buy P/R/F1 AND pv) and has the best buy_F1 overall (buy_F1 is concave
in γ, peak ≈1). Two standing facts: (1) **test-wide joint_Rec is flat (~0.19) and will NOT move**
from behavior focal — it's pv-dominated and we're trading pv↔buy; track `t3_behavior_*_buy`, not
joint_Rec. (2) At γ=1 **cart/fav are massively over-predicted** (recall ~0.64/0.68 but precision
~0.03/0.06) — wasted capacity; the next lever is selective `focal_alpha` (boost buy, damp cart/fav).

α sweep at γ=1 (indexed [pv,fav,cart,buy]): α_buy=1 → buy_F1 **0.341** (best); α_buy=1.3 → 0.303;
α_buy=2 → 0.292 (recall 0.53, precision 0.20). Raising α_buy just trades precision for recall →
buy_F1 falls. **cart/fav precision ≈ their base rates in every run = unlearnable behaviors; only
buy responds.** Conclusion: **behavior focal is exhausted** (buy_F1 ceiling ≈0.34, joint flat).

### 12e. Two-View Semantic Alignment + Retrieval (implemented, NOT yet run)

Rationale from §12c: misses are **near-misses** (pred_sim 2.5–3× random) but modest, and the
aggregate joint_Rec (pv-dominated, stuck 0.19) is the only thing left to move — buy SID is solved,
behavior focal is exhausted. Encoder-input 2-view (BMDV) failed and isn't the bottleneck; this puts
the 2-view on the **output/retrieval** side instead (the one pipeline stage never touched).

Plan (both halves needed; see §6 "Two-View Semantic Alignment + Retrieval" for the switches):
1. **Train** two heads to predict next item's `(v_t=e_f+e_t, v_i=e_f+e_i)` from the decoder
   item-summary hidden state; MSE to stop-grad target (`use_semantic_align`, `semantic_align_weight`).
2. **Infer** widen `top_k_for_generation` to ~50 and **rerank** the pool by cosine of predicted
   `(q_t,q_i)` to candidates' `(v_t,v_i)`, blended with decoder score (`use_semantic_rerank`,
   `rerank_weight`). Evaluator takes top-10 by the blended score.

Experiment recipe (isolates the rerank effect):
- Train: `use_semantic_align=true top_k_for_generation=50` (+ existing best data/loss config).
- Compare at test: `use_semantic_rerank=false` (wide-pool baseline) vs `=true` (rerank), same pool.
- Read `test/t3_recall_10` (aggregate joint) and `test/t2_recall_10`. Sweep `rerank_weight` (0.3–1.0).
- If the rerank pool ceiling bites (GT not in top-50), escalate to dense catalog retrieval; if MSE
  align is too weak, upgrade to a contrastive loss (in-batch negatives).
Honest expectation: near-miss signal is real but modest (pred_sim ≈0.19) → a meaningful nudge on the
aggregate metric, not a jump. A null result (rerank ≈ baseline) is itself informative (misses not
recoverable from the pool → the ceiling is genuine pv unpredictability, not scoring).

**Result — MSE align + rerank (userdrop, pool=50):**
- wide-pool baseline (align on, rerank off): t1_Rec 0.940, **t3_Rec@10 0.207** (pool 50 lifts joint
  from ~0.19 → 0.207, as expected; this is the reference).
- rerank w=0.5: t1_Rec **0.759**, t3_Rec@10 **0.175** — rerank **HURT**. Probe shows GT is in the
  50-pool 98.6% of the time but the rerank pushes it out of the top-10. Not a plumbing bug (query
  hidden state train↔infer consistent, table/broadcast/direction checked); the semantic query is
  **weaker than the decoder ranking**, and min-max blend at 0.5 injects that weak signal over a good
  order. Root causes: (a) views dominated by shared `e_f` → cosine barely discriminates; (b) **MSE
  trains proximity, not ranking** — never optimized what rerank needs.

**Retry — contrastive align (`align_loss_type=contrastive`).** InfoNCE trains the query to rank GT
above in-batch negatives (cosine/temp, false-neg SID mask) — matches the rerank geometry and forces
discriminative structure. Scripts: `train_tiger_mb_twoview_contrastive_{baseline,rerank}.sbatch`
(matched pair, `semantic_align_weight=0.5` so InfoNCE doesn't swamp the SID CE). Compare rerank vs
its OWN contrastive baseline. If contrastive rerank still ≤ baseline → the decoder ranking is the
ceiling for pool reranking; escalate to dense catalog retrieval or accept pv unpredictability.

### 12f. SID<->behavior coupling probe (two go/no-go gates)

`rec-tmall/analyze_sid_behavior_coupling.py` (sbatch `analyze_sid_coupling_{mi,asym}.sbatch`).
Fit on userdrop **training**, evaluate on **eval** under the natural ~93.6% pv prior (prior-shift
correction active: train pv 80.0% -> eval 93.75%). Coverage-skip **0.000%** both parts (item sets
match: 5,409,793 valid items). H(B) on eval = 0.4293 bits (behavior is very low-entropy).

**Part A — is behavior predictable from the SID beyond previous behavior?  -> GREEN.**
- MI (eval, bits / MI÷H(B)): `ID_shared` 0.0176/0.041, `ID_text` 0.0090/0.021, `ID_img` 0.0096/0.022,
  **`b_prev` 0.0150/0.035** (reference). The shared SID carries *more* marginal behavior signal than
  previous behavior — small absolutely, but real.
- Buy lift (shared codes, >=200 train occ): **718 codes at >2x** buy-lift cover 7.6% of eval traffic
  (mean buy rate 10.4% vs 3.36% base); only 1 code >5x.
- Held-out buy detection (max-F1, prior-corrected): (i) `P(b|b_prev)` **0.197**, (ii)
  `P(b|shared,b_prev)` **0.253**, (iii) `P(b|ID_shared)` alone 0.095 (weak standalone -> the signal is
  *complementary* to b_prev, not standalone).
- **Gate: Δbuy_F1 (ii)-(i) = +0.056 ≥ 0.05 -> GREEN** for Step 2 (SID-conditioned behavior head).
  Nuance: (ii)=0.253 is still *below* the model's focal buy_F1 ceiling (~0.34), so a lookup table
  does NOT beat the learned head. But behavior is generated *before* the SID in the current order,
  so the head never sees the target SID; the +0.056 says a **reorder (2a: SID-then-behavior)** has
  genuine signal to exploit that the current architecture cannot.

**Part A.4 — modality-additivity of behavior MI (held-out predictive MI).** Plug-in MI on the
2048x1024x1024 joint would overfit (~every triple unique), so fit `P(B|code)` on train (Dirichlet
a=1, >=50 support, prior-corrected, backoff->shared) and score cross-entropy on eval; pred-MI =
H(B)-CE. Positive & honest (held-out can't inflate):

| code set | pred-MI (bits) | Δ vs shared |
|---|---|---|
| shared | 0.0146 | - |
| shared + text | 0.0165 | +0.0020 |
| shared + img | 0.0167 | +0.0021 |
| shared + text + img | 0.0176 | **+0.0030** |

**Count-level proof that the text/image residuals carry behavior info the shared code lacks: +0.0030
bits = +21% over shared.** Text and image each add ~equally and are partly redundant (sub-additive:
0.0020+0.0021 > 0.0030). Absolute is small (0.003 of H(B)=0.43 bits ≈ 0.7%) but real — the paper
sentence for "modality decomposition adds behavior signal," established before any model is trained.

**Part A.5 — repeat-evidence value (attribution: repeat feature vs SID).** Per transition, the
target item's prior-interaction 4-bit mask (which behaviors it had earlier in the sequence). P(buy |
category) on eval:

| prior interaction of next item | n(eval) | P(buy) | lift |
|---|---|---|---|
| not in history | 4.07M | 0.01% | 0.0 |
| pv only | 1.07M | 1.36% | 1.3 |
| has fav (no cart) | 129k | 1.66% | 1.6 |
| **has cart** | 263k | **15.91%** | **15.1** |
| has buy | 62k | 3.09% | 2.9 |

The **cart->buy funnel dominates**: a previously-carted item is bought 15.9% of the time (15x base);
a never-seen item is essentially never bought (0.01%). Buy-detection max-F1 attribution ladder
(train-fit tables, prior-corrected): (i) `b_prev` 0.197 -> (ii) `+ID_shared` 0.253 (+0.056) -> (b1)
`+repeat-set` 0.376 (**+0.179**) -> (b2) `+repeat-set+ID_shared` **0.416** (+0.040 on top of b1).
**The repeat feature is ~3x the SID's value (+0.179 vs +0.056)**, and SID's marginal add shrinks to
+0.040 once repeat is present (partly redundant). The full count table (0.416) **exceeds the learned
model's focal buy_F1 ceiling (~0.34)** — the model is leaving the repeat signal on the table. (Caveat:
F1 threshold is picked on eval -> mild optimism on absolutes; the deltas/ranking are the honest read.)

**Part B — per-behavior modality asymmetry (text-view vs image-view ranking)?  -> RED.**
Transitions: 851,753 (pv capped 500k; fav 126k, cart 166k, buy 60k). **Repeat share is extreme for
buy: 92%** (54,980/59,614) — Tmall's re-click-then-buy loop — vs cart 42%, fav 28%, pv 17%.
R@10 vs an 8192-negative pool, Δ = text - img:

| behavior | R@10 text | R@10 img | Δ(t-i) | 95% CI | shared-only |
|---|---|---|---|---|---|
| pv (non-repeat, last) | 0.223 | 0.219 | +0.004 | [+0.003,+0.004] | 0.172 |
| fav | 0.909 | 0.909 | -0.000 | [-0.000,+0.000] | 0.719 |
| cart | 0.941 | 0.941 | -0.000 | [-0.000,+0.000] | 0.763 |
| buy | 0.911 | 0.911 | -0.001 | [-0.002,+0.001] | 0.741 |

Text beats image by a hair, but **uniformly across all behaviors** (no buy-specific text advantage).
Mean-of-5 history shows larger text advantages (cart +0.019, fav +0.015) but again the *same ordering
for every behavior*. Interaction **Δ_buy - Δ_pv = -0.43 pts, 95% CI [-0.58, -0.28]** — excludes 0 but
is negative and |gap| < 2 pts.
- **Gate: RED** — behavior-gated BM-TV is not licensed; buy does not prefer the text view more than pv
  does. This **retroactively explains the flat ungated BM-TV result with data**, not speculation.
- Bonus (e_f-dominance test): shared-only R@10 is well *below* the full-view average (buy 0.741 vs
  0.911; pv 0.172 vs 0.221) -> **residuals add real ranking signal**; e_f does NOT dominate the
  ranking geometry (so §12e's rerank failure is not purely e_f dominance).

**Net (updated with A.4/A.5):** the biggest behavior win is an **explicit repeat / prior-behavior-set
feature** (+0.179 buy_F1; the cart->buy funnel), not the SID — prioritize adding it (a count table with
it already beats the model's 0.34). The SID-conditioned behavior head (Step 2 reorder) is a real but
**secondary** gain (~+0.040 on top of repeat; A.4 confirms modality residuals do carry +21% behavior
info). Shelve BM-TV / behavior-gating (Part B RED). Honest ordering: repeat feature >> SID-head reorder
>> modality decomposition; do not over-credit the tokenizer/architecture for a win the repeat feature
supplies.

## 13. MBGen tokenizer baseline (alternative SID, not GRID's MD-RQ-VAE)

Goal: a clean **tokenizer ablation** — feed the *same* multi-behavior TIGER the semantic IDs
produced by MMMB-Genrec's tokenizer (referred to as **MBGen** here) instead of GRID's 3-level
MD-RQ-VAE SIDs. Identical data, identical T5, only the SID assignment changes.

**MBGen SID scheme (3 tokens/item), faithfully re-implemented in `src/data/generate_mbgen_sid.py`:**
- **Level 1 — QAE:** MLP encoder (2304→2048→1024→512→256→32) → single **EMA-VQ codebook**
  (`num_id1=2048`, kmeans-init, latent 32) → MLP decoder. Loss = MSE(recon) + `beta`·commit.
  Nearest code = `id1`; `residual = encoded − quantized`.
- **Level 2 — conditional KMeans:** group items by `id1`; run KMeans(`num_id2=2048`) on the
  residuals **within each level-1 group** (local codes, `expand_id2=false`). Guard: groups with
  ≤ num_id2 items get a degenerate one-cluster-per-item assignment (sklearn can't do k>n).
- **Level 3 — dedup counter (`add_dedup_token=true`, MBGen's UNIQUENESS mechanism):** a running
  index over items sharing the same `(id1,id2)` bucket → guarantees a distinct 3-token SID per
  item (a TIGER-style dedup token; see MBGen `QAE_Kmeans_item_Tokenizer.__init__`). Level-3 vocab
  = largest bucket (printed at end of run). This is why the QAE codebook collapsing to <2048
  active codes is NOT a uniqueness problem in MBGen — the 3rd token resolves all collisions; a
  weaker QAE just makes buckets (and the dedup vocab) larger. The generator prints the exact
  `num_hierarchies=3 codebook_sizes=[2048, <max id2+1>, <max bucket+1>]` line to copy into the
  training run. Dead-code revival is NOT MBGen's mechanism and is intentionally absent.

**Deliberate deviations from MBGen upstream (all scale-forced):** input = concat(SigLIP text,
image)=2304-d standardised (MBGen uses one opaque `embedding.pkl`); codebook kmeans-init on a
300k subsample; MiniBatchKMeans instead of full KMeans. Codebook sizes **2048×2048** chosen to
match GRID's ~4.19M unique-SID capacity (vs MBGen's tiny 96×96, tuned for ~10k-item catalogs).

**Structural caveat:** a 2-token scheme has a hard collision ceiling on 5.4M items — max unique ≈
Σ_group min(nₘ, num_id2). With avg group ≈2640 > 2048, expect collision in the same ballpark as
GRID's 36% (that's the point of capacity-matching). Pure MBGen sizes would be ~99% collision.

**Output:** `[2, N]` int64 tensor, column == item_id, gaps/modality-missing items = `[0,0]`
(exact layout of the MD-RQ-VAE `semantic_ids.pt`; column indexing verified against
`map_sparse_id_to_semantic_id`). Item set = text∩image id-map intersection (the multimodal
MD-RQ-VAE required both modalities too, so this covers the same ~5.4M items). Written to
`/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/{semantic_ids.pt,mbgen_sid_artifacts.pt}`.

**Files:**
- `src/data/generate_mbgen_sid.py` — the tokenizer (standalone; not hydra/lightning).
- `configs/experiment/mbgen_sid_gen.yaml` — all knobs (read via OmegaConf; CLI dotlist overrides).
- `sbatchfiles/generate_mbgen_sid.sbatch` — runs the generator (A100, 300GB RAM, ~50GB matrix).
- `sbatchfiles/train_tiger_mb_mbgen_baseline.sbatch` — trains tiger_mb on the MBGen SIDs.

**How to run (order matters):**
1. `sbatch sbatchfiles/generate_mbgen_sid.sbatch` → produces `semantic_ids.pt`; check the
   VERIFICATION block in `expoutfile/generate_mbgen_sid.out` (active codes, collision rate).
2. `sbatch sbatchfiles/train_tiger_mb_mbgen_baseline.sbatch` — uses `num_hierarchies=3`
   (id1, id2, dedup), `codebook_sizes` from the generator printout, all GRID extensions OFF
   (BM-TV/two-view/probe need the MD-RQ-VAE geometry; codebook-init needs its ckpt). Compare
   `test/t*_recall_*` vs the MD-RQ-VAE runs. This uses the plain **tiger** model on MBGen SIDs.

## 14. MBGen PBA model (Position & Behavior Aware architecture)

Goal: the complementary ablation to §13 — same SIDs, but MBGen's **model** instead of GRID's
tiger. Isolates the architecture. Faithful port of `MMMB-Genrec/model/PBA_transformer.py`.

**What PBA is** (two ideas, both FFN-level; see the module docstring):
- **Position-routed experts (deterministic MoE):** each within-item token position gets its
  own FFN expert — expert 0 = special (user/BOS/pad), 1 = behavior token, 2..(1+H) = the H SID
  levels. Routing is by *position*, not a learned gate ⇒ no router z-loss.
- **Behavior injection:** on selected layers, the item's behavior embedding is concatenated
  onto the FFN input, conditioning every SID token on its behavior.

**How it's built (`src/models/modules/semantic_id/pba_generation_model.py`):** subclass of
`SemanticIDMultiBehaviorEncoderDecoder` that swaps each T5 `T5LayerFF` for `PBAFeedForward`
(position-routed experts + optional behavior injection). Everything else — SID embedding
tables, the (1+H) flat sequence, beam search, `model_step`, `MultiBehaviorSIDRetrievalEvaluator`
— is **inherited unchanged**, so `test/t1_* / t2_* / t3_*` are directly comparable to the tiger
runs. The T5 block FFN can't see token positions, so the parent computes the per-token
position/behavior indices (matching MBGen's PBAEncoder/DecoderRouter) and stashes them on every
`PBAFeedForward` before each backbone call (`_encoder_indices` / `_decoder_indices`).

**Faithfulness / deliberate adaptations:** ideas ported exactly (position experts, behavior
injection, MBGen ijcai layer config — encoder dense, all 4 decoder layers sparse, inject on
layers 0-1, `behavior_embedding_dim=64`, `d_model=256`, `d_ff=512`). Adaptations forced by
GRID integration: T5 backbone instead of HF SwitchTransformers (same encoder-decoder + MoE-FFN,
different plumbing); no SEP token / no user token (GRID `num_user_bins=null`, matching the tiger
baseline — position routing carries the structure); `shared_expert` and MBGen's own generation
loop are not used (GRID's beam search is reused for metric comparability). DDP-safe with plain
`ddp`: the training decoder always runs the full [BOS, beh, sid0..H] block, so every decoder
expert receives tokens every step (no unused params); keep `sparse_layers_encoder=[]` (encoder
expert 0 would otherwise see only masked pad tokens and go unused).

**Files:** `pba_generation_model.py`, `configs/experiment/tiger_mb_pba_flat.yaml`
(model=`PBAMultiBehaviorEncoderDecoder`, `mlp_layers: null`, `codebook_sizes: null`, PBA params),
`sbatchfiles/train_tiger_mb_pba.sbatch` (PBA on MBGen SID) and
`sbatchfiles/train_tiger_mb_pba_mdrqvae.sbatch` (PBA on the MD-RQ-VAE 2048_1024 SID). The same
yaml drives both; only `semantic_id_path` + `codebook_sizes` differ (CLI overrides).

**The 2x2 (model x tokenizer), all sharing the same evaluator/metrics:**
| | MD-RQ-VAE SID | MBGen SID |
|---|---|---|
| **tiger model** | existing baseline | §13 `train_tiger_mb_mbgen_baseline.sbatch` |
| **PBA model** | `train_tiger_mb_pba_mdrqvae.sbatch` | §14 `train_tiger_mb_pba.sbatch` |

Reading a row isolates the tokenizer; reading a column isolates the architecture.

**Run:** `sbatch sbatchfiles/train_tiger_mb_pba_mdrqvae.sbatch` runs immediately (MD-RQ-VAE SID
already exists); `train_tiger_mb_pba.sbatch` needs the regenerated 3-level MBGen SID first.

**Results (test, userdrop, top_k_for_generation=10):**
| metric | PBA + MBGen SID (3-lvl, dedup) | PBA + MD-RQ-VAE SID (2048_1024) |
|---|---|---|
| t3_recall_10 (joint) | 0.1976 | 0.1971 |
| t3_recall_5 | 0.1885 | 0.1900 |
| t3_ndcg_10 | 0.1558 | 0.1566 |
| t2_recall_10 (beh-conditioned) | 0.1569 | 0.1574 |
| t1_recall_10 | 0.9714 | 0.9689 |
| t1_recall_5 | 0.9050 | 0.9176 |
| t3_behavior_accuracy | 0.826 | 0.752 |
| t3_behavior_f1_buy | 0.308 | 0.312 |
| t3_behavior_f1_pv | 0.904 | 0.857 |
| test/loss | 12.25 | 12.76 |

**Verdict — PBA does not help; tokenizer swap does not help.** All retrieval metrics are tied to
the 3rd decimal and sit at the same **~0.197 joint ceiling** as every tiger run. `t2`
(conditioned on GT behavior, isolates SID retrieval) is identical (0.1569 vs 0.1574) → the core
retrieval capability is unchanged by architecture or tokenizer. The only real difference is the
behavior head's operating point: the MBGen run is more pv-skewed (higher pv recall/accuracy,
lower cart/fav recall) but **buy_f1 is identical (0.308 vs 0.312)** — a precision/recall bias
shift that nets to zero on the pv-dominated joint, not a capability gain. The collision caveat
does NOT manifest: MD (36% collision) is not higher on t3 than MBGen (0% collision, dedup token).
Conclusion: PBA's FFN-level changes (position experts + behavior injection) don't touch the
bottleneck, which is pv-dominance (~93% of test) + pv next-item unpredictability. Both the model
axis (two-view rerank §12e, PBA §14) and the tokenizer axis (§13) converge on ~0.197 → the ceiling
is a problem-definition/data issue (aggregate joint measures mostly pv), not architecture/tokenizer.
## 15. MD-RQ-VAE — the core tokenizer (★ primary research contribution)

The **Multi-modal Dual Residual Quantized VAE** (`src/modules/clustering/md_rqvae.py`,
`class MDRQVAE`) is the heart of this project: it turns each item's paired SigLIP text + image
embeddings into a 3-token semantic ID whose levels are **modality-decomposed and additive**.
Every downstream idea (BM-TV encoder §6, Two-View align/rerank §12e, the near-miss probe §12c)
depends on the additive property this tokenizer is designed to produce. Trained via
`configs/experiment/md_rqvae_train_flat.yaml`; SIDs extracted via `md_rqvae_inference_flat.yaml`.

### 15.1 Inputs & preprocessing
- Raw features: SigLIP2-so400m-patch14-384 **text** `x_t` (1152-d) and **image** `x_i` (1152-d),
  one pair per item (only items with BOTH modalities are tokenized).
- Per modality: **L2-normalise then LayerNorm** (`text_input_norm`, `image_input_norm`) — removes
  magnitude variation, then rescales to a learned distribution on the unit sphere.

### 15.2 Architecture / forward path (latent_dim d = 512)
```
z_t = E_t(x_t),   z_i = E_i(x_i)                 # encoders: MLP 1152 -> 768 -> 256 -> 512
z_f = F(z_t, z_i)                                # CrossAttentionFusion -> shared "commonality" z_f
# ---- Level 1: SHARED (commonality) ----
id_shared, e_f1 ~ C_f(z_f)                       # shared codebook  C_f (2048 entries)
r_f = z_f - e_f1.detach()                        # residual w.r.t. the HARD codebook vector
# ---- Level 2: TEXT-specific ----
v_t = text_fusion_mlp( concat[r_f, z_t] )        # 1024 -> 512 ; residual + text latent
id_text,  e_t2 ~ C_t(v_t)                        # text codebook   C_t (1024 entries)
# ---- Level 3: IMAGE-specific ----
v_i = image_fusion_mlp( concat[r_f, z_i] )       # 1024 -> 512 ; residual + image latent
id_img,   e_i3 ~ C_i(v_i)                        # image codebook  C_i (1024 entries)
# ---- Reconstruction (this is what forces the additive property) ----
x_t_hat = D_t( e_f1 + e_t2 )                     # decoders: MLP 512 -> 256 -> 768 -> 1152
x_i_hat = D_i( e_f1 + e_i3 )
```
- **CrossAttentionFusion** `F`: bidirectional cross-attention — text queries image context and
  image queries text context (`t2i_attn`, `i2t_attn`, 4 heads), each residual-added + LayerNormed,
  concatenated and projected `2d -> d`. Produces the commonality embedding `z_f`.
- **Codebooks** `C_f / C_t / C_i`: `EMAVectorQuantization` (EMA decay 0.99, dead-code threshold
  1.0, squared-Euclidean distance, **STE** straight-through, **KMeans++** init, init_buffer 3072).
  Sizes **[2048, 1024, 1024]** for the `2048_1024` run.
- `e_f1` is **detached** when forming `r_f`, so the residual is taken against the hard code; `z_f`
  still receives gradient from the downstream text/image commitment terms (extra regularisation).

### 15.3 The additive property (why this design)
Because the decoders reconstruct each modality from a **sum**:
`x_t_hat = D_t(e_f1 + e_t2)` and `x_i_hat = D_i(e_f1 + e_i3)`, training drives
**`e_f + e_t ≈ text latent`** and **`e_f + e_i ≈ image latent`**. The shared code carries the
cross-modal commonality; each modality residual adds only what is modality-specific. This is the
exact structure exploited by BM-TV (V_shared=e_f, V_text=e_t, V_img=e_i), by the Two-View
align/rerank (`v_t=e_f+e_t`, `v_i=e_f+e_i`), and by the near-miss probe. It is the property MBGen's
tokenizer (single fused embedding, non-decomposed levels) does NOT have.

### 15.4 Loss
`L_total = L_rec + alpha * L_aux + beta * L_vq`  (alpha = beta = 1)
- `L_rec = MSE(x_t, x_t_hat) + MSE(x_i, x_i_hat)` — reconstruct both modalities from the summed codes.
- `L_aux = InfoNCE(z_f, z_t) + InfoNCE(z_f, z_i)` — in-batch, **learnable temperature**; pulls the
  shared `z_f` toward BOTH modality latents so it genuinely captures commonality (not one modality).
- `L_vq` = sum of VQ + commitment losses over `{C_f, C_t, C_i}` (commitment weight in each codebook's
  `loss_function.beta`). Codebooks are updated by EMA, not gradient.

### 15.5 Output SID
3 tokens per item `[id_shared, id_text, id_img]`, each 0-indexed into its codebook; widths
`[2048, 1024, 1024]`. `predict_step` writes `item_id -> (3 ids)`, merged/deduped/transposed to the
`[3, N]` `semantic_ids.pt` (column == item_id; see §13 and `verify_md_sid_2048_1024.out`).
Quality of the `2048_1024` run: all three codebooks ~100% active, per-level entropy 95-99% of max,
levels near-independent, **collision 36.2%** (4.16M unique / 5.41M items) — acceptable (TIGER-class).

## 16. MFD — Modality-Factorized Decoding (decoder ablation on the MD-RQ-VAE SID)

Goal: spend the §12g "levels near-independent" license structurally. The chain decoder generates
`P(b,s|x)=P(b|x)·P(f|x,b)·P(t|x,b,f)·P(i|x,b,f,t)`; MFD **drops the last edge** (`t->i`), the
one §12g's CMI(t;i|f) says carries ~1.7% of the info:
`P(b,s|x)=P(b|x)·P(f|x,b)·P(t|x,b,f)·P(i|x,b,f)`. Text and image are predicted by **two parallel
heads reading the SAME `h_f`** (never fed as inputs), so the teacher-forced decoder is 3 positions
`[BOS, behavior, f]` instead of 4. Everything else — MD-RQ-VAE SID, T5 backbone (d_model=128, 6
heads, d_ff=1024, 4 layers, mlp_layers=2), data pipeline (`next_k` stays 4; MFD slices the first
two label columns internally), evaluator, focal/behavior-weighting — is inherited **unchanged**, so
this is a clean isolation of the decoder factorization vs the chain baseline in §14.

**Files:** model `src/models/modules/semantic_id/mfd_generation_model.py`
(`class MFDMultiBehaviorEncoderDecoder(SemanticIDMultiBehaviorEncoderDecoder)` — overrides only
`model_step`, `generate_multibehavior`, `generate_conditioned`); config
`configs/experiment/tiger_mb_mfd_flat.yaml` (byte-for-byte copy of `tiger_mb_train_flat.yaml`
except the model target + MFD knobs); sbatch `sbatchfiles/train_tiger_mb_mfd.sbatch`.

**Loss** = CE_beh (focal-ready) + CE_f + CE_t|f + CE_i|f; the two modality CEs backprop into the
same `h_f` (shared-state contention — this is deliberate). **Generation** = compositional beam with
agreement scoring (§5 of the spec): beam over `(behavior,f)` → K_f roots; one forward to `h_f`,
read BOTH heads → top-K_t text × top-K_i image; **catalog-filter** each `(f,t,i)` against the 4.16M
valid tuples in `self.codebooks` (replaces the chain's validity-by-construction); score
`logP(f) + (1/τ)[logP(t|f)+logP(i|f)]`; global top-K_pool. Defaults K_f=K_t=K_i=10 (→ **1000**
explored per item), **K_pool=50**, τ=1.0 (exact product-of-experts). τ is frozen/inference-swept by
design so the training path == the chain baseline's; ablation knobs `mfd_tau`, `mfd_score_text/image`
(agreement vs single-branch), `mfd_top_k_text/image` (asymmetry), `mfd_catalog_constraint`.

**Results (test, userdrop, MD-RQ-VAE 2048_1024 SID). Chain = §14 PBA+MD-RQ-VAE.**
| metric | chain (§14, pool=10) | **MFD (τ=1, pool=50)** | Δ |
|---|---|---|---|
| t3_recall_10 (free-gen joint) | 0.1971 | **0.1899** | **−0.0072** |
| t3_recall_5 | 0.1900 | 0.1812 | −0.0088 |
| t3_ndcg_10 | 0.1566 | 0.1473 | −0.0093 |
| **t2_recall_10** (beh-conditioned SID) | 0.1574 | **0.1996** | **+0.0422** |
| t2_recall_5 | — | 0.1943 | (big) |
| t1_recall_10 (buy SID) | 0.9689 | 0.9403 | −0.0286 |
| t1_recall_5 | 0.9176 | 0.8932 | −0.0244 |
| t3_behavior_accuracy | 0.752 | **0.552** | **−0.200** |
| t3_behavior_f1_buy | 0.312 | 0.296 | −0.016 |
| t3_behavior_f1_pv | 0.857 | 0.704 | −0.153 |
| test/loss | 12.76 | 13.46 | +0.70 |

**The headline is NOT the −1% on t3 — it is the split:**
- **t2 (pure SID retrieval, behavior given) JUMPS +27% relative** (0.157→0.200). Given the behavior,
  the factorized compositional beam retrieves the next-item SID far better than the chain's beam.
  This is the promising signal and the reason to keep exploring MFD.
- **t3 (free-gen joint) drops −3.7% rel**, driven almost entirely by **behavior accuracy collapsing
  0.752→0.552** (pv_f1 0.857→0.704). The SID pool got better but the *behavior chosen for the top
  beam* got worse, and the joint metric multiplies the two.
- **t1 (exact buy tuple) drops −3%** — consistent with §12g: dropping the `t->i` edge loses a little
  of the fine joint structure that the chain nails on repeat-heavy buy items.

**Diagnosis of the t3/behavior drop.** In free generation the top beam's behavior is `argmax` of the
composed joint score over a **5× larger, SID-diverse pool** (1000 explored / 50 kept vs the chain's
10). A rare-behavior item with a very sharp `(f,t,i)` can outscore the marginally-more-likely pv
item, so maximizing joint prob over a bigger pool naturally depresses behavior accuracy. **This is
plausibly a pool-size artifact, not intrinsic to factorization** — it is confounded because the
chain baseline ran at pool=10. ⚠ **The comparison is not yet apples-to-apples: the chain baseline
must be re-run at `top_k_for_generation=50` before attributing the t2 gain or the t3/behavior loss
to factorization.** The empirical t3 hit (~1%) is, however, consistent in magnitude with §12g's
predicted ~1.7% information loss from the removed edge.

**Verdict — keep exploring.** MFD is not a flat loss: it materially improves conditioned SID
retrieval (t2) and only loses on the behavior-selection step of free generation, which is the most
fixable part. Improvement directions (evidence-ranked) recorded in `sbatchfiles/train_tiger_mb_mfd.sbatch`
header and the chat log: (0) pool-matched chain baseline; (1) decouple behavior selection from SID
sharpness / behavior-score weight to recover t3; (2) joint-rerank tail to recover the §12g 1.7% on
t1/t3 while keeping the t2-winning pool; (3) τ sweep {0.7,1.5,2.0} (built-in); (4) train-time
agreement objective (remove the train/inference mismatch); (5) shared-`h_f` contention relief.

### 16.1 Behavior-marginalized pooling + the §0 diagnostic audit

**Behavior-marginalized pooling** (`mfd_behavior_marginalized=true`, default): free-gen no longer
commits one behavior in the beam — it runs the f-branch under ALL 4 behaviors, builds a per-behavior
pool (each behavior its own top-K_f f-roots), and ranks the union by `w_b·logP(b)+SID_score(.|b)`.
So the per-behavior retrieval that wins t2 runs inside t3. `w_b` (`mfd_behavior_weight`) trades
behavior-likelihood vs SID sharpness. **Result: at `w_b=1` it lifted t3_recall_10 0.1899→0.1938;
raising `w_b` HURTS the marginalized/union t3** (0.1884 at 2, 0.1682 at 4) because up-weighting
`logP(b)` lets the dominant pv branch crowd the ~7% non-pv events out of the top-10 (the entire drop
is the non-pv rows; pv stays flat). **So `w_b=1` is best for aggregate t3 here — this supersedes the
earlier "increase w_b" guidance, which was framed on the committed-beam behavior accuracy, not the
union t3.**

**§0 audit** — one instrumented test pass (`mfd_audit_dump=true`, wb=1 ckpt, 83k stratified events)
→ `mfd_audit_dump.pt` → three offline analyses (`analyze_mfd_audit.py`; sbatch `mfd_audit.sbatch`).

- **0a — pool routing (t1 collapse is a SCORING ARTIFACT).** Re-ranking the SAME dump three ways:
  `t1_recall@10` — union 0.973, **routed (GT-beh branch) 0.975**, committed 0.940; `@5` — routed
  **0.968** vs committed 0.893. `t2@10` routed 0.1987 ≈ committed 0.1996. `t3@10` union 0.1939.
  → The marginalized run's earlier "t1 drop to 0.909" was an artifact of scoring behavior-conditioned
  tracks against the 4-way union pool. **Standing readout: report t1/t2 from the ROUTED (GT-behavior)
  pool, t3 from the union.** Costs nothing, recovers ~8 pts of phantom t1.
- **0c — coverage vs ordering (COVERAGE is the binding constraint).** oracle coverage@pool = **0.2085**
  (this IS the t3 ceiling any reranker can reach; current t3@10 0.1939 → only +0.015 ordering headroom).
  Loss decomposition: **f_pruned=0.596** (60% of GT have the shared code f* outside the top-10 f-roots)
  ≫ branch_t 0.151 ≫ branch_i 0.030. By behavior: pv coverage only **0.158** (f_pruned 0.634) vs
  fav/cart/buy **0.96–0.98** — the model nails f for rare behaviors, misses it for pv. In-pool ordering
  is good (P(rank<10)=0.93). **Lever = widen K_f** (attacks f_pruned) — and this is the rare non-crowding
  scaling case: more f-roots add genuinely-new shared-code coverage, unlike widening the behavior union.
  - **catalog_killed=1.38% was a mislabel.** The original metric conflated a real catalog bug with pool
    truncation (GT valid+composed but ranked >Kpool=50). The catalog is built from the same
    `semantic_ids.pt` the labels use, so GT is valid by construction. Added a DIRECT `gt_in_catalog`
    flag + a printed unit test in the dump (`GT tuples NOT in catalog = N/total`), and the analysis now
    splits `cat_invalid` (real bug) vs `trunc` (ordering; does NOT affect recall@10). Re-dump to read
    the unit test; expect 0 invalid → the 1.38% is truncation, not a catalog rebuild.
- **0b — contention (PMQ weakly licensed).** Teacher-forced acc at GT f: MFD `acc(t|GT f)` top1
  **0.238** vs chain **0.244** (gap +0.6pt top1, +0.9pt top5); `acc(i|·)` MFD 0.254 vs chain 0.304
  (gap +5.1pt). Text has NO dropped edge (both predict t|f), so the text gap is **pure contention** from
  the two heads sharing one `h_f` → PMQ (per-modality query / relieve shared `h_f`) is licensed. The
  larger image gap = contention + the legitimate §12g dropped-edge (~1.7%). CAVEAT: the chain reference
  (`2026-06-30/19-35-17`) used codebook_init+focal vs MFD's random/no-focal, biasing toward
  over-detecting contention, and +0.6–0.9pt sits near the 0.5pt threshold → treat as **weakly** confirmed;
  a regime-matched chain would firm it up.

**Next (evidence-ordered): (1)** re-dump to run the catalog unit test (blocking sanity, expected pass);
**(2)** widen K_f — coverage sweep is offline in `analyze_mfd_audit.py` (from the full f* ranks), real
t3@10 vs K_f∈{10,20,50,100,200} via `sbatchfiles/mfd_kf_sweep.sbatch`; find the knee where pv coverage
converts to t3. K-widening here is endorsed by the audit (the biggest single t3 lever), unlike the
behavior-union widening that caused the earlier t1 artifact.

### 16.2 K_f sweep result — right direction, LOW ROOF

Real eval (`mfd_kf_sweep.sbatch`, wb=1, marginalized, K_t=K_i=10, frozen wb1 ckpt):

| K_f | t3_recall_10 | t3_recall_5 | t2_recall_10 | t1_recall_10 (union) | Δt3@10 vs prev |
|---|---|---|---|---|---|
| 10  | 0.1938 | 0.1827 | 0.1987 | 0.9089 | — |
| 20  | 0.1969 | 0.1843 | 0.2026 | 0.9083 | +0.0031 |
| 50  | 0.1993 | 0.1849 | 0.2059 | 0.9090 | +0.0024 |
| 100 | 0.2005 | 0.1850 | 0.2076 | 0.9082 | +0.0012 |
| 200 | 0.2010 | 0.1850 | 0.2085 | 0.9087 | +0.0005 |

**Findings.** (a) Widening K_f works exactly as 0c predicted — monotonic t3 gain, **knee at K_f≈50**
(captures 0.0055 of the total 0.0072), fully saturated by K_f=100-200. (b) At K_f≥50 MFD marginalized
now **edges past the chain baseline** (chain t3@10=0.1971; MFD K_f=50 **0.1993**, K_f=200 **0.2010**) —
the first clean t3 win for MFD, but only **+0.002–0.004**. (c) t2@10 climbs in lockstep (0.199→0.2085)
and t3@10 saturates just under it — i.e. t3 has converged onto the **coverage ceiling (oracle
0.2085)**; once f-coverage is spent, the residual loss is branch pruning (branch_t 0.151, the next
constraint) + behavior crowding, not ordering. (d) t3_recall_**5** barely moves (0.1827→0.1850): the
extra K_f candidates land at ranks 5–10, not the very top. (e) t1 (union) and behavior_accuracy are
K_f-invariant (0.909 / 0.534) as expected — K_f is purely a shared-code coverage knob.

**The low roof is structural, not a K_f problem.** MFD converges to **t3@10 ≈ 0.201**, essentially the
0.2085 coverage oracle, which sits at the same **~0.20 joint ceiling** every tiger/PBA run hits (§14) —
pv dominance (93%) + pv next-item unpredictability. K_f converted MFD's coverage debt into a small,
free, inference-only win over the chain, but did not break the ceiling. **Cost/benefit: use K_f=50**
(the knee — 5× the modality forwards for ~0.0055 of the 0.0072; K_f=200 is 20× for the last 0.0005).
Beyond K_f, the only remaining levers are the ones that lift the ceiling itself: the joint-rerank tail
/ PMQ (0b, ~0.015 ordering + contention headroom) and, structurally, the repeat-evidence behavior
feature (§12f A.5) — not more beam width.

### 16.3 PMQ / Query-f — learnable query slots (implemented; grid A/B/C)

Attacks the two measured MFD problems (0b head contention: acc(t|GT f) trails chain by 0.9pt with
t,i sharing h_f; 0c f-pruning: pv f_pruned 0.634, K_f only masks it, roof 0.201). Gives each
prediction its OWN decoder slot / hidden state / attention pass, keeping the factorization
P(f|hist,b)·P(t|hist,b,f)·P(i|hist,b,f) exact. Config flags (`MFDMultiBehaviorEncoderDecoder`):
`use_pmq`, `use_query_f`, `decoder_slot_type_emb` (zero-init identity signal, T5 has no absolute PE),
`mfd_warmstart_path` (load a prior ckpt strict=False; new query params start at init).

| run | use_pmq | use_query_f | block | launcher |
|---|---|---|---|---|
| A (ref) | false | false | [BOS, beh, f] | `train_tiger_mb_mfd.sbatch` (existing) |
| B | true | false | [BOS, beh, f, q_t/q_i] | `train_mfd_pmq.sbatch` |
| C | true | true | [BOS, beh, q_f, f, q_t/q_i] | `train_mfd_pmq_qf.sbatch` |

**Two implementation decisions that deviate from the original spec (both deliberate):**
1. **In-model, NOT collate/label.** GRID's MFD decoder is single-target-item (history lives in the
   untouched stride-4 encoder), so the query slots + mask are a within-block, model-internal concern.
   Building them in the model (not collate_functions / label_function / next_k) keeps the data pipeline
   byte-identical → **run A reproduces the old metrics exactly** and there's no encoder/label-alignment risk.
2. **Two separate passes for q_t and q_i, NOT one block with a custom q_i-/->q_t 4D mask.** Running
   each modality query as the last slot of its own pass makes them conditionally independent given f
   BY CONSTRUCTION and uses only STANDARD causal masking (attention_mask=None → T5's own causal, the
   path the current MFD already relies on) — avoids depending on unverified T5 3D/4D-mask semantics.
   Cost: one extra tiny modality forward under PMQ. q_f sits before f so causal masking alone stops it
   seeing f* (no mask edit needed).

Each launcher runs warm-start train+test (t1/t2/t3 + behavior) THEN an audit on the trained ckpt
(f_pruned / coverage@pool / teacher-forced acc(t|GT f) vs chain_tf_audit.pt). Decision metrics — B:
acc(t|GT f) top1/5 toward chain 0.2437/0.3412 (from 0.2377/0.3322); C: pv f_pruned down / coverage up
/ t3@10 above the 0.201 roof.

**Run-C shipped and is the FROZEN base for the reranker (§17).** Checkpoint
`/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt`
(launcher `train_mfd_pmq_qf.sbatch`). CMIR (cross-modal iterative refinement) was also built
(`use_cmir`, `train_mfd_cmir.sbatch`) and gives t2@10=0.2096 / t3@10=0.2024 — a hair below Run-C, so
the direction moved to a **second-stage reranker** rather than more decoder surgery.

---

## 17. ★ Two-stage listwise reranker — the FINAL result (t3@10 0.2041 → 0.2150, t2@10 → 0.2225)

The single biggest win after the tokenizer. A shallow **listwise cross-encoder** rescores the frozen
Run-C candidate pool; the generative model is **never retrained**. Files:
`src/.../mfd_generation_model.py` (Stage-1 dump), `rec-tmall/train_mfd_reranker.py` (Stages 2–3),
`rec-tmall/analyze_mfd_audit.py` (offline coverage math, reused verbatim for comparable metrics).

### 17.1 Why a second stage — the ceiling diagnosis (the §0 audit, §16.1)
The joint-t3 ceiling near 0.20 is **structural, not a beam-width bug**: recall@10 is K_pool-invariant,
and the §0 audit decomposes every miss into COVERAGE (GT absent from the pool) vs ORDERING (GT present
but ranked ≥10). At the operating point the GT is in the pool only **26.2%** of the time
(`coverage@pool`), yet **P(rank<10 | covered)=0.781** — i.e. an ordering headroom of ~+0.057 that no
K-widening touches (widening lands the extra GTs at rank ≥10). The hand-weighted log-prob sum
`w_b·logP(b)+logP(f)+(1/τ)[logP(t|f)+logP(i|f)]` simply cannot surface them → **learn the ordering.**

### 17.2 Architecture (three deliberate choices; earlier MSE/random-neg/pooled-query rerankers degraded recall — not reused)
- **Stage 1 — candidate dump** (`mfd_dump_memory=true`, an `mfd_audit_dump` pass on frozen Run-C).
  Per event stores: the per-behavior candidate pools `(f,t,i)`, their four component log-probs
  `(logP(b), logP(f), logP(t|f), logP(i|f))`, the **frozen encoder memory** (+mask), the raw history
  token sequence `hist_ids`, GT SID + `gt_in_catalog`. Self-contained → Stage 2 needs no T5.
- **Stage 2 — reranker** (`MFDReranker`, ~0.47M params, 2 layers, d_model=128). Each candidate token =
  three **MD-RQ-VAE view vectors** `[u_f, u_f+u_t, u_f+u_i]` (codebook centroids lifted through the
  orthogonal `W_up`; NO new item table) + the 4 log-probs as scalars, then it **cross-attends into the
  FULL encoder-memory sequence** (not a pooled summary) → one scalar per candidate. Zero-init head +
  `base_scale=1` ⇒ at step 0 the score == baseline `union_score` exactly (identity start).
  **Listwise cross-entropy** over the whole pool, GT positive, trained only on GT-in-pool events with
  the generator's own real-pool hard negatives.
- **Stage 3 — two-stage eval.** Offline rescoring of the dumped pool over the dumped memory is
  bit-identical to live "frozen generator → reranker → top-10". Metrics reuse
  `analyze_mfd_audit.best_rank/recall_at/union_score` verbatim (only the score changes) so numbers stay
  comparable: t3/t2 recall@{5,10,20}, NDCG@{5,10}, P(rank<10|cov), per-behavior t3@10 & t2@10.
- **Split hygiene:** TRAIN on the training split (~250k events → ~98k GT-in-pool, the axis-2 data
  scaling), SELECT on the evaluation split, REPORT on the testing split — **no test peeking**.

### 17.3 Engagement features — the win (`--use_hard` / `--use_soft`)
The reranker's strongest signal is **repeat interaction** (measured on this data: P(buy)=0.01% if the
item never appeared in history, 1.36% after a page-view, **15.91%** after add-to-cart; ~92% of buys are
repeats). Two per-candidate families (the ONLY change vs the views-only baseline is `scalar_proj`'s
input width — architecture otherwise identical):
- **HARD** — per behavior {pv,fav,cart,buy}: `[indicator, log1p(count), recency]` of history items with
  the candidate's EXACT `(f,t,i)` tuple (recency∈(0,1], most-recent history item=1.0).
- **SOFT** — per behavior × per view {u_f, u_f+u_t, u_f+u_i}: max cosine of the candidate's view vs
  that behavior's history items. Catches variant re-engagement (same product, different SKU/colorway)
  and is robust to the 36% SID collision; only constructible because the tokenizer gives additive
  shared/text/image views.

**Ablation (TEST t3@10, baseline union_score = 0.2041):** views 0.2084 · **+hard 0.2144** · +soft 0.2083
· **+both 0.2150**. Hard match drives the repeat-heavy behaviors; soft adds a little on top.
+both per-behavior t3@10: pv 0.1562→**0.1661**, fav →0.9471, cart →0.9607, buy →0.9429;
P(r<10|cov) 0.781→**0.822**. Model selected on VAL @ ep1.

### 17.4 Wide pools (G5) + K_pool — informational (negative) result
To raise the 0.262 coverage ceiling: re-dump at **K_t=K_i=1024** (= full head width → no t/i pruning).
The infeasible dense `K_f·K_t·K_i` grid (209M/behavior, 53GB tensor) is replaced by the
`mfd_wide_catalog_pool` path — score the **unique catalog tuples directly** in one vectorized pass
(`code2root` scatter + gather of each root's per-modality log-probs), ~same cost as the current dump.
Coverage@**grid** rises to ~0.756, but at **K_pool=500** coverage@**pool** stayed FLAT at 0.2618
(≈ G4's 0.2615) and t3@10 = 0.2146 — the top-500 by generator score is the same set regardless of
branch width; the newly branch-covered GTs (low generator score) are truncated before the reranker
sees them. **K_pool, not branch width, is the binding constraint at 500.** The escalation is G5 +
**K_pool=2000** (`mfd_rerank_engagement_g5_kpool2000.sbatch`; C=nb·2000=8000, reranker stores f/t/i
int16 + lp/base fp16 to fit RAM) — the last lever that admits those GTs into the pool.

### 17.5 Final results (TEST split; ★ = shipped final)
| run | t2_recall@10 | t2_ndcg@10 | t3_behavior_acc | t3_recall@10 | t3_ndcg@10 |
|---|---|---|---|---|---|
| MD-RQ-VAE + MB + Focal Loss | 0.1557 | 0.1557 | 0.508 | 0.1955 | 0.1496 |
| MBGen arch + MBGen SID | 0.1569 | 0.1569 | 0.8258 | 0.1976 | 0.1558 |
| MBGen arch + MD-RQ-VAE SID | 0.1574 | 0.1574 | 0.752 | 0.1971 | 0.1566 |
| MFD (original) | 0.1996 | 0.1803 | 0.5517 | 0.1899 | 0.1473 |
| MFD + behavior-marginalized pooling | 0.1987 | 0.18 | 0.534 | 0.1938 | 0.1484 |
| MFD + wider shared-f retrieval (K_f sweep) | 0.2085 | 0.1842 | 0.534 | 0.201 | 0.1509 |
| + PMQ (q_t,q_i) | 0.2074 | 0.1842 | 0.3848 | 0.2006 | 0.1469 |
| + PMQ + q_f  (**Run-C**, the reranker base) | 0.2094 | 0.1855 | 0.5291 | 0.203 | 0.153 |
| + CMIR | 0.2096 | 0.1852 | 0.5441 | 0.2024 | 0.1533 |
| **★ Final reranker (engagement, +both)** | **0.2225** | **0.1931** | 0.5305 | **0.215** | **0.1556** |

### 17.6 How to run — full pipeline, in order (multimodal embeddings → final reranker)
| # | stage | script / command | produces |
|---|---|---|---|
| 1 | **Multimodal item embeddings** (text + image per item) | upstream feature extraction | per-item text/image vectors |
| 2 | **MD-RQ-VAE tokenizer** (§15): encode each item into shared/text/image codes | tokenizer training run | `.../2048_1024/checkpoints/checkpoint_000_010000.ckpt` (codebook centroids) + `.../pickle/semantic_ids.pt` (the SID, 5.41M items, widths [2048,1024,1024]) |
| 3 | **MB TFRecords** (§5) | `build_tfrecords_mb_userdrop.sbatch` (`build_tmall_tfrecords_mb.py`) | `tfrecords_mb_userdrop/{training,evaluation,testing}` |
| 4 | **Train the generative recommender = Run-C** (MFD + PMQ + Q_f, §16.3) | `train_mfd_pmq_qf.sbatch` | frozen base ckpt `.../mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt` |
| 5 | **Reranker — full ablation** (Stages 1–3; re-dumps train/val/test with memory + `hist_ids`, then views/+hard/+soft/+both) | `mfd_rerank_engagement.sbatch` | dumps `mfd_rerank_eng_{trainsplit,val,test}_dump.pt`; ablation table |
| 5b | **Reproduce the shipped best (+both) with full metrics** | `mfd_rerank_best.sbatch` | `mfd_reranker_best.pt`; TEST t3@10 ≈ 0.2150 |
| — | (probes) wide-pool coverage experiments | `mfd_rerank_engagement_g5.sbatch` (K_pool=500), `mfd_rerank_engagement_g5_kpool2000.sbatch` | G5 dumps + reranker |

Steps 1–2 are the tokenizer (the primary research contribution); 3–4 build and train the frozen
generator; 5/5b are the second-stage reranker. Every reranker sbatch is guarded (skips a dump if its
`.pt` exists) and enables W&B (`MMMBGRec`). Do **not** disable W&B. The reranker needs the cached
MD-RQ-VAE centroids (`mfd_rerank_engagement.sbatch` builds `mdrqvae_centroids.pt` once from the 50GB
tokenizer ckpt; loading it requires the GRID `src` package on `sys.path`, handled in the trainer).

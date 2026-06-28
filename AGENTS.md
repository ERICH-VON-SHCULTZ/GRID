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
- [ ] Training comparison user_drop vs event_downsample, and use_item_pe / use_bmtv ablations,
      not yet run/validated end-to-end.
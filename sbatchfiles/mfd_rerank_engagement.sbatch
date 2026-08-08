#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_rerank_eng
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_rerank_engagement.out

# ============================================================================================
# ENGAGEMENT-FEATURE ablation for the listwise reranker, on the FROZEN MFD+PMQ+Q_f model + the
# axis-2 (more training data) setup. The main model is NOT retrained.
#
# WHY: the reranker currently sees only the 3 MD-RQ-VAE view vectors + 4 component log-probs. It
# ignores the strongest measured signal on this dataset -- repeat interaction: P(buy)=0.01% if the
# next item never appeared in history, 1.36% after a page view, 15.91% after add-to-cart (~15x),
# and ~92% of buys are repeats. Two new per-candidate feature families encode it:
#   HARD MATCH  per behavior {pv,fav,cart,buy}: does any history item have EXACTLY this candidate's
#               (f,t,i) tuple -> indicator, log1p(count), recency (distance from the sequence end).
#   SOFT MATCH  per behavior x per modality view {u_f, u_f+u_t, u_f+u_i}: max cosine of the
#               candidate's view vs the history items of that behavior. Catches variant
#               re-engagement (same product, different SKU/colorway) that exact match misses, and is
#               robust to the 36% SID collision rate. Only constructible because the tokenizer gives
#               additive shared/text/image views ("looks identical" vs "same specs" are separate
#               channels).
#
# The dump now also stores the raw history token sequence (hist_ids); the encoder memory is stored
# as before, so the reranker architecture is otherwise UNCHANGED (2 layers, d_model=128) -> the
# feature contribution is isolated. NOTE: the earlier reranker dumps lack hist_ids, so these are
# fresh dumps (multi-worker dataloading order is non-deterministic, so an append/merge would risk
# silent misalignment -- a single combined re-dump is the only guaranteed-aligned option).
#
# STAGE 1  dump train/val/test WITH hist_ids (train=training split 250k, val=evaluation, test=testing).
# STAGE 2  ONE reranker run with --ablation -> trains 4 configs from identical init on the same data:
#          views (current baseline) / +hard / +soft / +both. Selects each on VAL, reports TEST.
#          Prints per-config test t3 recall@{5,10}, t2@{5,10} routed, P(rank<10|cov), and the
#          per-behavior t3@10 breakdown (the aggregate is pv-dominated ~93%, so the mechanism shows
#          in the per-behavior split), plus a final comparison table.
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_rerank_runc.ckpt   # '='-free symlink for Hydra (shared)
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

# Fresh dumps (with hist_ids). Distinct names so the earlier axis-2 dumps stay intact.
TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt   # training split -> TRAIN
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt            # evaluation split -> VAL
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt          # testing split -> TEST

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='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'

# NOTE: Hydra args inlined per command line, list value single-quoted (variable-in-quotes trips
# the lexer). +model.mfd_dump_memory=true now also writes hist_ids.

# ---- STAGE 1a: EVALUATION split -> VAL ----
if [ ! -f $VAL_DUMP ]; then
  echo '=== STAGE 1a: dumping EVALUATION split (VAL, +hist_ids) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=6000 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation \
      mfd_audit_path=$VAL_DUMP
else echo '=== STAGE 1a: VAL_DUMP exists, skipping ==='; fi

# ---- STAGE 1b: TESTING split -> TEST ----
if [ ! -f $TEST_DUMP ]; then
  echo '=== STAGE 1b: dumping TESTING split (TEST, +hist_ids) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=6000 \
      mfd_audit_path=$TEST_DUMP
else echo '=== STAGE 1b: TEST_DUMP exists, skipping ==='; fi

# ---- STAGE 1c: TRAINING split -> TRAIN (250k cap) ----
if [ ! -f $TRAIN_DUMP ]; then
  echo '=== STAGE 1c: dumping TRAINING split (TRAIN, +hist_ids) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true mfd_audit_dump=true \
      mfd_audit_max_events=250000 mfd_audit_pv_cap=300000 +trainer.limit_test_batches=6000 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training \
      mfd_audit_path=$TRAIN_DUMP
else echo '=== STAGE 1c: TRAIN_DUMP exists, skipping ==='; fi

# ---- STAGE 2: engagement-feature ablation (views / +hard / +soft / +both) ----
echo '=== STAGE 2: reranker engagement ablation ==='
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE \
    --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --wb 1.0 --layers 2 --heads 4 --dropout 0.3 --wd 1e-3 \
    --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 \
    --batch 32 --eval_batch 64 --ablation \
    --save /scratch/yw8866/rec-tmall/mfd_reranker_eng.pt
"
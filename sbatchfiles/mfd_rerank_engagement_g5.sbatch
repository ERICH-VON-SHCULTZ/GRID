#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_rerank_g5
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_rerank_engagement_g5.out

# ============================================================================================
# G5 WIDE-POOL + ENGAGEMENT reranker. Same frozen MFD+PMQ+Q_f model (NOT retrained), same
# engagement reranker as mfd_rerank_engagement (views/+hard/+soft/+both ablation, UNCHANGED),
# but the candidate pool is re-dumped WIDE:
#     G4 (current): K_f=200, K_t=200,  K_i=50,   K_pool=500  -> coverage@pool = 0.2615
#     G5 (this):    K_f=200, K_t=1024, K_i=1024, K_pool=500
# K_t=K_i=1024 = the FULL head width (no t/i pruning). Instead of the infeasible dense
# K_f*K_t*K_i grid (209M/behavior, 53GB tensor), the pool is the top-K_pool by SID score over the
# UNIQUE CATALOG TUPLES directly (mfd_wide_catalog_pool=true) -- one vectorized pass, ~same cost as
# the current dump. This lifts coverage@grid to ~0.756; how much reaches the K_pool=500 pool
# (coverage@pool, printed as `covered` on each [data] line) is the question. GOAL: TEST t3@10 -> 0.22.
#
# Only the DUMP changes; the reranker is byte-identical to the engagement run (reads pool_* +
# component log-probs + enc_mem + hist_ids). K_pool stays 500 -> per-event candidate count
# C = nb*500 = 2000 unchanged, so reranker cost is unchanged.
#
# batch_size=32 (headroom for the (B, M_catalog~4.16M) scoring tensors) -> limit_test_batches raised
# so the 250k-event training dump is reached (250k/32 ~ 7.8k batches).
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_rerank_runc.ckpt   # '='-free symlink for Hydra (shared)
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

# G5 wide-pool dumps (with hist_ids + enc_mem). Distinct names from the G4 engagement dumps.
TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_g5_trainsplit_dump.pt   # training split -> TRAIN
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_g5_val_dump.pt            # evaluation split -> VAL
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_g5_test_dump.pt          # testing split -> TEST

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

# NOTE: Hydra args inlined per command line, list value single-quoted. +model.mfd_wide_catalog_pool
# =true selects the catalog-tuple pool; +model.mfd_dump_memory=true writes enc_mem + hist_ids.

# ---- STAGE 1a: EVALUATION split -> VAL (G5 wide) ----
if [ ! -f $VAL_DUMP ]; then
  echo '=== STAGE 1a: dumping EVALUATION split (VAL, G5 wide + hist_ids) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true +model.mfd_wide_catalog_pool=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=9000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation \
      mfd_audit_path=$VAL_DUMP
else echo '=== STAGE 1a: VAL_DUMP exists, skipping ==='; fi

# ---- STAGE 1b: TESTING split -> TEST (G5 wide) ----
if [ ! -f $TEST_DUMP ]; then
  echo '=== STAGE 1b: dumping TESTING split (TEST, G5 wide + hist_ids) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true +model.mfd_wide_catalog_pool=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=9000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
      mfd_audit_path=$TEST_DUMP
else echo '=== STAGE 1b: TEST_DUMP exists, skipping ==='; fi

# ---- STAGE 1c: TRAINING split -> TRAIN (250k cap, G5 wide) ----
if [ ! -f $TRAIN_DUMP ]; then
  echo '=== STAGE 1c: dumping TRAINING split (TRAIN, G5 wide + hist_ids) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true +model.mfd_wide_catalog_pool=true mfd_audit_dump=true \
      mfd_audit_max_events=250000 mfd_audit_pv_cap=300000 +trainer.limit_test_batches=9000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training \
      mfd_audit_path=$TRAIN_DUMP
else echo '=== STAGE 1c: TRAIN_DUMP exists, skipping ==='; fi

# ---- STAGE 2: SAME engagement ablation reranker, on the G5 wide dumps ----
echo '=== STAGE 2: reranker engagement ablation (G5 wide pools) ==='
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE \
    --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --wb 1.0 --layers 2 --heads 4 --dropout 0.3 --wd 1e-3 \
    --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 \
    --batch 32 --eval_batch 64 --ablation \
    --save /scratch/yw8866/rec-tmall/mfd_reranker_g5.pt
"
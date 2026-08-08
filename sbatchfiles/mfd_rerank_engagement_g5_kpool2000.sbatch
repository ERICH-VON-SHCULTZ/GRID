#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_g5_k2000
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_rerank_engagement_g5_kpool2000.out

# ============================================================================================
# FINAL SHOT at TEST t3@10 = 0.22: G5 wide branches (K_t=K_i=1024) + K_pool=2000 + engagement(+both).
# Frozen MFD+PMQ+Q_f (NOT retrained); reranker architecture UNCHANGED (2 layers, d_model=128).
#
# WHY K_pool=2000 (not 500): the G5 run at K_pool=500 left coverage@pool FLAT at 0.2618 (== G4's
# 0.2615) and t3@10 at 0.2146. Widening branches to full width added no coverage to a 500-pool
# because the top-500 by generator score is the same set regardless of branch width -- the newly
# branch-covered GTs (low generator score, that's why their t*/i* were low-ranked) sit at pool-rank
# 500-2000+ and were truncated before the reranker saw them. K_pool=2000 is the only lever that
# admits them: G5's full-width branches make them branch-eligible, K_pool=2000 lets them into the
# pool, and the engagement reranker (P(r<10|cov)~=0.82) can then order the repeat-heavy ones into
# the top-10.  t3@10 = coverage@pool x P(r<10|cov): watch `covered` on the [data] lines -- if it
# rises from 0.262 to ~0.27+ with ordering holding, 0.22 is cleared.
#
# COST: K_pool=2000 -> per-event candidate count C = nb*2000 = 8000 (4x). Reranker batch=16,
# eval_batch=32; RerankData now stores f/t/i as int16 and lp/base as fp16 so the wide dumps load.
# The DUMP is unchanged in cost (wide-catalog scoring is O(catalog) independent of K_pool); only
# the stored pool is 4x larger. Runs ONLY the +both config (the fixed engagement setting).
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_rerank_runc.ckpt   # '='-free symlink for Hydra (shared)
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

# G5 wide + K_pool=2000 dumps (with hist_ids + enc_mem). Distinct names.
TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_g5k2_trainsplit_dump.pt   # training split -> TRAIN
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_g5k2_val_dump.pt            # evaluation split -> VAL
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_g5k2_test_dump.pt          # testing split -> TEST

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

# Hydra args inlined; list value single-quoted. mfd_pool_size=2000 is the only change vs the G5-500
# dump; mfd_wide_catalog_pool=true keeps the catalog-tuple pool (full-width, no dense grid).

# ---- STAGE 1a: EVALUATION split -> VAL ----
if [ ! -f $VAL_DUMP ]; then
  echo '=== STAGE 1a: dumping EVALUATION split (VAL, G5 wide, K_pool=2000) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 mfd_pool_size=2000 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true +model.mfd_wide_catalog_pool=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=9000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation \
      mfd_audit_path=$VAL_DUMP
else echo '=== STAGE 1a: VAL_DUMP exists, skipping ==='; fi

# ---- STAGE 1b: TESTING split -> TEST ----
if [ ! -f $TEST_DUMP ]; then
  echo '=== STAGE 1b: dumping TESTING split (TEST, G5 wide, K_pool=2000) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 mfd_pool_size=2000 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true +model.mfd_wide_catalog_pool=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=9000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
      mfd_audit_path=$TEST_DUMP
else echo '=== STAGE 1b: TEST_DUMP exists, skipping ==='; fi

# ---- STAGE 1c: TRAINING split -> TRAIN (250k cap) ----
if [ ! -f $TRAIN_DUMP ]; then
  echo '=== STAGE 1c: dumping TRAINING split (TRAIN, G5 wide, K_pool=2000) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 mfd_pool_size=2000 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true +model.mfd_wide_catalog_pool=true mfd_audit_dump=true \
      mfd_audit_max_events=250000 mfd_audit_pv_cap=300000 +trainer.limit_test_batches=9000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training \
      mfd_audit_path=$TRAIN_DUMP
else echo '=== STAGE 1c: TRAIN_DUMP exists, skipping ==='; fi

# ---- STAGE 2: engagement reranker, +both only (the fixed best setting) ----
echo '=== STAGE 2: reranker (+both) on G5 wide K_pool=2000 dumps ==='
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE \
    --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --wb 1.0 --layers 2 --heads 4 --dropout 0.3 --wd 1e-3 \
    --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 \
    --batch 16 --eval_batch 32 --use_hard --use_soft \
    --save /scratch/yw8866/rec-tmall/mfd_reranker_g5k2.pt
"
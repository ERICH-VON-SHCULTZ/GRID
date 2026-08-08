#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_rerank_best
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_rerank_best.out

# ============================================================================================
# BEST reranker: reproduces the +both engagement run that reached TEST t3@10 = 0.2150 (baseline
# union_score 0.2041), the best result to date. This is the +both config of mfd_rerank_engagement
# run on its own -- identical because each ablation config reseeds (torch.manual_seed(seed)) right
# before building its model, so a standalone +both gets the same init + training trajectory.
#
# Same frozen MFD+PMQ+Q_f model (NOT retrained). Same G4-width engagement dumps (K_f=200,K_t=200,
# K_i=50,K_pool=500,+hist_ids). TRAIN on training split (~250k -> ~98k GT-in-pool), SELECT on the
# evaluation split, REPORT on the testing split. +both = hard tuple-match + soft cosine-match
# engagement features. Expected: VAL best @ ep1, TEST t3@10 ~= 0.215.
#
# report() now emits the FULL test metric set each epoch: t3/t2 recall@{5,10,20}, NDCG@{5,10},
# P(rank<10|covered), coverage, and the per-behavior t3@10 AND t2@10 breakdowns.
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_rerank_runc.ckpt   # '='-free symlink for Hydra (shared)
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

# The exact dumps behind the 0.2150 run (created by mfd_rerank_engagement.sbatch). Re-created here
# only if missing (guarded), so this script is self-contained and reproducible from scratch.
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

# ---- STAGE 1a: EVALUATION split -> VAL (guarded) ----
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

# ---- STAGE 1b: TESTING split -> TEST (guarded) ----
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

# ---- STAGE 1c: TRAINING split -> TRAIN (250k cap, guarded) ----
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

# ---- STAGE 2: the BEST reranker (+both), reproducing TEST t3@10 = 0.2150 ----
echo '=== STAGE 2: best reranker (+both engagement), full metrics ==='
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE \
    --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --wb 1.0 --layers 2 --heads 4 --dropout 0.3 --wd 1e-3 \
    --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 \
    --batch 32 --eval_batch 64 --seed 0 --use_hard --use_soft \
    --save /scratch/yw8866/rec-tmall/mfd_reranker_best.pt
"
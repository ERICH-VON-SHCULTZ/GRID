#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_reranker
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_reranker.out

# ============================================================================================
# Listwise reranker (stage 2) on top of the FROZEN MFD+PMQ+Q_f (run-C) model. The main model is
# never retrained. Three stages, all in this one job:
#
#   STAGE 1  candidate dump (frozen model, K_f=200 K_t=200 K_i=50 K_pool=500, +encoder memory):
#            run the instrumented mfd_audit_dump ONCE per split -- the evaluation split (reranker
#            TRAINING) and the testing split (EVALUATION). Each dump stores, per event, the
#            per-behavior candidate pools (f,t,i), the four component log-probs (beh,f,t|f,i|f),
#            the frozen ENCODER MEMORY + mask, the GT SID and gt_in_catalog. Guarded: skipped if
#            the .pt already exists (re-runs resume at stage 2).
#
#   STAGE 2  train the reranker: a shallow cross-encoder where candidate tokens (built from the
#            MD-RQ-VAE codebook geometry [u_f, u_f+u_t, u_f+u_i] via orthogonal W_up, plus the 4
#            log-probs as scalars) CROSS-ATTEND into the full encoder memory. Listwise CE over the
#            whole pool with the GT positive; trained only on GT-in-pool events; real-pool hard
#            negatives. (MSE / random negatives / pooled-query -- the three approaches that
#            degraded recall before -- are deliberately NOT used.)
#
#   STAGE 3  two-stage eval: offline rescoring of the dumped test pool over the dumped memory is
#            bit-identical to live "frozen generator -> reranker -> top-10". Reports t3 recall@
#            {5,10} (union/marginalized, w_b=1) and t2 recall@{5,10} (routed to GT behavior) on
#            the SAME denominators as analyze_mfd_audit (union_score/best_rank), plus
#            P(rank<10 | covered). BASELINE (union_score) is printed next to every reranked eval.
#
# SUCCESS: test t3 recall@10 clearly above the 0.2041 baseline; target 0.233 (half the +0.057
# ordering headroom; coverage ceiling / oracle recall is ~0.262).
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_reranker_runc.ckpt   # '='-free symlink for Hydra
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

# Reranker split hygiene: TRAIN on the training split, SELECT on the evaluation split, REPORT on
# the testing split (no test peeking for early stopping). The eval/test dumps already exist from
# the earlier run and are reused as VAL/TEST; only the (new) training-split dump is produced here.
TRAINSPLIT_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_trainsplit_dump.pt  # training split -> TRAIN
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_train_dump.pt              # evaluation split -> VAL (select)
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_test_dump.pt              # testing split -> TEST (report)

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

# NOTE: all Hydra args are inlined on each python command line (NOT a shell variable) and the
# list value is single-quoted -- putting quoted args in a variable makes the quotes literal and
# trips Hydra's lexer (LexerNoViableAltException). This is the known-good pattern.

# ---- STAGE 1a: EVALUATION-split dump (reranker VAL / model selection) ----
# Already exists from the earlier run; re-created only if missing.
if [ ! -f $VAL_DUMP ]; then
  echo '=== STAGE 1a: dumping EVALUATION split (reranker VAL) ==='
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
else
  echo '=== STAGE 1a: VAL_DUMP exists, skipping ==='
fi

# ---- STAGE 1c: TRAINING-split dump (reranker TRAINING data; axis-2 data scaling) ----
# Same G4 branch width as VAL/TEST so the reranker's train/val/test pools are structurally
# identical. Capped at 250k events (~3x the eval split) -> ~65k GT-in-pool at coverage 0.26.
if [ ! -f $TRAINSPLIT_DUMP ]; then
  echo '=== STAGE 1c: dumping TRAINING split (reranker training data) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true mfd_audit_dump=true \
      mfd_audit_max_events=250000 mfd_audit_pv_cap=300000 +trainer.limit_test_batches=6000 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training \
      mfd_audit_path=$TRAINSPLIT_DUMP
else
  echo '=== STAGE 1c: TRAINSPLIT_DUMP exists, skipping ==='
fi

# ---- STAGE 1b: test-split dump (EVALUATION data) ----
if [ ! -f $TEST_DUMP ]; then
  echo '=== STAGE 1b: dumping TESTING split (evaluation) ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      +model.mfd_dump_memory=true mfd_audit_dump=true \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 +trainer.limit_test_batches=6000 \
      mfd_audit_path=$TEST_DUMP
else
  echo '=== STAGE 1b: TEST_DUMP exists, skipping ==='
fi

# ---- STAGE 2+3: train the reranker and evaluate two-stage ----
echo '=== STAGE 2+3: training + evaluating the listwise reranker ==='
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \
    --train_dump $TRAINSPLIT_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE \
    --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --wb 1.0 --layers 2 --heads 4 --dropout 0.3 --wd 1e-3 \
    --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 \
    --batch 32 --eval_batch 64 \
    --save /scratch/yw8866/rec-tmall/mfd_reranker.pt
"
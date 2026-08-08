#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_cmir
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_cmir.out

# CMIR (Cross-Modal Iterative Refinement) on top of Run-C (MFD + PMQ + Q_f), warm-started from
# the Run-C checkpoint. Round 0 = the current masked P0(t|f)/P0(i|f); Round 1 conditions each
# modality query on the OTHER modality's soft top-m estimate (q_t on i_est, q_i on t_est), still
# one pass each and mutually masked -> P1(t|f,i_est), P1(i|f,t_est). Composition/scoring use R1.
# The estimate-injection projections are zero-init, so at step 0 R1 == R0 and Run-C survives.
#
# ALL data/loss settings match Run-C so the ONLY difference is the refinement round:
#   userdrop split, codebook_init=null (embeddings come from the warm start), marginalized,
#   w_b=1, K_f=50/K_t=10/K_i=10/pool=50, and focal OFF.
#   NOTE: the request said "focal gamma=1", but Run-C (train_mfd_pmq_qf.sbatch) actually trained
#   with focal OFF -- matched here to avoid a second confound. Add focal_behavior=True
#   focal_gamma=1.0 to BOTH phases if you specifically want the focal loss instead.
#
# Two phases: (1) warm-start train + test -> t1/t2/t3 + behavior; (2) audit on the trained ckpt
# -> branch_t / branch_i coverage-loss + coverage@pool (the decision metrics for CMIR: does
# conditioning on the other modality sharpen P(t|f), P(i|f) enough to lower branch pruning?).

RUNC=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
WARM=/scratch/yw8866/rec-tmall/_mfd_cmir_warm.ckpt          # '='-free symlink for warm start
ln -sf "$RUNC" "$WARM"
OUT=/scratch/yw8866/logs/train/mfd_cmir
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
DUMP=/scratch/yw8866/rec-tmall/mfd_cmir_audit_dump.pt
CHAIN_DUMP=/scratch/yw8866/rec-tmall/chain_tf_audit.pt

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

# ---- 1) warm-start train + test (use_cmir=true) ----
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_cmir \
    data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
    codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
    mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
    mfd_top_k_shared=50 mfd_top_k_text=10 mfd_top_k_image=10 mfd_pool_size=50 \
    use_cmir=true cmir_estimate_top_m=3 cmir_loss_weight=1.0 cmir_refine_top_m=-1 \
    cmir_scheduled_sampling=false \
    mfd_warmstart_path=$WARM hydra.run.dir=$OUT

# ---- 2) audit on the trained best checkpoint -> branch_t / branch_i / coverage@pool ----
CKPT=\$(ls -t $OUT/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n \"\$CKPT\" ]; then
  ALINK=/scratch/yw8866/rec-tmall/_mfd_cmir_audit.ckpt; ln -sf \"\$CKPT\" \$ALINK
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_cmir train=false test=true \
      ckpt_path=\$ALINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=50 mfd_top_k_text=10 mfd_top_k_image=10 mfd_pool_size=50 \
      use_cmir=true cmir_estimate_top_m=3 cmir_refine_top_m=-1 \
      mfd_audit_dump=true mfd_audit_path=$DUMP +trainer.limit_test_batches=6000
  CA=\"\"; [ -f $CHAIN_DUMP ] && CA=\"--chain_dump $CHAIN_DUMP\"
  python /scratch/yw8866/rec-tmall/analyze_mfd_audit.py --dump $DUMP --wb 1 \$CA
fi
"
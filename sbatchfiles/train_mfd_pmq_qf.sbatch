#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_pmq_qf
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_pmq_qf.out

# Run C of the PMQ grid (full): use_pmq=true, use_query_f=true.
# On top of B (per-modality query slots for t,i), the SHARED code f also gets its own query
# slot q_f -- its own hidden state and attention pass -- instead of reading h_beh. Block is
# [BOS, beh, q_f, f, q_t/q_i]; q_f is BEFORE f so causal masking already prevents it seeing f*.
# t,i still run in SEPARATE passes (exact P(t|f)*P(i|f)). Encoder untouched (stride 4). Warm-
# started from the wb=1 MFD checkpoint (new query params + slot-type emb start at init).
#
# Two phases in one job:
#   1) train + test (warm-start) -> test/t1_* t2_* t3_* + behavior acc/f1  (the headline metrics)
#   2) audit on the trained best ckpt -> f_pruned / coverage@pool / teacher-forced acc(t|GT f)
#
# Decision metric for C: pv f_pruned (currently 0.634 -> lower is the win) and coverage@pool
# (pv 0.158, overall 0.2085 -> higher), hence t3@10 above the 0.201 K_f=200 roof. The shared
# head getting its own capacity is the direct antidote to f-pruning that bigger K_f only masked.

WB1=/scratch/yw8866/logs/train/runs/2026-07-19/mfd_wb1/checkpoints/checkpoint_epoch=000_step=007500.ckpt
WARM=/scratch/yw8866/rec-tmall/_mfd_pmqqf_warm.ckpt
ln -sf "$WB1" "$WARM"
OUT=/scratch/yw8866/logs/train/mfd_pmq_qf
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
DUMP=/scratch/yw8866/rec-tmall/mfd_pmqqf_audit_dump.pt
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
export WANDB_MODE=disabled
export WANDB_API_KEY='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'

# ---- 1) warm-start train + test ----
# (args inlined -- Hydra list values like codebook_sizes must be single-quoted ON the command
#  line, NOT inside a shell variable, or the quotes reach Hydra as literal characters.)
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat \
    data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
    codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
    mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
    mfd_top_k_shared=50 mfd_top_k_text=10 mfd_top_k_image=10 mfd_pool_size=50 \
    +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
    +model.mfd_warmstart_path=$WARM hydra.run.dir=$OUT

# ---- 2) audit on the trained best checkpoint (=-safe symlink) ----
CKPT=\$(ls -t $OUT/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -n \"\$CKPT\" ]; then
  ALINK=/scratch/yw8866/rec-tmall/_mfd_pmqqf_audit.ckpt; ln -sf \"\$CKPT\" \$ALINK
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=\$ALINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=50 mfd_top_k_text=10 mfd_top_k_image=10 mfd_pool_size=50 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      mfd_audit_dump=true mfd_audit_path=$DUMP +trainer.limit_test_batches=6000
  CA=\"\"; [ -f $CHAIN_DUMP ] && CA=\"--chain_dump $CHAIN_DUMP\"
  python /scratch/yw8866/rec-tmall/analyze_mfd_audit.py --dump $DUMP --wb 1 \$CA
fi
"
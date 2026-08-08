#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_audit
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_audit.out

# MFD §0 audit: ONE instrumented test pass over the frozen wb=1 MFD checkpoint writes
# mfd_audit_dump.pt (per-behavior pools + beh_logp for 0a, teacher-forced GT-f reads for 0b,
# f*/t*/i* ranks for 0c). Then three CPU analyses read it. Optionally a chain teacher-forced
# pass (set CHAIN_CKPT) completes 0b's per-level-accuracy gap.
#
# NOTE: the checkpoint filenames contain '=' (checkpoint_epoch=..._step=...), which Hydra's
# override grammar rejects inside a value. We symlink them to '='-free paths and pass those.

WB1_CKPT=/scratch/yw8866/logs/train/runs/2026-07-19/mfd_wb1/checkpoints/checkpoint_epoch=000_step=007500.ckpt
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
DUMP=/scratch/yw8866/rec-tmall/mfd_audit_dump.pt

# CHAIN reference for 0b: a chain (SemanticIDMultiBehaviorEncoderDecoder) on the SAME MD-RQ-VAE
# 2048_1024 SID as MFD. NOTE: the 2026-07-10 chain ckpt is on a DIFFERENT SID (tiger_1024) and
# is NOT valid here. This one (2026-06-30) is on the correct SID, same backbone/data, align+bmtv
# OFF (clean state_dict load). CAVEAT: it was trained with codebook_init=MD-RQ-VAE-geometry +
# focal behavior, whereas MFD wb1 used codebook_init=null + no focal -> a small (second-order)
# regime confound on the text gap. For a fully clean 0b, train a chain with EXACTLY MFD's regime
# (tiger_mb_train_flat, codebook_init=null, focal off) -- that is also §16's matched baseline.
# Set CHAIN_CKPT="" to skip 0b's gap entirely (MFD absolutes only).
CHAIN_CKPT=/scratch/yw8866/logs/train/runs/2026-06-30/19-35-17/checkpoints/checkpoint_epoch=000_step=022500.ckpt
CHAIN_DUMP=/scratch/yw8866/rec-tmall/chain_tf_audit.pt

# Hydra-safe symlinks (no '=' in the name). /scratch is writable and visible in-container.
WB1_LINK=/scratch/yw8866/rec-tmall/_mfd_audit_wb1.ckpt
CHAIN_LINK=/scratch/yw8866/rec-tmall/_mfd_audit_chain.ckpt
ln -sf "$WB1_CKPT" "$WB1_LINK"
[ -n "$CHAIN_CKPT" ] && ln -sf "$CHAIN_CKPT" "$CHAIN_LINK"

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Audit runs log nothing to W&B; disable it so the WandbLogger never tries to log in
# (batch jobs have no tty -> otherwise 'api_key not configured' aborts the run).
# WANDB_MODE=disabled makes wandb.init a no-op; the key/SSL exports below are a fallback that
# matches the working training sbatches in case a run is still created.
export WANDB_MODE=disabled
export WANDB_API_KEY='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'
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true

# ---- 1) MFD rich audit dump (GPU, frozen wb=1 ckpt, test only) ----
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat \
    train=false test=true ckpt_path=$WB1_LINK \
    data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
    codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
    mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
    mfd_audit_dump=true mfd_audit_path=$DUMP \
    +trainer.limit_test_batches=6000

# ---- 2) (optional) chain teacher-forced dump for 0b ----
if [ -n \"$CHAIN_CKPT\" ]; then
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat \
    train=false test=true ckpt_path=$CHAIN_LINK \
    data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
    codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
    focal_behavior=True focal_gamma=1.0 \
    +model.audit_tf_dump=true +model.audit_tf_path=$CHAIN_DUMP \
    +trainer.limit_test_batches=6000
CHAIN_ARG=\"--chain_dump $CHAIN_DUMP\"
else
CHAIN_ARG=\"\"
fi

# ---- 3) offline analyses 0a / 0b / 0c (CPU, seconds) ----
if [ -f $DUMP ]; then
  python /scratch/yw8866/rec-tmall/analyze_mfd_audit.py --dump $DUMP --wb 1 2 4 \$CHAIN_ARG
else
  echo \"ERROR: MFD dump $DUMP was not produced (step 1 failed) — skipping analysis.\"
fi
"
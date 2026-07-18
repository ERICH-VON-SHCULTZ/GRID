#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=tiger_mb_pba_mdrqvae
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/tiger_mb_pba_mdrqvae.out

# MBGen PBA model on GRID's MD-RQ-VAE SIDs (2048_1024), train + test in one run.
#
# This is the MODEL ablation with the TOKENIZER held fixed: same 3-level MD-RQ-VAE SIDs the
# tiger baselines use, but the PBAMultiBehaviorEncoderDecoder (position-routed experts +
# behavior injection) instead of the plain tiger T5. Compare test/t1_* / t2_* / t3_* against
# the tiger-on-MD-RQ-VAE runs (isolates the architecture) -- and against PBA-on-MBGen-SID
# (train_tiger_mb_pba.sbatch, which isolates the tokenizer for the SAME model).
#
# Reuses configs/experiment/tiger_mb_pba_flat.yaml unchanged; only the SID path and the
# per-level codebook sizes differ from the MBGen-SID run and are passed here:
#   semantic_id_path -> the MD-RQ-VAE [3, N] tensor (shared/text/image levels)
#   codebook_sizes   -> [2048, 1024, 1024] (the MD-RQ-VAE per-level widths; tight heads)
#   num_hierarchies=3 -> stride 4, sequence_length auto 4*40=160
#
# codebook_init_path is left null (random SID-embedding init) to isolate the architecture.
# To instead match a tiger run that injected MD-RQ-VAE codebook geometry, set
# codebook_init_path=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export WANDB_API_KEY='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'
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_pba_flat \
    data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    semantic_id_path=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt \
    num_hierarchies=3 \
    codebook_init_path=null \
    'codebook_sizes=[2048,1024,1024]' \
    use_bmtv=False \
    use_item_pe=False \
    focal_behavior=False \
    behavior_weighted_sid=False \
    use_semantic_align=False \
    use_semantic_rerank=False \
    probe_sid_distance=False
"
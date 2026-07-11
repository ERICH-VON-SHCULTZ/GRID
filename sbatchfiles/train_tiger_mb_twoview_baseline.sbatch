#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=twoview_baseline
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/twoview_baseline.out

# Two-View experiment — WIDE-POOL BASELINE (train + test in one run).
#   - use_semantic_align=True   : train the align heads (identical model to the rerank run).
#   - top_k_for_generation=50   : widen the beam pool (same pool as the rerank run).
#   - use_semantic_rerank=False : candidates ranked by the decoder score (no rerank).
#   - focal OFF                 : clean attribution — two-view targets aggregate joint_Rec.
# Compare test/t3_recall_10 & test/t2_recall_10 against train_tiger_mb_twoview_rerank.sbatch.

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

python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat \
    codebook_init_path=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt \
    data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    semantic_id_path=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt \
    probe_sid_distance=True \
    use_semantic_align=True \
    semantic_align_weight=1.0 \
    use_semantic_rerank=False \
    top_k_for_generation=50 \
    num_hierarchies=3
"
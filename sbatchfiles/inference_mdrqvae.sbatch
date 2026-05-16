#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=inference_mdrqvae
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/inference_mdrqvae_1024.out

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache

python /scratch/yw8866/GRID/src/inference.py experiment=md_rqvae_inference_flat \
    items_dir=/scratch/yw8866/rec-tmall/image_sqf \
    codebook_width_l1=2048 \
    codebook_width_l2=1024 \
    codebook_width_l3=1024 \
    ckpt_path=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt
"
#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=10:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=train_mdrqvae
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/train_tiger_1024.out

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

python /scratch/yw8866/GRID/src/train.py experiment=rqvae_siglip_train_flat \
    items_dir=/scratch/yw8866/rec-tmall/image_sqf \
    codebook_width=1024 \
    ++should_skip_retry=True
"
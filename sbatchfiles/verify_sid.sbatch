#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=verify_sid
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/verify_md_sid_2048_1024.out

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310


python /scratch/yw8866/GRID/src/utils/verify_sids.py \
    --pkl_path /scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/merged_predictions.pkl \
    --codebook_widths 2048,1024,1024 \
     --convert
"
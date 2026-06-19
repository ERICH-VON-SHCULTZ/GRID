#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --mem=8GB
#SBATCH --job-name=debug_behavior_strings
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/debug_behavior_strings.out

module purge

singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
python /scratch/yw8866/rec-tmall/debug_behavior_strings.py
"
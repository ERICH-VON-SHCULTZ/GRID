#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=analyze_tfrecords_mb
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/analyze_tfrecords_mb.out

module purge

singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
python /scratch/yw8866/rec-tmall/analyze_tfrecords_mb.py
"
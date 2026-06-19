#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=150GB
#SBATCH --job-name=build_tfrecords_mb_aug
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/build_tfrecords_mb_aug.out

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache

python /scratch/yw8866/GRID/src/data/build_tmall_tfrecords_mb.py \
    --valid_data_dir    /scratch/yw8866/rec-tmall/valid_data \
    --output_dir        /scratch/yw8866/rec-tmall/tfrecords_mb_aug \
    --min_seq_len       5 \
    --users_per_file    5000 \
    --train_frac        0.8 \
    --val_frac          0.1 \
    --augment \
    --drop_pv_only_prob 0.5 \
    --seed              42
"
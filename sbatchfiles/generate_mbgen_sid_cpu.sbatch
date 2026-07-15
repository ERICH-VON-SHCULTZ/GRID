#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=gen_mbgen_sid_cpu
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/generate_mbgen_sid_cpu.out

# CPU-ONLY MBGen semantic-ID generation (no GPU allocation -> no GPU walltime cap).
#
# Why this works: the GPU only ever accelerated QAE training. The two other stages --
# the ~50GB SigLIP embedding gather and the level-2 conditional KMeans over 2048 groups --
# are CPU/IO-bound and cost the same either way.
#
# The QAE is the one stage that would be slow on CPU (~15M-param MLP over 5.4M items), so
# we fit it on a 1.5M-item subsample. ALL 5.4M items are still ENCODED afterwards, so every
# item gets a semantic ID -- only the autoencoder's training set is subsampled. That is a
# standard scale adaptation and costs very little codebook quality.
#
# Rough budget on 48 cores: gather ~1h, QAE ~2-4h, conditional KMeans ~1-2h.
# If it is still too slow, drop qae.train_sample to 750000 or qae.epochs to 16.

module purge

singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310

cd /scratch/yw8866/GRID
python src/data/generate_mbgen_sid.py \
    --config configs/experiment/mbgen_sid_gen.yaml \
    device=cpu \
    num_workers=48 \
    qae.train_sample=1500000
"
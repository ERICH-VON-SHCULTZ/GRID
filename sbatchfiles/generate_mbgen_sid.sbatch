#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=gen_mbgen_sid
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/generate_mbgen_sid.out

# Generate MBGen (MMMB-Genrec)-style semantic IDs for the Tmall catalogue.
#   Level 1: QAE (MLP autoencoder + EMA-VQ codebook, num_id1=2048, latent 32)
#   Level 2: conditional residual KMeans within each level-1 group (num_id2=2048)
# Input embedding = concat(SigLIP text, SigLIP image) = 2304-d, standardised.
# Output: /scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/semantic_ids.pt  [2, N]
#         (column == item_id, same layout as the MD-RQ-VAE semantic_ids.pt).
# ~200GB RAM: the assembled 5.4M x 2304 f32 matrix is ~50GB; StandardScaler done in place.
# Tune knobs via CLI dotlist overrides (e.g. qae.epochs=60 num_id2=1024).

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/yw8866/GRID
python src/data/generate_mbgen_sid.py \
    --config configs/experiment/mbgen_sid_gen.yaml
"
#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=sid_factor
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/analyze_sid_factor.out

# Part G (§12g) -- is the MD-RQ-VAE text/image factorization near-lossless?
# Count-level, from semantic_ids.pt ALONE (no tfrecords, no behavior, no model): over all
# 5.4M items measures CMI(L1;L2 | L0) -- the residual text<->image coupling once the shared
# fusion code L0 is known -- with a within-L0 shuffle as the plug-in bias floor. If the
# bias-corrected CMI is small vs H(L1|L0), H(L2|L0), then given L0 the modality codes are
# near-independent and MFD (modality-factorized decoding) is licensed by the DATA before any
# training. Structural companion to A.4. CPU-only, ~10-20 min, well under the 64GB / 1h ask.
#
# Point --sid_path at whichever SID you want to certify (default = MD-RQ-VAE 2048_1024).

module purge

singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
python /scratch/yw8866/rec-tmall/analyze_sid_behavior_coupling.py \
    --part factor \
    --sid_path /scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt \
    --out_artifacts /scratch/yw8866/rec-tmall/coupling_artifacts_factor.pt
"
#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sid_coupling_asym
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/analyze_sid_coupling_asym.out

# Part B -- per-behavior modality asymmetry (text-view vs image-view ranking of the next
# item). Builds V_t = E_f[sid0]+E_t[sid1], V_i = E_f[sid0]+E_i[sid2] from the MD-RQ-VAE
# centroids (loaded mmap so only the ~8MB of codebooks materialise, not the 50GB ckpt),
# ranks the GT against 8192 fixed negatives per view, and bootstraps Δ(text-img) per
# behavior. Gates Step 3 (behavior-gated BM-TV). GPU used only for the ranking matmuls.
# ~2-3 h with the 500k-pv cap. ~200GB RAM: 3 x [N,512] fp16 view tables (~25GB) + overhead.

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /scratch/yw8866/rec-tmall/analyze_sid_behavior_coupling.py \
    --part asym \
    --data_dir /scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    --out_artifacts /scratch/yw8866/rec-tmall/coupling_artifacts_asym.pt
"
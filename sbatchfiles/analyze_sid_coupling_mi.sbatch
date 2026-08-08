#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=sid_coupling_mi
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/analyze_sid_coupling_mi.out

# Part A -- is behavior predictable from the SID beyond what previous behavior gives?
# CPU-only (no GPU); fits count tables on training, scores held-out buy detection on eval.
# Gates Step 2 (SID-conditioned behavior head). ~1 h on the full userdrop splits.
#
# Swap --data_dir to /scratch/yw8866/rec-tmall/tfrecords_mb (natural build) to make the
# prior correction a no-op (train & eval both ~93.6% pv). userdrop is the verified build.

module purge

singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
python /scratch/yw8866/rec-tmall/analyze_sid_behavior_coupling.py \
    --part mi \
    --data_dir /scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    --out_artifacts /scratch/yw8866/rec-tmall/coupling_artifacts_mi.pt
"
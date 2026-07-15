#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=tiger_mbgen_baseline
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/tiger_mbgen_baseline.out

# MBGen-tokenizer BASELINE (train + test in one run).
# Trains the SAME multi-behavior TIGER model as the other experiments, but on the MBGen
# (QAE + conditional-KMeans) 2-token semantic IDs instead of GRID's 3-level MD-RQ-VAE SIDs.
# This isolates the effect of the TOKENIZER: identical data, identical T5, only the SID
# assignment differs. Compare test/t*_recall_* against the MD-RQ-VAE runs.
#
# MBGen SID is 3 tokens: [id1 (QAE), id2 (conditional KMeans), id3 (dedup counter)].
# num_hierarchies=3 -> stride_per_item=4, sequence_length auto = 4*40 = 160.
# codebook_sizes = [2048, <max id2 +1>, <max dedup bucket +1>]  -- the generator prints the
# exact line "=> train with: num_hierarchies=3 codebook_sizes=[...]" at the end of its run
# (see expoutfile/generate_mbgen_sid_cpu.out). COPY those values into codebook_sizes below.
# All GRID-specific extensions are OFF (BM-TV/two-view/probe need the MD-RQ-VAE geometry;
# codebook init needs its checkpoint), so this is the plain flat baseline on MBGen SIDs.
#
# PREREQUISITE: run generate_mbgen_sid(.cpu).sbatch first so semantic_ids.pt exists.

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat \
    data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    semantic_id_path=/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/semantic_ids.pt \
    num_hierarchies=3 \
    codebook_init_path=null \
    'codebook_sizes=[2048,2048,DEDUP]' \
    use_bmtv=False \
    use_item_pe=False \
    focal_behavior=False \
    behavior_weighted_sid=False \
    use_semantic_align=False \
    use_semantic_rerank=False \
    probe_sid_distance=False
"

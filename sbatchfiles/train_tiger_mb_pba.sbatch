#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=tiger_mb_pba
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/tiger_mb_pba.out

# MBGen PBA-architecture run (train + test in one run).
# Same data + SIDs + evaluator as the tiger baselines; the difference is the MODEL:
# PBAMultiBehaviorEncoderDecoder replaces each T5 FFN with MBGen's position-routed experts
# (one expert per within-item token position) + behavior injection (behavior embedding
# concatenated into the FFN input on selected layers). Everything else -- SID embeddings,
# beam search, model_step, MultiBehaviorSIDRetrievalEvaluator -- is inherited from the
# tiger model, so test/t1_* / t2_* / t3_* are directly comparable to the tiger runs.
#
# Consumes the MBGen SIDs (3 tokens: id1, id2, dedup) from generate_mbgen_sid.
#   num_hierarchies=3 -> stride 4, sequence_length auto 4*40=160.
#   codebook_sizes is left null in the config -> the model sizes every SID level to
#   (SID tensor max + 1), which covers id1/id2/dedup with no manual value. (For tighter
#   per-level heads, pass codebook_sizes=[2048,<max id2+1>,<max bucket+1>] from the
#   generator's "=> train with: ..." line.)
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

python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_pba_flat \
    data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    semantic_id_path=/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/semantic_ids.pt \
    num_hierarchies=3 \
    codebook_init_path=null \
    use_bmtv=False \
    use_item_pe=False \
    focal_behavior=False \
    behavior_weighted_sid=False \
    use_semantic_align=False \
    use_semantic_rerank=False \
    probe_sid_distance=False
"
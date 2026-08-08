#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=tiger_mb_mfd
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/tiger_mb_mfd_wb2.out

# MFD (Modality-Factorized Decoding) on GRID's MD-RQ-VAE SIDs (2048_1024), train + test.
#
# This is the DECODER ablation with the tokenizer + data + backbone held fixed: same 3-level
# MD-RQ-VAE SIDs and the same T5 backbone the chain baselines use, but the decoder drops the
# t->i edge (licensed by §12g CMI) and generates by a compositional beam with agreement
# scoring. Compare test/t1_* / t2_* / t3_* against:
#   * the CHAIN baseline at matched params/data: tiger_mb_train_flat.yaml, SAME overrides
#     (this is ablation (a) chain-vs-factorized, same everything);
#   * pool-matched: set the chain baseline's top_k_for_generation=50 to match mfd_pool_size.
#
# codebook_init_path is left null (random SID-embedding init) to isolate the decoder change.
# To inject MD-RQ-VAE codebook geometry instead, set
#   codebook_init_path=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt
#
# Free-gen (t3) pooling defaults to behavior-marginalized (mfd_behavior_marginalized=true): the
# beam runs the f-branch under ALL 4 behaviors and ranks the union by w_b*logP(b)+SID_score, so
# the per-behavior retrieval that wins t2 runs inside t3. Tune the behavior/SID trade-off with
# mfd_behavior_weight (w_b); sweep it FIRST since it is the lever for the t3 behavior-accuracy.
#
# Ablation sweeps (rerun this file with an extra override, new --output each time):
#   w_b sweep (t3 behavior)        : mfd_behavior_weight=2.0   /   mfd_behavior_weight=4.0
#   committed vs marginalized      : mfd_behavior_marginalized=False   (the old single-behavior beam)
#   (b) agreement vs single-branch : mfd_score_image=False   (or mfd_score_text=False)
#   (c) tau sweep                  : mfd_tau=0.7   /   mfd_tau=1.5
#   (d) K_t/K_i asymmetry          : mfd_top_k_text=20 mfd_top_k_image=5   (etc.)
#   catalog off (diagnostic)       : mfd_catalog_constraint=False

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

python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat \
    data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    semantic_id_path=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt \
    num_hierarchies=3 \
    codebook_init_path=null \
    'codebook_sizes=[2048,1024,1024]' \
    focal_behavior=False \
    behavior_weighted_sid=False \
    use_item_pe=False \
    use_bmtv=False \
    use_semantic_align=False \
    use_semantic_rerank=False \
    mfd_behavior_weight=2.0 \
    probe_sid_distance=False
"
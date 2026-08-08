#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_pmqqf_poolsweep
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_pmqqf_poolsweep.out

# G4 (K_f=200, K_t=200, K_i=50, K_pool=500) audit on the run-C (PMQ+Q_f) ckpt, on the FULL
# 83k-event test set (pv_cap=100000 -> all pv kept), so t2/t3 are COMPARABLE to Run-C / G3.
#
# WHY the full set: the previous G4 dump used pv_cap=45000 (to bound the K_pool=2000 dump) ->
# 50172 events = 89.7% pv vs the baselines' 93.8% pv. Since pv recall (~0.16) << non-pv (~0.92),
# the lighter pv mix INFLATED aggregate t3 to 0.2365; on the full set the same model gives
# t3@10 ~= 0.205 (comparable to Run-C 0.2030 / G3 0.2042). recall@10 is K_pool-invariant, so
# K_pool=500 vs 2000 is irrelevant here -- the pv mix was the whole difference. This run settles
# the real number. Read 0a (t2/t3 union+routed) on n=83005.

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_pmqqf_poolsweep.ckpt     # '='-free symlink for Hydra
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
DUMP=/scratch/yw8866/rec-tmall/mfd_pmqqf_poolsweep_dump.pt
CHAIN_DUMP=/scratch/yw8866/rec-tmall/chain_tf_audit.pt

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='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'

# ---- 1) one G4-branch-width audit dump at K_pool=2000 (covers {500,1000,2000} by thresholding) ----
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
    ckpt_path=$LINK \
    data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
    codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
    mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
    mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
    +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
    mfd_audit_dump=true mfd_audit_path=$DUMP \
    mfd_audit_max_events=200000 mfd_audit_pv_cap=100000 +trainer.limit_test_batches=6000

# ---- 2) analysis: the K_pool sweep (coverage@pool + P(rank<10)) + 0c + the G1..G6 grid ----
CA=\"\"; [ -f $CHAIN_DUMP ] && CA=\"--chain_dump $CHAIN_DUMP\"
python /scratch/yw8866/rec-tmall/analyze_mfd_audit.py --dump $DUMP --wb 1 \$CA
"
#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=08:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_kf_sweep
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_kf_sweep.out

# Step 2: real t3@10 vs K_f (the audit's coverage sweep gives the oracle ceiling offline; this
# gives the ACTUAL recall). Normal MFD eval (marginalized, w_b=1), frozen wb=1 ckpt, test only,
# K_t=K_i=10 fixed, K_f in {10,20,50,100,200}. f_pruned=0.596 at K_f=10 is the coverage killer;
# widening K_f adds genuinely-new shared-code branches (NOT the behavior-union crowding case), so
# coverage -- and, since in-pool ordering is good (P(rank<10)=0.93), t3 -- should climb. Find the knee.
#
# NOTE run this AFTER the audit re-dump confirms the catalog unit test passes (n_invalid==0).

WB1_CKPT=/scratch/yw8866/logs/train/runs/2026-07-19/mfd_wb1/checkpoints/checkpoint_epoch=000_step=007500.ckpt
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
WB1_LINK=/scratch/yw8866/rec-tmall/_mfd_kf_wb1.ckpt
ln -sf "$WB1_CKPT" "$WB1_LINK"

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export WANDB_API_KEY='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'

for KF in 10 20 50 100 200; do
  echo \"##### MFD K_f = \$KF (K_t=K_i=10, w_b=1, marginalized) #####\"
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat \
      train=false test=true ckpt_path=$WB1_LINK \
      data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=\$KF mfd_top_k_text=10 mfd_top_k_image=10 mfd_pool_size=50 \
      +trainer.limit_test_batches=3000
done

echo \"===== t3/t2/t1 recall vs K_f =====\"
grep -nE '##### MFD K_f|test/t3_recall_10|test/t3_recall_5|test/t2_recall_10|test/t1_recall_10|test/t3_behavior_accuracy' /scratch/yw8866/GRID/expoutfile/mfd_kf_sweep.out | tail -60
"
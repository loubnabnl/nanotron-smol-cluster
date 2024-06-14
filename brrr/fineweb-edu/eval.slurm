#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/fsx/loubna/logs/ablations_v2/350B/slurm-logs/evals/logs-%x-%n-%j.out
#SBATCH --error=/fsx/loubna/logs/ablations_v2/350B/slurm-logs/evals/logs-%x-%n-%j.err

###########################################
# [BEGINING] ADAPT TO YOUR ENVIRONMENT

source ~/.bashrc
source /admin/home/loubna/miniconda3/etc/profile.d/conda.sh
conda activate nanotron 

BRRR_FOLDER=/fsx/loubna/projects/brrr
LOCAL_DOWNLOAD_CHECKPOINT_FOLDER=/scratch/loubna/checkpoint/

export HUGGINGFACE_HUB_CACHE=/fsx/loubna/.cache
export HF_DATASETS_CACHE=/fsx/loubna/.cache
export HF_MODULES_CACHE=/fsx/loubna/.cache

# [END] ADAPT TO YOUR ENVIRONMENT
###########################################
set -x -e
echo "START TIME: $(date)"
echo python3 version = `python3 --version`

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR=/scratch
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.1

echo go $COUNT_NODE
echo $HOSTNAMES

# Copying checkpoint from s3 to the node on node
mkdir -p $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER
s5cmd cp --exclude "optimizer/*" s3://synthetic-project-models/ablations/FW-Edu_28B_gpt2_ablation-1p82G-FW-Edu-threshold-2-seed-0-/12500/* $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER

torch_dist_args="--nproc_per_node 8 \
    --nnodes $COUNT_NODE \
    --max_restarts 0 \
    --tee 3 \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: "

launch_args="$torch_dist_args $BRRR_FOLDER/run_evals_nanotron.py \
    --checkpoint-config-path ${LOCAL_DOWNLOAD_CHECKPOINT_FOLDER}/config.yaml \
        "

srun  -u bash -c "python3 -u -m torch.distributed.run ${launch_args}"

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=hopper-prod
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:4
#SBATCH --qos=high
#SBATCH --cpus-per-task=48
#SBATCH --output=/fsx/loubna/logs/leaderboard/%x-%j.out

set -x -e

source /admin/home/loubna/miniconda3/etc/profile.d/conda.sh
conda activate textbooks

echo "START TIME: $(date)"

GPUS_PER_NODE=4
NUM_NODES=$SLURM_NNODES
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000


model=$1
task=$2
org=$3
out_path=$4

CMD="\
    /fsx/loubna/projects/bigcode-evaluation-harness/main.py \
    --model $org/$model \
    --tasks $task \
    --max_length_generation 1024 \
    --batch_size 20 \
    --n_samples 50 \
    --temperature 0.2 \
    --precision bf16 \
    --allow_code_execution \
    --trust_remote_code \
    --save_generations \
    --use_auth_token \
    --generation_only \
    --save_generations_path $out_path/generations_$task\_$model.json \
"

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file /fsx/loubna/projects/bigcode-evaluation-harness/leaderboard/multi_gpu.yaml \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# Hugging Face Cluster specific, activates fast networking
module load cuda/12.1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"

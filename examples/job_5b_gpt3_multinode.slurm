#!/bin/bash

#SBATCH --job-name=5b_gpt3_nemo_32
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --partition=production-cluster
#SBATCH --exclusive
#SBATCH -o /fsx/loubna/code/Megatron-LM/scaling_laws/5b/logs/%x-%j.out
#SBATCH -e /fsx/loubna/code/Megatron-LM/scaling_laws/5b/logs/%x-%j.err

set -x -e
source /admin/home/loubna/.bashrc

conda activate megatron

# File Path setup
echo "START TIME: $(date)"
SCRIPT_REPO=/fsx/loubna/code/Megatron-LM
pushd $SCRIPT_REPO
LOG_PATH=$SCRIPT_REPO/main_log.txt

# Experiment parameters
JOB=5b_gpt3_nemo_32
CHECKPOINT_PATH=/fsx/loubna/experiments/scaling-laws/5b/$JOB
TOKENIZER_FILE=/fsx/loubna/data/tokenizer/tokenizer-the-stack-march-sample-v3-no-prefix-spaces/tokenizer.json
DATA_PATH=/fsx/loubna/data/stack_new/gpt2-preprocessed_content_document
WEIGHTS_TRAIN=/fsx/loubna/code/Megatron-LM/scaling_laws/bigcode-data-mix/data/train_data_paths.txt.tmp
WEIGHTS_VALID=/fsx/loubna/code/Megatron-LM/scaling_laws/bigcode-data-mix/data/valid_data_paths.txt.tmp


N_GPUS=8
TP=2
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024
NLAYERS=24
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048
LEARNING_RATE=0.00016
LR_WARMUP_ITERS=250
INIT=0.01
SAVE_INTERVAL=5000
TRAIN_ITERS=14826

# Training Setup
GPUS_PER_NODE=$N_GPUS
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "args defined"
GPT_ARGS="\
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size 1 \
    --recompute-activations \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --attention-head-type multiquery \
    --init-method-std $INIT \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr $LEARNING_RATE \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters $TRAIN_ITERS  \
    --lr-decay-style cosine \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --weight-decay .1 \
    --adam-beta2 .95 \
    --clip-grad 1.0 \
    --bf16 \
    --initial-loss-scale 65536 \
    --fim-rate 0.5 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $SAVE_INTERVAL \
    --eval-iters 5 \
    --valid-num-workers 0 \
    "
# we only evaluate every SAVE_INTERVAL because eval is slow, we have many validation sets

TENSORBOARD_ARGS="--tensorboard-dir $CHECKPOINT_PATH/tensorboard"


CMD=" \
    /fsx/loubna/code/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path $WEIGHTS_TRAIN \
    --valid-weighted-split-paths-path $WEIGHTS_VALID \
    $TENSORBOARD_ARGS \
    --wandb-entity-name loubnabnl \
    --wandb-project-name scaling_laws \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
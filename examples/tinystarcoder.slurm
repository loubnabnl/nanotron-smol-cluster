#!/bin/bash

#SBATCH --job-name=tiny_starcoder
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --mem-per-cpu=11G
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --partition=production-cluster
#SBATCH -o /fsx/bigcode/tinystarcoder/logs/%x-%j.out
#SBATCH -e /fsx/bigcode/tinystarcoder/logs/%x-%j.err

set -x -e
source /admin/home/loubna/.bashrc
conda activate megatron

# File Path setup
echo "START TIME: $(date)"
SCRIPT_REPO=/fsx/loubna/code/Megatron-LM
pushd $SCRIPT_REPO
LOG_PATH=$SCRIPT_REPO/main_log.txt

# Experiment parameters
CHECKPOINT_PATH=/fsx/bigcode/tinystarcoder/saves/
TOKENIZER_FILE=/fsx/bigcode/bigcode-training/tokenizer-starcoder/tokenizer.json
DATA_PATH=/fsx/bigcode/tinystarcoder/data/gpt2-preprocessed_content_document

N_GPUS=8
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=256
LEARNING_RATE=0.0003
LR_WARMUP_ITERS=1000
NLAYERS=20
NHIDDEN=768
NHEADS=12
SEQ_LEN=8192
SAVE_INTERVAL=5000
TRAIN_ITERS=50000


# Training Setup
GPUS_PER_NODE=$N_GPUS
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


GPT_ARGS="    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --recompute-activations \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --attention-head-type multiquery \
    --init-method-std 0.036 \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr $LEARNING_RATE \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters $TRAIN_ITERS \
    --lr-decay-style cosine \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --weight-decay .1 \
    --adam-beta2 .95 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --fim-rate 0.5 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 500 \
    --eval-iters 10 \
"

TENSORBOARD_ARGS="--tensorboard-dir $/fsx/bigcode/tinystarcoder/saves/tensorboard"


CMD="     /fsx/loubna/code/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    $TENSORBOARD_ARGS \
    --wandb-entity-name loubnabnl \
    --wandb-project-name tiny_starcoder "
    
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
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

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
SRUN_ARGS="     --wait=60     --kill-on-bad-exit=1     "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"

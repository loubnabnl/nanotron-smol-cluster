# code  adapted from
# https://github.com/huggingface/datablations/blob/98bc331ee97ca465263b72fc49371bcacefb712b/training_scripts/job_pretrain_gpt.py

import os
import pandas as pd

DATA_PATH = "/fsx/loubna/data/santacoder/gpt2-preprocessed_content_document"
CHECKPOINT_PATH = "/fsx/loubna/experiments/scaling-laws"
TOKENIZER_FILE = (
    "/fsx/loubna/data/tokenizer/digit-bytelevel-bpe-jss-v1.1-49152/tokenizer.json"
)


def makejob(
    JOB_NAME="scaling-laws",
    N_GPUS=2,
    CHECKPOINT_PATH=CHECKPOINT_PATH,
    TOKENIZER_FILE=TOKENIZER_FILE,
    DATA_PATH=DATA_PATH,
    MICRO_BATCH_SIZE=2,
    GLOBAL_BATCH_SIZE=16,
    LEARNING_RATE=0.0001,
    LR_WARMUP_ITERS=175,
    NLAYERS=2,
    NHIDDEN=8,
    NHEADS=2,
    SEQ_LEN=2048,
    SAVE_INTERVAL=1000,
    TRAIN_ITERS="10_000",
):
    return f"""#!/bin/bash

#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --partition=production-cluster
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH -o /fsx/loubna/code/Megatron-LM/scaling_laws/logs/%x-%j.out
#SBATCH -e /fsx/loubna/code/Megatron-LM/scaling_laws/logs/%x-%j.err

set -x -e
source /admin/home/loubna/.bashrc
conda activate megatron

# File Path setup
echo "START TIME: $(date)"
SCRIPT_REPO=/fsx/loubna/code/Megatron-LM
pushd $SCRIPT_REPO
LOG_PATH=$SCRIPT_REPO/main_log.txt

# Experiment parameters
CHECKPOINT_PATH={CHECKPOINT_PATH}
TOKENIZER_FILE={TOKENIZER_FILE}
DATA_PATH={DATA_PATH}
N_GPUS={N_GPUS}
MICRO_BATCH_SIZE={MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE}
LEARNING_RATE={LEARNING_RATE}
LR_WARMUP_ITERS={LR_WARMUP_ITERS}
NLAYERS={NLAYERS}
NHIDDEN={NHIDDEN}
NHEADS={NHEADS}
SEQ_LEN={SEQ_LEN}
SAVE_INTERVAL={SAVE_INTERVAL}
TRAIN_ITERS={TRAIN_ITERS}

# Training Setup
GPUS_PER_NODE=$N_GPUS
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


GPT_ARGS="\
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --recompute-activations \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --attention-head-type multiquery \
    --init-method-std 0.022 \
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
    --eval-interval 200 \
    --eval-iters 10 \
    "
TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"


CMD=" \
    /fsx/loubna/code/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFileWithFIM \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
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
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
"""


def submit_job(job, job_name="job"):
    with open(f"jobs/{job_name}.sbatch", "w") as fp:
        fp.write(job)
    os.system(f"sbatch jobs/{job_name}.sbatch")


# Ensure the log directory exists
os.system("mkdir -p /fsx/loubna/code/Magtron-LM/scaling_laws/logs")

df = pd.read_csv("params_sheet.csv")
for i in range(1, 3):
    print(f"Preparing job {i}")
    row = df.loc[i]

    # write a job for each row
    num_layers = row["num_layer"]
    hidden_size = row["hidden_size"]
    num_heads = row["num_heads"]
    sequence_length = row["sequence_length"]
    global_batch_size = row["global_batch_size"]
    micro_batch_size = int(row["micro_batch_size"] / 2)
    learning_rate = row["learning_rate"]
    num_gpu = 2 * row["num_gpu"]
    training_iters = row["training_iters"]
    lr_warmup_iters = row["lr_warmup_iters"]

    job_name = (
        f"model_id{i}_{num_layers}_{hidden_size}_{num_heads}_bs{micro_batch_size}"
    )
    ckpt_path = f"{CHECKPOINT_PATH}/model_id{i}"
    print(f"Checkpoints path: {ckpt_path}")
    os.makedirs(ckpt_path, exist_ok=True)
    
    job = makejob(
        JOB_NAME=job_name,
        CHECKPOINT_PATH=ckpt_path,
        N_GPUS=num_gpu,
        NLAYERS=num_layers,
        NHIDDEN=hidden_size,
        NHEADS=num_heads,
        SEQ_LEN=sequence_length,
        GLOBAL_BATCH_SIZE=global_batch_size,
        MICRO_BATCH_SIZE=micro_batch_size,
        LEARNING_RATE=learning_rate,
        LR_WARMUP_ITERS=lr_warmup_iters,
        TRAIN_ITERS=training_iters,
        # max(round(training_iters // 10 / 1000), 1) * 1000
        SAVE_INTERVAL=1000,
    )
    # submit the job
    print(f"Submitting job {i}, saved at jobs/{job_name}.sbatch")
    submit_job(job, job_name)

# View logs
# tail -f logs/<JOB_NAME>-<JOB_ID>.out logs/<JOB_NAME>-<JOB_ID>.err

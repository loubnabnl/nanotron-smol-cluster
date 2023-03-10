# code  adapted from
# https://github.com/huggingface/datablations/blob/98bc331ee97ca465263b72fc49371bcacefb712b/training_scripts/job_pretrain_gpt.py

import os
import argparse
import pandas as pd

# DATA_PATH is not used here, but added for non weighted data runs
DATA_PATH = "/fsx/loubna/data/stack_new/gpt2-preprocessed_content_document"
WEIGHTS_TRAIN = "/fsx/loubna/code/Megatron-LM/scaling_laws/bigcode-data-mix/data/train_data_paths.txt.tmp"
WEIGHTS_VALID = "/fsx/loubna/code/Megatron-LM/scaling_laws/bigcode-data-mix/data/valid_data_paths.txt.tmp"
CHECKPOINT_PATH = "/fsx/loubna/experiments/scaling-laws"
TOKENIZER_FILE = "/fsx/loubna/data/tokenizer/tokenizer-the-stack-march-sample-v3-no-prefix-spaces/tokenizer.json"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    args = parser.parse_args()
    return args


def submit_job(job, job_name="job"):
    with open(f"jobs/{job_name}.sbatch", "w") as fp:
        fp.write(job)
    os.system(f"sbatch jobs/{job_name}.sbatch")


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
#SBATCH --cpus-per-task={int(96*N_GPUS/8)}
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --partition=production-cluster
#SBATCH --hint=nomultithread
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
    --eval-interval 10000 \
    --eval-iters 5 \
    --valid-num-workers 0 \
    --structured-logs \
    --structured-logs-dir {CHECKPOINT_PATH}/logs \
    "
TENSORBOARD_ARGS="--tensorboard-dir {CHECKPOINT_PATH}/tensorboard"


CMD=" \
    /fsx/loubna/code/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path {WEIGHTS_TRAIN} \
    --valid-weighted-split-paths-path {WEIGHTS_VALID} \
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


if __name__ == "__main__":
    df = pd.read_csv("params_sheet.csv")
    args = get_args()
    print(f"Submitting jobs for experiments from {args.start} to {args.end}")
    for i in range(args.start, args.end + 1):
        row = df.loc[i]
        # write a job for each row
        num_layers = row["num_layer"]
        hidden_size = row["hidden_size"]
        num_heads = row["num_heads"]
        sequence_length = row["sequence_length"]
        global_batch_size = row["global_batch_size"]
        # TODO: add multi-node setup
        # TODO: batch sizes in csv are large => OOM => divide by 2 or 4 gor 2GPU runs
        param_count = row["param_count"]
        if param_count > 2000:
            micro_batch_size = 4
        elif param_count > 1000:
            micro_batch_size = 8
        elif param_count > 210:
            micro_batch_size = 16
        elif param_count > 50:
            micro_batch_size = 32
        else:
            micro_batch_size = 64
        num_gpu = row["num_gpu"]
        learning_rate = row["learning_rate"]
        training_iters = row["training_iters"]
        lr_warmup_iters = row["lr_warmup_iters"]

        identifier = f"{row['param_count']}M_{row['compute']}"
        job_name = f"run_{identifier}_bs{micro_batch_size}_idx_{i}"
        ckpt_path = f"{CHECKPOINT_PATH}/{job_name}"
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
            # save ~5 checkpoints and round to nearest 1000 multiple
            SAVE_INTERVAL=max(round(training_iters // 5 / 1000), 1) * 1000,
        )
        # submit the job
        print(f"Job for index {i} ready and saved at jobs/{job_name}.sbatch")
        submit_job(job, job_name)

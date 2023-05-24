#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=X
#SBATCH --output=/leonardo_scratch/large/userexternal/lbenalla/trainings/logs/run-%x-%j.out

set -x -e
source /leonardo/home/userexternal/lbenalla/.bashrc

conda activate megatron-2

CUDA_HOME=/leonardo/home/userexternal/lbenalla/miniconda3/envs/megatron-2
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

echo "START TIME: $(date)"

# File Path setup
SCRIPT_REPO=/leonardo/home/userexternal/lbenalla/code/Megatron-LM
pushd $SCRIPT_REPO

LOG_PATH=$SCRIPT_REPO/main_log.txt

# Training setup
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6001
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# File path setup
CHECKPOINT_PATH=/leonardo_scratch/large/userexternal/lbenalla/trainings/ckpts
TOKENIZER_FILE=/leonardo/home/userexternal/lbenalla/tokenizer-starcoder/tokenizer.json
WEIGHTS_TRAIN=/leonardo/home/userexternal/lbenalla/code/bigcode-data-mix/data/train_data_paths.txt.tmp
WEIGHTS_VALID=/leonardo/home/userexternal/lbenalla/code/bigcode-data-mix/data/valid_data_paths.txt.tmp

mkdir -p $CHECKPOINT_PATH/tensorboard


GPT_ARGS="\
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --attention-head-type multiquery \
       --init-method-std 0.01 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --micro-batch-size 2 \
       --global-batch-size 512 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --train-iters 18642 \
       --lr-decay-iters 18642 \
       --lr-decay-style cosine \
       --lr-warmup-iters 1000 \
       --weight-decay .1 \
       --adam-beta2 .95 \
       --clip-grad 1.0 \
       --bf16 \
       --fim-rate 0.5 \
       --log-interval 10 \
       --save-interval 5000 \
       --eval-interval 5000 \
       --eval-iters 2 \
       --valid-num-workers 0 \
"

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"

CMD=" \
    /leonardo/home/userexternal/lbenalla/code/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path $WEIGHTS_TRAIN \
    --valid-weighted-split-paths-path $WEIGHTS_VALID \
    --structured-logs \
    --structured-logs-dir $CHECKPOINT_PATH/logs \
    $TENSORBOARD_ARGS \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "

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

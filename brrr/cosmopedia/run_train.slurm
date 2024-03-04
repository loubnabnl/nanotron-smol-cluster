#!/bin/bash
#SBATCH --job-name=1b_w_code
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/loubna/logs/trainings/%x-%j
#SBATCH --array=1-2%1
#SBATCH --qos=high
#SBATCH --begin=now+0minutes


# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 use_trainer.py --config-file ./examples/loubna/config_llama_1b.yaml

###########################################
# [BEGINING] ADAPT TO YOUR ENVIRONMENT
CONFIG_FILE=/fsx/loubna/projects/training/brrr/examples/loubna/config_llama_1b.yaml
TRAINER_SCRIPT=/fsx/loubna/projects/training/brrr/use_trainer.py

# [END] ADAPT TO YOUR ENVIRONMENT
###########################################


set -x -e
echo "START TIME: $(date)"
secs_to_human(){
    echo "$(( ${1} / 3600 )):$(( (${1} / 60) % 60 )):$(( ${1} % 60 ))"
}
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"


# Set up env
source ~/.bashrc
export AWS_DEFAULT_REGION=us-east-1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export USE_FAST=1

# export USE_FAST=1
source /admin/home/loubna/miniconda3/etc/profile.d/conda.sh
#conda activate /fsx/nouamane/miniconda/envs/2-1-cu121
conda activate nanotron
module load cuda/12.1

GPUS_PER_NODE=8

if [ -d "checkpoint" ]; then
    echo "Removing previous checkpoint directory"
    rm -rf checkpoint
fi


##### Copy config file to logs ######
echo "CONFIG_FILE is set to '$CONFIG_FILE'"
# remove yaml extension
CONFIG_FILE_NAME=${CONFIG_FILE##*/}
CONFIG_FILE_NAME=${CONFIG_FILE_NAME%.yaml}

# copy config file to tmp folder
TMP_CONFIG_FILE=/fsx/loubna/logs/trainings/configs/${CONFIG_FILE_NAME}_${SLURM_JOB_ID}.yaml
echo "Renaming config file to $TMP_CONFIG_FILE"
# create TMP_CONFIG_FILE folder if it doesn't exist
mkdir -p $(dirname $TMP_CONFIG_FILE)
cp $CONFIG_FILE $TMP_CONFIG_FILE
CONFIG_FILE=$TMP_CONFIG_FILE


CMD=" \
    $TRAINER_SCRIPT \
    --config-file $CONFIG_FILE
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_JOB_NUM_NODES \
    --rdzv-backend etcd-v2 \
    --rdzv-endpoint etcd.hpc-cluster-hopper.hpc.internal.huggingface.tech:2379 \
    --rdzv-id $SLURM_JOB_ID \
    --max_restarts 0 \
    --tee 3 \
    "


# Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

launch_args="srun $SRUN_ARGS -u bash -c $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"


echo "END TIME: $(date)"

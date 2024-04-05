import os
import subprocess
import tempfile
from datetime import datetime

import torch
from nanotron.logging import human_format
from nanotron.models.llama import LlamaConfig

from brrr.config import (
    BrrrConfig,
    BrrrDataArgs,
    BrrrExperimentLoggerArgs,
    BrrrS3UploadArgs,
    CheckpointsArgs,
    GeneralArgs,
    WandbLoggerConfig,
    HubTensorBoardLoggerConfig,
    LightEvalWandbLoggerConfig,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizedBytesDatasetArgs,
    TokenizedBytesDatasetFolderArgs,
    TokenizerArgs,
    TokensArgs,
)

###########################################
# CHANGE THIS SECTION
BRRR_FOLDER = "/fsx/loubna/projects/brrr"
RUN_EVAL_SLURM_TEMPLATE = "/fsx/loubna/projects/brrr/examples/loubna/eval_1b.slurm.jinja"
EVAL_SLURM_SCRIPT_DIR = "/fsx/loubna/logs/ablations/eval-scripts"
S5CMD_PATH = "/fsx/nouamane/miniconda/envs/2-1-cu121/bin/s5cmd"
LOCAL_TMP_PATH_ON_NODE = "/scratch/loubna"

SLURM_LOGS_PATH = "/fsx/loubna/logs/ablations/slurm-logs"
BRRR_CONFIGS_PATH = "/fsx/loubna/logs/ablations/launch-configs"

EMAIL = "loubna@huggingface.co"

S3_CHECKPOINTS_PREFIX = "s3://synthetic-project-models/ablations/"

NODES = 16

# General name to gather the runs on the hub
PROJECT = "ablations_faq"

REPO_ID = f"HuggingFaceTB/loubna-{PROJECT}"
# END CHANGE THIS SECTION
###########################################

# uncomment whatever model you want to use
model_config = LlamaConfig(
    # Config for a 1.82/1.61B model
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=8192,
    max_position_embeddings=2048,
    num_attention_heads=32,
    num_hidden_layers=24,
    num_key_value_heads=32,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=49152, 
)

# model_config = LlamaConfig(
#     # Config for a 3.28/3.59B model
#     bos_token_id=1,
#     eos_token_id=2,
#     hidden_act="silu",
#     hidden_size=3072,
#     initializer_range=0.02,
#     intermediate_size=10752,
#     max_position_embeddings=4096,
#     num_attention_heads=32,
#     num_hidden_layers=24,
#     num_key_value_heads=32,
#     pretraining_tp=1,
#     rms_norm_eps=1e-05,
#     rope_scaling=None,
#     tie_word_embeddings=False,
#     use_cache=True,
#     vocab_size=50272,  # GPT2 tokenizer rounded to next multiple of 8
# )

# model_config = LlamaConfig(
#     # Config for a 6.48/6.74B model
#     bos_token_id=1,
#     eos_token_id=2,
#     hidden_act="silu",
#     hidden_size=4096,
#     initializer_range=0.02,
#     intermediate_size=11008,
#     max_position_embeddings=4096,
#     num_attention_heads=32,
#     num_hidden_layers=32,
#     num_key_value_heads=32,
#     pretraining_tp=1,
#     rms_norm_eps=1e-05,
#     rope_scaling=None,
#     tie_word_embeddings=False,
#     use_cache=True,
#     vocab_size=50272,  # GPT2 tokenizer rounded to next multiple of 8
# )

num_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        3 * model_config.hidden_size * model_config.intermediate_size
        + 4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

print(f"Model has {num_params} parameters")

# Do we  have a SLURM task ID?
# You can SLURM_ARRAY_TASK_ID to run multiple runs with predefined HP
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
job_id = os.environ.get("SLURM_JOB_ID", "")

# Seed for model and data
SEED = [5, 6][task_id % 2]


def launch_slurm_job(launch_file_contents, *args):
    """
        Small helper function to save a sbatch script and call it.
    Args:
        launch_file_contents: Contents of the sbatch script
        *args: any other arguments to pass to the sbatch command

    Returns: the id of the launched slurm job

    """
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(launch_file_contents)
        f.flush()
        return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]


if __name__ == "__main__":
    import argparse
    from dataclasses import fields, is_dataclass

    from brrr.config import get_config_from_file

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="dataset folder", type=str)
    parser.add_argument("run_name", help="run name", type=str)
    args = parser.parse_args()

    def print_differences(target, updates):
        if not is_dataclass(target) or not is_dataclass(updates):
            raise ValueError("Both target and updates should be dataclass instances")

        for field in fields(target):
            update_value = getattr(updates, field.name)

            if update_value is not None:
                if is_dataclass(update_value):
                    print_differences(getattr(target, field.name), update_value)
                else:
                    target_value = getattr(target, field.name)
                    if update_value != target_value:
                        if update_value.__class__.__module__ != "builtins":
                            continue
                        print(f"{field.name}: {target_value} -> {update_value}")

    dataset_name = run_name = args.run_name.replace(" ", "_")

    # Specific name for this run (checkpoints/logs/tensorboard)
    RUN = f"{PROJECT}-{num_params}-{dataset_name}-{job_id}"
    datasets = [
        TokenizedBytesDatasetFolderArgs(
            folder=args.data,
            filename_pattern=r".*\.ds$",
            shuffle=True,
            seed=SEED,
        )
    ]

    data = BrrrDataArgs(
        seed=SEED,
        num_loading_workers=0,
        dataset=TokenizedBytesDatasetArgs(
            datasets=datasets,
            # dataset_weights=dataset_weights,  # No upsampling of any (these are just the relative number of mega-tokens)
            # dataset_max_tokens=dataset_max_tokens,
            dataloader_type="single",  # cyclic,
            pad_samples_to_global_batch_size=False,
            # Set to True if you want to pad the last partial batch with -1's to equal global batch size,
        ),
    )

    general = GeneralArgs(
        project=PROJECT,
        run=RUN,
        ignore_sanity_checks=True,
    )

    lighteval = LightEvalConfig(
        tasks=LightEvalTasksArgs(
            tasks="early-signal",  # "generatives", "all"
            custom_tasks="brrr.lighteval.evaluation_tasks",
            max_samples=1000,  # Cap very large evals or for debugging
            dataset_loading_processes=8,
        ),
        parallelism=ParallelismArgs(
            dp=8,
            pp=1,
            tp=1,
            pp_engine="1f1b",
            tp_mode="ALL_REDUCE",
            # recompute_granularity="selective",
            tp_linear_async_communication=False,
        ),
        batch_size=16,
        wandb=LightEvalWandbLoggerConfig(
            wandb_project=PROJECT,
            wandb_entity="loubnabnl",
            wandb_run_name=f"{RUN}_evals",
        ),
        logging=LightEvalLoggingArgs(
            local_output_path=f"{LOCAL_TMP_PATH_ON_NODE}/lighteval/{RUN}",
            push_details_to_hub=False,
            push_results_to_hub=True,
            push_results_to_tensorboard=True,
            # hub_repo_details=REPO_ID,
            hub_repo_results=REPO_ID,
            hub_repo_tensorboard=REPO_ID,
            tensorboard_metric_prefix="e",
        ),
        slurm_template=RUN_EVAL_SLURM_TEMPLATE,
        slurm_script_dir=EVAL_SLURM_SCRIPT_DIR,
    )

    checkpoints = CheckpointsArgs(
        checkpoints_path=f"{LOCAL_TMP_PATH_ON_NODE}/checkpoints/{RUN}",
        checkpoints_path_is_shared_file_system=False,
        resume_checkpoint_path=f"{S3_CHECKPOINTS_PREFIX}/{RUN}",
        checkpoint_interval=500,
        save_initial_state=False,
    )

    parallelism = ParallelismArgs(
        dp=128, # 16 nodes
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    # num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    # parallelism.dp=int(num_nodes*8//parallelism.pp//parallelism.tp),  # How many remaining GPU when taking into account PP, TP and 8 GPUs per node

    tokens = TokensArgs(
        batch_accumulation_per_replica=1,
        micro_batch_size=4,
        sequence_length=2048, # GBS 1M
        train_steps=8000,   # 8B tokens, 4 epochs
        val_check_interval=500,
    )

    model = ModelArgs(
        model_config=model_config,
        make_vocab_size_divisible_by=1,
        init_method=RandomInit(
            std=0.02,
            # std=1
            # / math.sqrt(model_config.hidden_size)  # 0.01275  # Basically 1/sqrt(N),
            # path="/fsx/shared-falcon-180B/brrr-falcon-180B"
        ),
        dtype=torch.bfloat16,
    )

    logging = LoggingArgs(
        # 'debug', 'info', 'warning', 'error', 'critical' and 'passive'
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=1,
    )

    optimizer = OptimizerArgs(
        accumulate_grad_in_fp32=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        torch_adam_is_fused=True,
        weight_decay=0.1,
        zero_stage=0,
        learning_rate_scheduler=LRSchedulerArgs(
            learning_rate=3e-4,
            lr_warmup_steps=400,
            lr_warmup_style="linear",
            lr_decay_style="cosine",
            lr_decay_steps=8000,
            min_decay_lr=3.0e-5,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="lvwerra/the-tokenizer-v1",
    )

    s3_upload = BrrrS3UploadArgs(
        upload_s3_path=f"{S3_CHECKPOINTS_PREFIX}/{RUN}",
        remove_after_upload=True,
        s5cmd_numworkers=16,
        s5cmd_concurrency=5,
        s5cmd_path=S5CMD_PATH,
    )

    experiment_logger = BrrrExperimentLoggerArgs(
        wandb_logger=WandbLoggerConfig(
            wandb_project=PROJECT,
            wandb_entity="loubnabnl",
        ),
        tensorboard_logger=HubTensorBoardLoggerConfig(
            tensorboard_dir=f"{LOCAL_TMP_PATH_ON_NODE}/tensorboard-cosmo-{PROJECT}",
            # flush_secs=20,
            repo_id=REPO_ID,
            push_to_hub_interval=5,
            repo_public=False,
        )
    )

    config = BrrrConfig(
        general=general,
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=model,
        tokenizer=tokenizer,
        logging=logging,
        tokens=tokens,
        optimizer=optimizer,
        data=data,
        profiler=None,
        experiment_logger=experiment_logger,
        s3_upload=s3_upload,
        lighteval=lighteval,
        kill_switch_path=None,
    )

    #### DEBUG MODE
    if os.environ.get("DEBUG_MODE", "0") != "0":
        print("##### WARNING DEBUG MODE #####")
        config.parallelism.dp = 2
        config.parallelism.pp = 2
        config.parallelism.tp = 2
        config.tokens.micro_batch_size = 3
        config.tokens.batch_accumulation_per_replica = 2
        config.checkpoints.save_initial_state = True
        NODES = 1

    # Sanity check that we can load, save to YAML and reload the config
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"{BRRR_CONFIGS_PATH}/{run_name}", exist_ok=True)
    config_path_yaml = f"{BRRR_CONFIGS_PATH}/{run_name}/{timestamp}.yaml"
    config.save_as_yaml(config_path_yaml)
    config2 = get_config_from_file(config_path_yaml, config_class=BrrrConfig)
    print_differences(config, config2)

    os.makedirs(f"{SLURM_LOGS_PATH}/{run_name}", exist_ok=True)
    #SBATCH --mail-user={EMAIL}
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output={SLURM_LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j
#SBATCH --array=1-1%1
#SBATCH --qos=high
#SBATCH --begin=now+0minutes
#SBATCH --mail-type=ALL


TRAINER_PYTHON_FILE={BRRR_FOLDER}/use_trainer.py
set -x -e

echo "START TIME: $(date)"
secs_to_human(){{
    echo "$(( ${{1}} / 3600 )):$(( (${{1}} / 60) % 60 )):$(( ${{1}} % 60 ))"
}}
start=$(date +%s)
echo "$(date -d @${{start}} "+%Y-%m-%d %H:%M:%S"): ${{SLURM_JOB_NAME}} start id=${{SLURM_JOB_ID}}\n"

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR=/scratch
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.1

echo go $COUNT_NODE
echo $HOSTNAMES

##### MOVE TO YAML ######

CMD=" \
    $TRAINER_PYTHON_FILE \
    --config-file {config_path_yaml}
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes $COUNT_NODE \
    --rdzv-backend etcd-v2 \
    --rdzv-endpoint etcd.hpc-cluster-hopper.hpc.internal.huggingface.tech:2379 \
    --rdzv-id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
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
"""
    print(f"Slurm job launched with id={launch_slurm_job(sbatch_script)}")

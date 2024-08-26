import os
import subprocess
import tempfile
from datetime import datetime

import torch
from nanotron.logging import human_format
from nanotron.models.llama import LlamaConfig

from brrr.config.brrr_config import BrrrDatasetStageArgs

from brrr.config import (
    BrrrConfig,
    BrrrDataArgs,
    BrrrExperimentLoggerArgs,
    BrrrS3UploadArgs,
    CheckpointsArgs,
    GeneralArgs,
    HubTensorBoardLoggerConfig,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    AdamWOptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizedBytesDatasetArgs,
    TokenizedBytesDatasetFolderArgs,
    TokenizerArgs,
    TokensArgs,
    LightEvalWandbLoggerConfig,
    WandbLoggerConfig,
)

###########################################
# CHANGE THIS SECTION
BRRR_FOLDER = "/fsx/loubna/projects/brrr"
RUN_EVAL_SLURM_TEMPLATE = "/fsx/loubna/projects/brrr/examples/loubna/eval_1b.slurm.jinja"
EVAL_SLURM_SCRIPT_DIR = "/fsx/loubna/logs/ablations_v2/350B/eval-scripts"
S5CMD_PATH = "/admin/home/loubna/miniconda3/envs/nanotron/bin/s5cmd"
LOCAL_TMP_PATH_ON_NODE = "/scratch/loubna"

SLURM_LOGS_PATH = "/fsx/loubna/logs/ablations_v2/350B/slurm-logs"
BRRR_CONFIGS_PATH = "/fsx/loubna/logs/ablations_v2/350B/launch-configs"

EMAIL = "loubna@huggingface.co"

S3_CHECKPOINTS_PREFIX = "s3://synthetic-project-models/big-run-5T/"

NODES = 32

# General name to gather the runs on the hub
PROJECT = "smollm-big-run"

REPO_ID = f"HuggingFaceTB/{PROJECT}"
# END CHANGE THIS SECTION
###########################################

# uncomment whatever model you want to use
model_config = LlamaConfig(
    # Config for a 1.82/1.61B model
    bos_token_id=0,
    eos_token_id=0,
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
    vocab_size=49152, # make sure to change when changing tokenizer
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

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
job_id = os.environ.get("SLURM_JOB_ID", "")

# Seed for model and data
SEED = 0
TRAIN_STEPS = 4800000

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
    parser.add_argument("--data", help="dataset folder", type=str)
    parser.add_argument("--run_name", help="run name", type=str)
    parser.add_argument("--train_steps", help="training steps", type=str, default=TRAIN_STEPS)
    parser.add_argument("--seed", help="train and data seed", type=str, default=SEED)
    parser.add_argument("--warmup_steps", help="number of warmup steps", type=int, default=2000)
    parser.add_argument("--ratio_decay_steps", help="percentage of decay steps with respect to the total number of steps", type=int, default=20)
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
    RUN = f"{PROJECT}-{num_params}-{dataset_name}-seed-{args.seed}-{job_id}"

    datasets_stable = [
        TokenizedBytesDatasetFolderArgs(
            folder="/fsx/loubna/tokenized_for_exps/fw_edu/fineweb-edu-full-cosmo2_merged", # 1.4T tokens
            filename_pattern=r".*\.ds$",
            shuffle=True,
            seed=SEED,
        ),
        TokenizedBytesDatasetFolderArgs(
            folder="/fsx/loubna/tokenized_for_exps/fw_edu/dclm-3T-cosmo2_merged", # 3T tokens
            filename_pattern=r".*\.ds$",
            shuffle=True,
            seed=SEED,
        ),
        TokenizedBytesDatasetFolderArgs(
            folder="/fsx/loubna/tokenized_for_exps/fw_edu/starcoderdata-full-cosmo_merged", # 230B tokens
            filename_pattern=r".*\.ds$",
            shuffle=True,
            seed=SEED,
        ),
        # TokenizedBytesDatasetFolderArgs(
        #     folder="/fsx/loubna/tokenized_for_exps/cosmo2_training_data/Open-Web-Math-fix_merged", # 13B tokens
        #     filename_pattern=r".*\.ds$",
        #     shuffle=True,
        #     seed=SEED,
        # )
    ]

    # datasets_decay = [
    #     TokenizedBytesDatasetFolderArgs(
    #         folder="/fsx/loubna/tokenized_for_exps/FineWeb-edu-70BT_merged2",
    #         filename_pattern=r".*\.ds$",
    #         shuffle=True,
    #         seed=SEED,
    #     ),
    #     TokenizedBytesDatasetFolderArgs(
    #         folder="/fsx/loubna/tokenized_for_exps/Cosmopedia-v2-final-30BT_merged",
    #         filename_pattern=r".*\.ds$",
    #         shuffle=True,
    #         seed=SEED,
    #     )
    # ]

    data_stable = BrrrDataArgs(
        seed=SEED,
        num_loading_workers=0,
        dataset=TokenizedBytesDatasetArgs(
            datasets=datasets_stable,
            dataloader_type="cyclic",
            pad_samples_to_global_batch_size=False,
            dataset_weights=[0.5, 0.4, 0.1],
        ),
    )

    # data_decay = BrrrDataArgs(
    #     seed=SEED,
    #     num_loading_workers=0,
    #     dataset=TokenizedBytesDatasetArgs(
    #         datasets=datasets_decay,
    #         dataloader_type="cyclic",
    #         pad_samples_to_global_batch_size=False,
    #         dataset_weights=[103, 29],
    #     ),
    # )

    ratio_decay_steps = args.ratio_decay_steps / 100 
    nb_decay_steps =  int(args.train_steps * ratio_decay_steps)
    print(f"Decay steps: {nb_decay_steps}")
    starting_decay_step = args.train_steps -  nb_decay_steps

    data_stages = [
        BrrrDatasetStageArgs(
            name="stable", 
            start_training_step=1, 
            data=data_stable,
        )
        # BrrrDatasetStageArgs(
        #     name="decay", 
        #     start_training_step=starting_decay_step, 
        #     data=data_decay,
        # )
    ]

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
        resume_checkpoint_path="s3://synthetic-project-models/big-run-5T/smollm-big-run-1p81G-smollm-1.7B-5T-seed-0-/1886000/", #f"{S3_CHECKPOINTS_PREFIX}/{RUN}",
        checkpoint_interval=1000,
        save_initial_state=True,
    )

    parallelism = ParallelismArgs(
        dp=256, # 16 nodes
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    tokens = TokensArgs(
        batch_accumulation_per_replica=1, # 2M GBS = 4*6*64*2048/1e6
        micro_batch_size=4,
        sequence_length=2048,
        train_steps=args.train_steps,
        val_check_interval=100,
    )
    gbs = parallelism.dp * tokens.batch_accumulation_per_replica * tokens.micro_batch_size * tokens.sequence_length
    total_tokens = gbs * args.train_steps
    print(f"GBS: {(gbs)/1e6:.2f}M, total training tokens: {(total_tokens)/1e9:.2f}B")

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

    nb_warmup_steps = args.warmup_steps
    # Cosine
    # learning_rate_scheduler = LRSchedulerArgs(
    #         learning_rate=6e-4,
    #         lr_warmup_steps=2000,
    #         lr_warmup_style="linear",
    #         lr_decay_style="cosine",
    #         lr_decay_steps=args.train_steps, 
    #         min_decay_lr=6.0e-5,
    # )

    # WSD
    learning_rate_scheduler = LRSchedulerArgs(
        learning_rate=5e-4,
        lr_warmup_steps=2000,
        lr_warmup_style="linear",
        lr_decay_style="linear",            
        lr_decay_steps = nb_decay_steps,
        lr_decay_starting_step= starting_decay_step,
        min_decay_lr=0,
    )

    optimizer = OptimizerArgs(
        zero_stage=0,
        weight_decay=0.01,
        clip_grad=1.0,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=learning_rate_scheduler,
        optimizer_factory=AdamWOptimizerArgs(
            adam_eps=1e-08,
            adam_beta1=0.9,
            adam_beta2=0.95,
            torch_adam_is_fused=True,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="HuggingFaceTB/cosmo2-tokenizer",
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
            push_to_hub_interval=50,
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
        data_stages=data_stages,
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

    print(f"Logs at {SLURM_LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j")
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=big-run
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --exclude ip-26-0-172-57
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output={SLURM_LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j.out
#SBATCH --array=1-2%1
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

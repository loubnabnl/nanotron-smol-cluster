import os
import argparse
import subprocess

SCRIPT_DIR = "/fsx/loubna/code/bigcode-dataset/tokenization/jobs/scripts"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    args = parser.parse_args()
    return args


def submit_job(job, job_name="job"):
    with open(f"{SCRIPT_DIR}/{job_name}.sbatch", "w") as fp:
        fp.write(job)
    #os.system(f"{SCRIPT_DIR}/{job_name}.sbatch")
    subprocess.run(["sbatch", f"{SCRIPT_DIR}/{job_name}.sbatch"])


def makejob(JOB_NAME="pii-redaction", LANG=None, OUT_PATH="/fsx/loubna/data/tokenized_stack_no_pii"):
    return f"""#!/bin/bash

#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --partition=production-cluster
#SBATCH -o /fsx/loubna/code/bigcode-dataset/tokenization/jobs/logs/%x-%j.out
#SBATCH -e /fsx/loubna/code/bigcode-dataset/tokenization/jobs/logs/%x-%j.err

set -x -e
source /admin/home/loubna/.bashrc
conda activate megatron

# File Path setup
echo "START TIME: $(date)"

# Experiment parameters
LANG={LANG}

# Training Setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID


CMD=" \
    /fsx/loubna/code/bigcode-dataset/tokenization/convert_to_json.py \
    --lang {LANG}
    "

export LAUNCHER="python \
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
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
"""


if __name__ == "__main__":
    args = get_args()
    languages = [
        "assembly",
        "verilog",
        "markdown",
        "cpp",
        "java",
        "c-sharp",
        "php",
        "assembly",
        "html",
        "c",
        "javascript",
        "python",
        "typescript",
        "ada",
        "haskell",
        "fortran",
        "sparql",
        "antlr",
        "tex",
        "lean",
        "literate-haskell",
        "elm",
        "standard-ml",
        "powershell",
        "stan",
        "matlab",
        "solidity",
        "smalltalk",
        "tcsh",
        "idris",
        "julia",
        "bluespec",
        "visual-basic",
        "java-server-pages",
        "cuda",
        "yacc",
        "racket",
        "thrift",
        "sql",
        "protocol-buffer",
        "elixir",
        "kotlin",
        "vhdl",
        "scheme",
        "tcl",
        "isabelle",
        "prolog",
        "json",
        "restructuredtext",
        "rmarkdown",
        "clojure",
        "r",
        "zig",
        "ruby",
        "batchfile",
        "erlang",
        "stata",
        "xslt",
        "css",
        "augeas",
        "agda",
        "awk",
        "groovy",
        "coffeescript",
        "lua",
        "systemverilog",
        "common-lisp",
        "scala",
        "verilog",
        "dart",
        "maple",
        "shell",
        "alloy",
        "rust",
        "sas",
        "ocaml",
        "go",
        "literate-coffeescript",
        "emacs-lisp",
        "literate-agda",
        "f-sharp",
        "pascal",
        "applescript",
        "glsl",
        "yaml",
        "makefile",
        "perl",
        "mathematica",
        "dockerfile",
        "cmake",
        "github-issues-filtered-structured",
        "jupyter-scripts-dedup-filtered",
        "jupyter-structured-clean-dedup",
        "git-commits-cleaned"
    ]
    for i in range(args.start, args.end + 1):
        language = languages[i]
        print(f"Submitting jobs for experiment on language {language}")
        job_name = f"{language}-json_idx_{i}"
        job = makejob(
            JOB_NAME=job_name,
            LANG=language,
        )
        # submit the job
        print(f"Job for lang {language} ready and saved at jobs/{job_name}.sbatch")
        submit_job(job, job_name)

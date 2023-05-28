# Megatron-smol-cluster

Setting up Megatron-LM on the smol-cluster

## Installation

After having installed miniconda in your user space at `/fsx`. Log into a compute node and follow these steps:

```
# create new env
conda create --name megatron
conda activate megatron
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install packaging
```

```
# clone repos in code folder under fsx user space
cd /fsx/loubna
mkdir code && cd code

git clone https://github.com/bigcode-project/Megatron-LM.git
git clone https://github.com/NVIDIA/apex.git
```

```
# sometimes apex confuses cuda versions
export CUDA_HOME=/usr/local/cuda-11.6
cd code/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Extra dependencies:
```
pip install wandb transformers
wandb login
```

## Prepare data

In data folder (e.g `/fsx/loubna/data`), download tokenizer and preprocessed data

```
cd data
git clone https://huggingface.co/bigcode/digit-bytelevel-bpe-jss-v1.1-49152
```

Download tokenized and preprocessed Santacoder data by Megatron-LM from GCP in a folder `data`, install `gcloud` or use `rclone` (already installed).
Other `slurm` examples can be found in `examples` folder.


## Prepare slurm file

A slurm file for submitting a job is `train.slurm` adapted from [brrr/examples/t5](https://github.com/huggingface/brrr/blob/main/examples/t5/train.slurm) by Thomas Wang, it must be placed inside `Megatron-LM`, change it accordingly and run:
`sbatch train.slurm`

If the job is terminated, check the logs to find termination reason. You might get some of the following errors:

- If you get `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`: Go to your `~/.bashrc`and comment these lines:

```bash
# If not running interactively, don't do anything
# case $- in
#     *i*) ;;
#       *) return;;
# esac
```

- If you get errors about missing C++ libraries like `pybind11` and `ninja`, run

```
conda install -c conda-forge pybind11
conda install -c conda-forge ninja
```

- If you get an error like:

```
21 | #include <cuda_profiler_api.h>
[ip-26-x]:      |          ^~~~~~~~~~~~~~~~~~~~~
[ip-26-x]:compilation terminated.
```
Make sure the correct cuda is loaded by adding this to your `~/.bashrc`:
```
export PATH="/usr/local/cuda-11.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
```
## Temporary directory
If you keep saving temporary files at `/tmp` or `admin`, you might run out of disk space, you can change `TMPDIR` to `/scratch` in the compute nodes with has 6T or to `/fsx` and clean it manually from time to time. (TODO: issue when adding `mkdir` to slurm job to create a folder in `scratch` , it seems to only create it in the first node):
```bash
# add to ~/.bashrc
export TMPDIR=/fsx/loubna/deleteme/tmp
# for wandb cache
export WANDB_CACHE_DIR=/fsx/loubna/deleteme/wandb
```

## Monitoring
Check your wandb board :rocket: or run this to check GPU utilization of your nodes:
```bash
# get jobid with squeue
NODES=$(scontrol show hostname `squeue -j JOBID --noheader -o %N`)
for ssh_host in $NODES
do
  echo $ssh_host
  ssh -q $ssh_host "nvidia-smi --format=csv --query-gpu=utilization.gpu,utilization.memory"
done
```
## Submitting multiple jobs
In `scaling_laws` folder you can find a python script to submit multiple slurm jobs based on different parameters from a `csv` file we used in our scaling laws experiments.
```bash
python3 submit_jobs.py
```
## Some useful slurm commands:

### Slurm basics:
`sinfo` view information about Slurm nodes and partitions

`squeue` view informatioon about running jobs

`scancel jobid` cancel a running job

`scontrol show jobid` display job state

### Other commands
To show full names of jobs (they are long and get truncated with `squeue`):
```
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```

To cancel all jobs from 2600 to 2640:
```
scancel $(seq 2675 2742)
```
The script for submitting multiple jobs saves many logs, to find latest modified log file of the job at index 39 in the spreadsheet, for example, do:
```
log_path=$(find logs -name "*idx_39-*" -printf '%T@ %f\n' | sort -rn | head -n 1 | cut -d ' ' -f 2-)
tail -f logs/$log_path
```
To access a compute node in bash
```
srun --nodes=1 --cpus-per-task=48 --gres=gpu:4 --mem-per-cpu=11G --partition=production-cluster --pty bash
```

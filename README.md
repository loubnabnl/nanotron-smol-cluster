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
# sometimes apex confuses cuda versions
export CUDA_HOME=/usr/local/cuda-11.6

# clone repos in code folder under fsx user space
cd /fsx/loubna
mkdir code && cd code

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```
git clone https://github.com/bigcode-project/Megatron-LM.git
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

Download preprocessed data from GCP in a folder `data`, install `gcloud` or use `rclone` (already installed).

## Prepare slurm file

A slurm file for submitting a job is `train.slurm`, it must be placed inside `Megatron-LM`, change it accordingly and run:
`sbatch train.slurm`

If the job is terminated, check the logs to find termination reason. You might get some of the following errors:

- If you get `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`: Go to your `~/.bashrc`and comment these lines:

```
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

- If you get:

```
21 | #include <cuda_profiler_api.h>
[ip-26-x]:      |          ^~~~~~~~~~~~~~~~~~~~~
[ip-26-x]:compilation terminated.
```
add
```
export PATH="/usr/local/cuda-11.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
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

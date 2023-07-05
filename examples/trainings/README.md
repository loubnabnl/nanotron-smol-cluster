Notes on the available code and natural language datasets, running jobs on the cluster and their evaluation

# Tokenized datasets
### The Pile & SlimPajama
The Raw and tokenized SlimPajama and The Pile are on S3 under `bigcode-experiments` bucket. The tokenizers were based on GPT2 tokenizer with digit splitting trained on sample of each dataset. 
The tokenizers are in `bigcode-data` org including a tokenizer for RedPajama dataset

### RedPajama
RedPajama is available on s3 under the bucket `hf-redpajama`, a better structured version ready for tokenization is at `/fsx/loubna/data/redpajama_lines`, where
data is sharded to ~100 shards tokenization doesn't OOM. One missing subset is the github data, the data on s3 seems corrupted, so one needs to download it and shard it separately.

- "arxiv": 3 shards arxiv_0, arxiv_1 and arxiv_2
- "c4-train": 20 shards
- "en_head": 30 shards (from CommonCrawl)
- "en_middle": 30 shards (from CommonCrawl)
- "stackExchange", "book": one shard each
- "github": missing

#### Tokenize RedPajama
To run tokenization on each shard you can use (check `tokenize_redpajama.slurm`):
````
# add github, book and stackexchange
OUT_PATH=/fsx/loubna/data/redpajama_tokenized
declare -A folders_dict

folders_dict=(
    ["arxiv"]=2
    ["c4-train"]=19
    ["en_head"]=29
    ["en_middle"]=29
)

for prefix in "${!folders_dict[@]}"; do
    num_subsets=${folders_dict[$prefix]}
    for i in $(seq 0 $num_subsets); do
        echo "Tokenizing shard $i from the $num_subsets shards in $prefix" 
        rm -rf OUT_PATH/$prefix\_$i
        sbatch -J $i-$prefix-tokenize tokenize_redpajama.slurm $prefix\_$i
    done
done
````
# Jobs

#### Training with `brrr`
Running jobs:
- 1.3b-pile: 1.3B model with MQA on 300B from The Pile (If the job crashes it can be restarted with 
````
sbatch /fsx/loubna/code/dev/pr/brrr/examples/gpt2_mqa/slurm_scripts/train_loubna_pile_1b.slurm
`````

Jobs to run:
- 1.3b-slimpajama: 1.3B model with MQA on 300B from SlimPajama
Download tokenized slimpajama from S3. And you can find `config_slimpajama_1_3b.yaml` and `train_slimpajama_1_3b.yaml` to 
be used with [brrr](https://github.com/huggingface/brrr/tree/main/brrr) after updating data/ckpt/tensorboard paths..
- 1.3b-redpajama: same

**Model conversion**

```bash
python /fsx/loubna/code/brrr/examples/gpt2_mqa/convert_checkpoint.py
    --checkpoint-path /fsx/loubna/br4-experiments/checkpoints/pile_2/300000 \
    --model-name bigcode-data/pile-1.3b \
    --save-path /fsx/loubna/br4-experiments/checkpoints/pile/converted
```

#### With Megatron-LM: 7B-StarCoder
The slurm file for the job is at `/fsx/loubna/code/Megatron-LM/train_7b.slurm`, checkpoints are saved at `/fsx/bigcode/experiments/pretraining/7b-starcoder`.

# Evaluation of 7B-StarCoder on MultiPL-E
Once the checkpoint is converted and pushed to the `bigcode-data` org
You can run MultiPL-E evaluation using: (`/fsx/loubna/code/bigcode-evaluation-harness/multiple_evals.sh`) 
```bash

langs=(java js cpp swift php d jl lua r rkt rb rs py)

model=starcoderbase-7b
for lang in "${langs[@]}"; do
    sbatch -J "eval-$model-$lang" /fsx/loubna/code/bigcode-evaluation-harness/multiple_evals.slurm "$model" "$lang"
done
```
(=`/fsx/loubna/code/bigcode-evaluation-harness/multiple_evals.sh`) 

This will run `/fsx/loubna/code/bigcode-evaluation-harness` and save the generations under `$OUT_PATH//$model-$lang.json`  (see `multiple_evals.slurm`)
The execution requires docker containers (only available on `dev-cluster`)
```bash
# get a dev-node with many cpus
srun --nodes=1 --cpus-per-task=96 --partition=dev-cluster --job-name=eval --time 5:00:00 --pty bash
cd /fsx/loubna/code/bigcode-evaluation-harness
# pull eval-aherness image
sudo docker pull ghcr.io/bigcode-project/evaluation-harness-multiple
sudo docker tag ghcr.io/bigcode-project/evaluation-harness-multiple evaluation-harness-multiple
````
Run execution with (each language takes 2-10 minutes):
```bash
#bash /fsx/loubna/code/bigcode-evaluation-harness/multiple_gens/container_eval.sh

langs=(java js cpp swift php d jl lua r rkt rb rs py
model=starcoderbase-7b
org=bigcode-data

for lang in "${langs[@]}"; do
    suffix=$model-$lang.json
    echo "Evaluation of $model on $lang benchmark, data in $suffix"
    gens_path=/fsx/loubna/code/bigcode-evaluation-harness/multiple_gens/$suffix
    sudo docker run -v $gens_path:/app/$suffix:ro -it evaluation-harness-multiple python3 main.py \
        --model /fsx/loubna/code/bigcode-evaluation-harness/replit-code-v1-3b \
        --tasks multiple-$lang \
        --load_generations_path /app/$suffix \
        --allow_code_execution  \
        --use_auth_token \
        --temperature 0.2 \
        --n_samples 50 | tee -a /fsx/loubna/code/bigcode-evaluation-harness/multiple_gens/logs_1b.txt
    echo "$lang Done"
done
```

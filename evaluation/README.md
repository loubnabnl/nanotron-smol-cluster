## HumanEval

Given HumanEval generations, we want to run post-processing to remove extra code and then execute them using `bigcode-evaluation-harness` to compute pass@1.

### Setup
Clone and install [bigcode harness](https://github.com/bigcode-project/bigcode-evaluation-harness) in your env,
```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
```

### Solutions post-processing
```bash
# to clean the generations and save them in bigcode-eval-harness format
# 7b_513b_tokens.json is generations from nvidia
python clean_solutions.py --load_path ./7b_513b_tokens.json --save_path bigcode-evaluation-harness/7b_513b_tokens_clean.json
```

### Solutions Execution
You can run execution with slurm using (provide name of json, change paths in slurm accordingly):
```
sbatch slurm_eval_nvidia.slurm  7b_513b_tokens_clean
```

Or just run this command instead on a cpu machine (given you have 20 geerations per problem):

```python
python bigcode-evaluation-harness/main.py \
    --model $model_name \
    --tasks humaneval \
    --n_samples 20 \
    --allow_code_execution \
    --use_auth_token \
    --load_generations_path bigcode-evaluation-harness/7b_513b_tokens_clean.json \
    --metric_output_path bigcode-evaluation-harness/metrics_7b_513b_tokens.json \
```

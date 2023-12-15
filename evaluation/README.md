## HumanEval

Given HumanEval generations, we want to run post-processing to remove extra code and then execute them using `bigcode-evaluation-harness` to compute pass@1.

### Solutions post-processing
```bash
# to clean the generations and save them in bigcode-eval-harness format
# update paths in the python script
python /fsx/loubna/ckpts/clean_solutions.py --load_path 7b_513b_tokens.json --save_path 7b_513b_tokens.json
```

### Solutions Execution
Assuming you have installed [bigcode harness](https://github.com/bigcode-project/bigcode-evaluation-harness) in your env, you can run execution with slurm using:
```
sbatch /fsx/loubna/ckpts/slurm_eval_nvidia.slurm  7b_513b_tokens
```

Or just run this command on a cpu machine (given you have 20 geerations per problem):

```python
python /fsx/loubna/bigcode_2/code/bigcode-evaluation-harness/main.py \
    --model $model_name \
    --tasks humaneval \
    --n_samples 20 \
    --allow_code_execution \
    --use_auth_token \
    --load_generations_path /fsx/loubna/bigcode_2/code/bigcode-evaluation-harness/nvidia/$load_path.json \
    --metric_output_path $save_path/metric_$load_path.json \
```

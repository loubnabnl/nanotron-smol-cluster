# Converting BigCode 15B Megatron-LM checkpoint to transformers

Clone `Megatron-LM` and `transformers` from `bigcode-project`, and run:
```
# merge TP and PP partitions
bash merge_bigcode_partitions.sh
# convert the merged checkpoint to transformers and push as shards to the hub
bash convert_ckpt.sh
```

Below is `merge_bigcode_partitions.sh`
```
# example for iter_200000
source ~/.bashrc
conda activate megatron

OUTPUT_PATH=/fsx/bigcode/experiments/pretraining/conversions/6672

python /fsx/loubna/code/new/Megatron-LM/tools/checkpoint_util.py \
        --model-type GPT  \
        --load-dir /fsx/bigcode/experiments/pretraining/6672/ \
        --save-dir $OUTPUT_PATH \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --use-distributed-optimizer | tee $OUTPUT_PATH/checkpoint_util_200k.log
```

Below is `convert_ckpt.sh`
```bash 
source ~/.bashrc
conda activate megatron
export PYTHONPATH=/fsx/loubna/code/new/Megatron-LM
cd transformers/src/transformers/models

python -m megatron_gpt_bigcode.push_checkpoints \
    --exp_dir /fsx/bigcode/experiments/pretraining/conversions/6672 \
    --repo_name bigcode/large-model \
    --branch_name main \
    --iter_interval 200000
```
We changed this code block in `transformers/src/transformers/models/megatron_gpt_bigcode.push_checkpoints` the code to load the model and push shards to the hub (it's too large can't push it in one shard).

```python
if iter_number % args.iter_interval == 0:
    print(f"Converting iteration {iter_number} and pushing with commit {ckpt_dir.name}")
    # TODO: this only works for 1-way tensor/pipeline parallelism
    file_path = next((ckpt_dir / "mp_rank_00").glob("*.pt"))
    convert(argv + [f"--save_dir={str(save_dir)}", str(file_path)])
    print("Load with transformers")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(save_dir)
    print("Model loaded and is being sharded and save to disk")
    model.save_pretrained(PUSH_DIR)
    print("Adding tokenizer files")
    copy_tokenizer_files(save_dir)
    print(f"Local copy saved at {PUSH_DIR}, now pushing:")
    # usually fails
    model.push_to_hub("bigcode/large-model", commit_message=f"{ckpt_dir.name}")


# Tokenization fo The Stack V2 Dataset (code & Extra source)

## Sharding
If the dataset is very large we'll need to shard it to smaller subsets (usually <40G each) so we don't get OOM (be careful sometimes Megatron-LM tokenization might fail silently, so make sure the final `.bin` and `.idx` files both exist for each shard and have reasonable sizes e.g if idx is too small check the logs something must've gone wrong).

- change the folder names and wanted number of shards in `python_scripts/shard_subsets.py` script

Copy code dataset into a folder named `code` here `/fsx/bigcode/bigcode-training/stack_v2`, this code will convert it to multiple folders/shards e.g `code_0 code_1...`
```bash
DATA_PATH="/fsx/bigcode/bigcode-training/stack_v2/"
cd $DATA_PATH
python /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/python_scripts/shard_subsets.py
```

## Convert shards to jsonl files
Megatron-LM exepcts as input a jsonl file for tokenization (can also use datasets but it's slow so we convert all shards to jsonl files):
Note that you might need to change the logs path in slurm file `slurm_files/create_json.slurm`

You can update the `folders_dict` variable  in `run_json_create.sh` to specify which folders to convert to jsonl files (each data source is a prefix in the dict with its corresponding number of shards)

```bash
bash /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/run_json_create.sh
```

## Tokenize
And now tokenize each shard:
```bash
bash /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/run_tokenization.sh
```

Note: 
- be careful sometimes Megatron-LM tokenization might fail silently, so make sure the final `.bin` and `.idx` files both exist for each shard and have reasonable sizes e.g if idx is too small check the logs something must've gone wrong), some sanity checks in `sanity.sh`

- To use tokenized data in training make sure to use appropriate custom weights (will probably be provided in a csv, fornmost teh wieght is the dataset size in  GB unless it's beging up/down-sampled), if you shard a data source make sure to divide weight of the shard by number of shards in that source (=> double check weights with Raymond and team).


## Update:
The raw code dataset will be saved here `/fsx/bigcode/data_v2` (and tokenized one here `/fsx/bigcode/bigcode-training/tokenized_stack_v2_final`). To run tokenization:
1- Update the bashrc path in `./slurm_files/tokenize_stackv2_anton.slurm`
2- Go to `./tokenize_anton_data.sh` and update `folders_dict` with the right dataset name (instead of `jupyter_scripts`) and number of shards.
3- Run :
```bash
cd  ./data_preparation/stack_v2
# launch slurm tokenization jobs
bash ./tokenize_anton_data.sh
```
4- You're done! 🎉 (make sure to check the logs for errors, and verify data sizes at the end)
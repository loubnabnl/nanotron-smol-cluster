# Data preparation for Megatron-LM training
## Ading metadata
The dataset avialable at [stack-march-no-pii](https://huggingface.co/datasets/bigcode/stack-march-no-pii) already has metadata added to the programming languages with special tokens.

## Data tokenization

We tokenize each data source separately. The dataset is avialable at [stack-march-no-pii](https://huggingface.co/datasets/bigcode/stack-march-no-pii), it contains 88 programming languages + Jupyter notebooks + Jupyter scripts + GitHub Issues + GitHub commits.

Beware that you might need up to 1.5TB of RAM to tokenize some data source with a large number of files like Markdown and Java, otherwose you will need to shard them.

Tokenization with megatron-LM can be run with the following command:

```bash
DATA_PATH=..
OUT_PATH=..
LANG=..
python /fsx/loubna/code/Megatron-LM/tools/preprocess_data.py \
    --input $DATA_PATH/{LANG} \
    --output-prefix $OUT_PATH/gpt2-preprocessed \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file /fsx/bigcode/bigcode-training/tokenizer-starcoder/tokenizer.json \
    --dataset-impl mmap \
    --append-eod \
    --json-keys content \
    --workers 64 \
    --chunk-size 100 \
    --log-interval 1000
```
We have a python script for submitting tokenization jobs on all resources at `tokenization/tokenize_slurm.py` (be careful of jobs failing silently because of OOM, check logs + final size of `idx` and `bin` files).

```bash
python tokenization/tokenize_slurm.py --start 0 --end 89
# pass --extra_languages flag for issues, jupyter & commits and --out_path "/fsx/bigcode/bigcode-training/tokenized_stack_no_pii/"
```

## Data weighting

You need to clone [bigcode-data-mix](https://github.com/loubnabnl/bigcode-data-mix/tree/new-data) (branch new-data) and follow readme instruction to create data paths defining weights for training.
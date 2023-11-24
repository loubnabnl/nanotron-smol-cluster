from datasets import load_dataset
from argparse import ArgumentParser
import os


# convert each shard into a jsonl file for tokenization (keep / at the end)
LOAD_PATH = "/fsx/bigcode/bigcode-training/stack_v2/"
SAVE_PATH = "/fsx/bigcode/bigcode-training/stack_v2_jsonl/"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
    )
    return parser.parse_args()


args = parse_args()

print(f"Loading dataset {LOAD_PATH + args.prefix} ")
ds = load_dataset(LOAD_PATH + args.prefix, split="train", num_proc=24)
ds.to_json(f"{SAVE_PATH + args.prefix}/data.jsonl", lines=True)
print(f"Dataset {args.prefix} saved at {SAVE_PATH + args.prefix}/data.jsonl")

# python3 /fsx/loubna/code/extreme-scale/data/redpajama/create_one_jsonl.py --prefix code_2

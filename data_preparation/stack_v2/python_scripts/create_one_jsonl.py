from datasets import load_dataset
from argparse import ArgumentParser
import os

LOAD_PATH = "/fsx/loubna/data/stack_v2_smol_all_json/"
SAVE_PATH = "/fsx/loubna/data/stack_v2_smol_all_jsonl/"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
    )
    return parser.parse_args()


args = parse_args()

folders_dict = {
    #"apps_codecontest": 1,
    "issues": 3,
    #"code": 40,
    "jupyter_notebooks": 2,
    "jupyter_scripts": 2,
    "kaggle": 1,
}

for folder, num_of_subfolders in folders_dict.items():
    if num_of_subfolders == 1:
        prefix = folder
        data_path = f"{LOAD_PATH}{prefix}"
        print(f"Load and dump {data_path} into one jsonl file")
        ds = load_dataset(data_path, split="train", num_proc=24)
        save = f"{SAVE_PATH}{prefix}/data.jsonl"
        os.makedirs(f"{SAVE_PATH}{prefix}", exist_ok=True)
        ds.to_json(save, lines=True)
        print(f"Dataset {prefix} saved at {save}")
    else:
        for i in range(num_of_subfolders):
            prefix = f"{folder}_{i}"
            data_path = f"{LOAD_PATH}{prefix}"
            print(f"Load and dump {data_path} into one jsonl file")
            ds = load_dataset(data_path, split="train", num_proc=12)
            save = f"{SAVE_PATH}{prefix}/data.jsonl"
            os.makedirs(f"{SAVE_PATH}{prefix}", exist_ok=True)
            ds.to_json(save, lines=True)
            print(f"Dataset {prefix} saved at {save}")

# python3 /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/create_one_jsonl.py --prefix code_2

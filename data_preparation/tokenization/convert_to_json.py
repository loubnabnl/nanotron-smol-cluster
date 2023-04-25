from datasets import load_dataset
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str,
                        default=""
                        )
    parser.add_argument("--output-path", type=str,
                        default="/fsx/loubna/data/stack_march_no_pii_json"
                        )
    args = parser.parse_args()
    return args

args = get_args()
dataset = load_dataset(f"/fsx/loubna/data/the-stack-march-no-pii/{args.lang}", data_dir="data/", split='train', num_proc=64)
dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'content'])
dataset.to_json(f"{args.output_path}/{args.lang}/data.json", lines=True)

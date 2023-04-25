import argparse
from multiprocessing import Pool
from tqdm import tqdm

from datasets import load_dataset


def save_shard(shard_tuple):
    """Save shard"""
    filename, shard = shard_tuple
    # use to_json instead to save as json file
    shard.to_json(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str,
                        # default="bigcode/github-issues-filtered-structured"
                        )
    parser.add_argument("--output-file", type=str,
                        # default="/data/march_datasets/github-issues-filtered-structured-full_text.jsonl"
                        )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--only-keep-col", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    num_shards = args.num_shards

    dataset = load_dataset(args.dataset_name, use_auth_token=True, data_dir=args.data_dir)

    if args.only_keep_col is not None:
        # Remove all columns except `args.only_keep_col`
        remove_cols = dataset.column_names['train']
        remove_cols.remove(args.only_keep_col)
        dataset = dataset.remove_columns(remove_cols)

    dataset = dataset['train']

    if num_shards > 1:
        print(f"will shard dataset into {num_shards} files")
        shards = (
            dataset.shard(num_shards=num_shards, index=i, contiguous=True)
            for i in range(num_shards)
        )
        filenames = (
            f"{args.output_file}-{index:05d}-of-{num_shards:05d}.jsonl"
            for index in range(num_shards)
        )
        with Pool(8) as p:
            list(
                tqdm(
                    p.imap_unordered(save_shard, zip(filenames, shards)),
                    total=num_shards,
                )
            )
    else:
        dataset.to_json(args.output_file)
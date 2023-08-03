import logging
from argparse import ArgumentParser

from datasets import load_dataset
from huggingface_hub import HfFileSystem


# add get_arguments for data_dir and data_name
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--local_data_dir",
        type=str,
        default="/fsx/loubna/code/extreme-scale/data/slimpajama_shards",
        help="Path to the directory where the shards will be saved",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="cerebras/SlimPajama-627B",
        help="Name of the dataset on the hub",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="/fsx/loubna/code/extreme-scale/data/logs/shard_slim_pajama_data.log",
        help="Path to the log file",
    )
    parser.add_argument(
        "--split_number",
        type=int,
        default=3,
        help="Number of shards to split each chunk into",
    )
    parser.add_argument(
        "--chunk_number",
        type=int,
        default=10,
        help="Number of chunks in the dataset",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of processes to use for loading the dataset",
    )
    return parser.parse_args()


args = parse_args()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(args.log_file),
        logging.StreamHandler(),
    ],
)
logger.info(f"Sharding of SlimPajama dataset")

# use HF filesystem
fs = HfFileSystem()

# the dataset has args.chunk_number shards we'll make 10*args.split_number shards out of it
total_size_check = 0
t = 20
for i in range(6, 7):
    list_dirs = fs.ls(f"datasets/{args.data_name}/train/chunk{i}", detail=False)
    real_dirs = [
        f"https://huggingface.co/datasets/{args.data_name}/resolve/main/train/{list_dirs[i].split('/train/')[-1]}"
        for i in range(len(list_dirs))
    ]
    count = len(real_dirs)
    proportion = count // args.split_number
    logger.info(
        f"Splitting chunk {i} to {args.split_number} shards with proportion {proportion} files each out of {count}"
    )
    start = 0
    if i == 7:
        start = 1
    if i == 9:
        start = 2
        t += 2
        logger.info(f"Chunk {i} parsing skipped shards 0 and 1 (aka 32 and 33), we'll start from shard {t}")
    for j in range(start, args.split_number):
        logger.info(f"Preparing shard {j} in chunk {i} saved to shard_{t}")
        if j == args.split_number - 1:
            # last shard
            ds = load_dataset(
                "json", data_files=real_dirs[proportion * j :], split="train", num_proc=args.num_proc
            )
            total_size_check += len(ds)
            logger.info(
                f"Shard {j} makes dataset:\n{ds}, will be saved to {args.local_data_dir}/shard_{t}"
            )
            ds.to_json(f"{args.local_data_dir}/shard_{t}.json", lines=True)
            t += 1
        else:
            # intermediate shards
            ds = load_dataset(
                "json",
                data_files=real_dirs[proportion * j : proportion * (j + 1)],
                split="train",
                num_proc=args.num_proc,
            )
            total_size_check += len(ds)
            logger.info(
                f"Shard {j} makes dataset:\n{ds}, will be saved to {args.local_data_dir}/shard_{t}"
            )
            ds.to_json(f"{args.local_data_dir}/shard_{t}.json", lines=True)
            t += 1
    logger.info(f"Chunk {i} done, current total size is {total_size_check} samples")

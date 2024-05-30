import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, Repository, create_repo, upload_folder

# cd /fsx/loubna/checkpoints/clones
# python /fsx/loubna/logs/ablations_v2/350B/upload.py --repo_name HuggingFaceTB/ablation-model-fineweb-edu --save_dir /fsx/loubna/checkpoints/clones


def copy_folder_contents(source: Path, destination: Path):
    print(f"Copying {source} to {destination}")
    if not destination.exists():
        destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_dir():
            shutil.copytree(item, destination / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination / item.name)


def get_iter_number(iter_dir: str):
    m = re.match(r"(\d+)", iter_dir)
    if m is not None:
        return int(m.group(1))
    else:
        raise ValueError(f"Invalid directory name: {iter_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=Path,
        help="Path to experiment folder.",
        default="/fsx/loubna/checkpoints/fineweb_edu_350B_converted",
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Name of repository on the Hub in 'ORG/NAME' format.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Path where repository is cloned to locally. Will use {exp_dir}/hf_checkpoints if not provided",
    )
    parser.add_argument(
        "--iter_interval",
        type=int,
        default=10000,
        help="Iteration number must be divisible by iter_interval in order to be pushed",
    )
    args, argv = parser.parse_known_args(argv)

    save_dir = args.save_dir or args.exp_dir / "hf_checkpoints"
    save_dir = Path(f"{save_dir}/{args.repo_name.split('/')[-1]}")

    api = HfApi()
    create_repo(args.repo_name, exist_ok=True)

    ckpt_dirs = sorted(
        [
            x
            for x in Path(args.exp_dir).iterdir()
            if re.match(r"(\d+)", x.name) and x.is_dir()
        ],
        key=lambda p: get_iter_number(p.name),
    )
    print(f"Found the following checkpoints: {ckpt_dirs}")

    for ckpt_dir in ckpt_dirs:
        iter_number = get_iter_number(ckpt_dir.name)
        if iter_number % args.iter_interval == 0 and iter_number > 0:
            dest_dir = save_dir
            copy_folder_contents(ckpt_dir, dest_dir)
            if not any(dest_dir.iterdir()):
                print(f"Failed to copy checkpoint files to {dest_dir}")
                continue

            print(f"Uploading iteration {ckpt_dir.name} to the hub")
            upload_folder(
                repo_id=args.repo_name,
                folder_path=str(dest_dir),
                commit_message=f"{ckpt_dir.name}",
            )


if __name__ == "__main__":
    main()

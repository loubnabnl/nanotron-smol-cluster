import argparse
import re
import subprocess
from pathlib import Path
from huggingface_hub import Repository

# python /fsx/loubna/logs/ablations_v2/350B/upload.py --repo_name HuggingFaceTB/test_conversion --branch_name checkpoints --save_dir /fsx/loubna/checkpoints/clones
def get_iter_number(iter_dir: str):
    m = re.match(r"(\d+)", iter_dir)
    if m is not None:
        return int(m.group(1))
    else:
        raise ValueError(f"Invalid directory name: {iter_dir}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, help="Path to experiment folder.", default="/fsx/loubna/checkpoints/fineweb_edu_350B_converted")
    parser.add_argument("--repo_name", required=True, help="Name of repository on the Hub in 'ORG/NAME' format.")
    parser.add_argument("--branch_name", required=True, help="Name of branch in repository to save experiments.")
    parser.add_argument("--save_dir", type=Path, help="Path where repository is cloned to locally. Will use {exp_dir}/hf_checkpoints if not provided")
    parser.add_argument("--iter_interval", type=int, default=10000, help="Iteration number must be divisible by iter_interval in order to be pushed")
    args, argv = parser.parse_known_args(argv)

    save_dir = args.save_dir or args.exp_dir / "hf_checkpoints"

    hf_repo = Repository(save_dir, clone_from=args.repo_name)
    hf_repo.git_checkout(args.branch_name, create_branch_ok=True)

    head_hash = hf_repo.git_head_hash()
    commit_msg = subprocess.check_output(["git", "show", "-s", "--format=%B", head_hash], cwd=save_dir).decode()
    try:
        last_commit_iter = get_iter_number(commit_msg.strip())
        print(f"Last commit iteration: {last_commit_iter}")
    except ValueError:
        last_commit_iter = -1

    ckpt_dirs = sorted(
        [x for x in args.exp_dir.iterdir() if re.match(r"(\d+)", x.name) and x.is_dir()],
        key=lambda p: get_iter_number(p.name)
    )
    print(f"Found the following checkpoints: {ckpt_dirs}")

    for ckpt_dir in ckpt_dirs:
        iter_number = get_iter_number(ckpt_dir.name)
        if iter_number <= last_commit_iter:
            continue
        if iter_number % args.iter_interval == 0:
            print(f"Pushing iteration {iter_number}")
            hf_repo.push_to_hub(commit_message=f"{ckpt_dir.name}")

if __name__ == "__main__":
    main()

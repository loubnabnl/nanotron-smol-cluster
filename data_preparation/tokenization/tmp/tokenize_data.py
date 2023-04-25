"""
Launch data preprocessing for BigCode March training
"""

from datetime import datetime
import subprocess
from pathlib import Path


OUT_PATH = Path("/data/march_datasets/fixed_tokenizer_2")  # Adjust


def code_args():
    languages = Path("/data/hf_repos/the_stack_march_training_with_meta/data/").glob(
        "*"
    )  # Adjust
    languages = [l.stem for l in languages]
    return [
        (
            "/data/hf_repos/the_stack_march_training_with_meta",
            f"--subset data/{l}",
            OUT_PATH / "code" / l,
            "content_with_meta",
        )
        for l in languages
    ]


def notebook_args():
    return [
        (
            "bigcode/jupyter-scripts-dedup-filtered",
            "",
            OUT_PATH / "jupyter_scripts",
            "content",
        ),
        (
            "bigcode/jupyter-structured-clean-dedup",
            "",
            OUT_PATH / "jupyter_structured",
            "content",
        ),
    ]


def gh_issues_args():
    return [
        (
            "bigcode/github-issues-filtered-structured",
            "",
            OUT_PATH / "gh_issues",
            "full_text",
        )
    ]


# If the job tokenizing the issues runs out of memory,
# first need to shard the issues locally
# python download_dataset.py --dataset-name bigcode/github-issues-filtered-structured --only-keep-col full_text  --output-file /data/march_datasets/gh_issues_sharded --num-shards 8
# then run the following jobs
def gh_issues_sharded_args():
    num_shards = 8
    return [
        (
            f"/data/march_datasets/gh_issues_sharded-{i:05d}-of-{num_shards:05d}.jsonl",
            "",
            OUT_PATH / f"gh_issues_sharded_{i}",
            "full_text",
        )
        for i in range(num_shards)
    ]


def gh_commits_args():
    return [("bigcode/git-commits-cleaned", "", OUT_PATH / "gh_commits", "content")]


def all_args():
    # return code_args() + notebook_args() + gh_issues_args() + gh_commits_args()
    return code_args() + notebook_args() + gh_issues_sharded_args() + gh_commits_args()
    # return gh_issues_sharded_args()


job_command = "mkdir -p {output} && python3 Megatron-LM/tools/preprocess_data.py      --input {dataset_name}   {subset_arg}     --output-prefix {output}/gpt2-preprocessed    --tokenizer-type TokenizerFromFile    --tokenizer-file /data/tokenizers/tokenizer-the-stack-march-sample-v3-no-prefix-spaces/tokenizer.json        --dataset-impl mmap             --append-eod --json-keys {json_key} --workers 64 --chunk-size 100 --log-interval 1000"


if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    args_list = all_args()
    print(args_list)

    for i, args in enumerate(args_list):
        dataset_name, subset_arg, output, json_key = args
        formatted_dataset_name = Path(dataset_name).stem.replace("-", "_")

        # Adjust to whatever tool is used to launch jobs
        toolkit_command = [
            "submit-job",
            "--name",
            f"megatron_preprocess_{dt_string}_{formatted_dataset_name}_{i}",
            "--mem",
            "256",
            "--cpu",
            "64",
            "--preemptable",
            "--",
            "bash",
            "-c",
            job_command.format(
                dataset_name=dataset_name,
                subset_arg=subset_arg,
                output=output,
                json_key=json_key,
            ),
        ]
        subprocess.run(toolkit_command)
        # break

import os

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


DATASET_NAME = "stack_v2_train_json"

data_sources = ["pull_requests",
    "stackoverflow",
    "owm",
    "lhq_data",
    "kaggle_scripts",
    "jupyter_structured",
    "jupyter_scripts",
    "issues",
]
data_sources = ["stack_3b"]

N_TASKS = 256
logs_file = "/fsx/loubna/logs/tokenize_mistral"
data_path = f"/fsx/bigcode/data/stack_tokenize_mistralr/stack_3b"
os.makedirs(data_path, exist_ok=True)
print(f"tokenizing with {N_TASKS} logs at /fsx/loubna/logs/tokenize_mistral")
print(f"data path is {data_path}")

dist_executor = SlurmPipelineExecutor(
    job_name=f"tokenize",
    pipeline=[
        JsonlReader(
            f"s3://bigcode-datasets-us-east-1/stack_v2_train_json/stack_3b/",
            text_key="content",
        ),
        DocumentTokenizer(
            output_folder=data_path,
            tokenizer_name="mistralai/Mistral-7B-v0.1",
            save_filename=f"code",
            eos_token="</s>",
        ),
    ],
    tasks=N_TASKS,
    time="20:00:00",
    partition="hopper-cpu",
    logging_dir=logs_file,
    cpus_per_task=20,
    mem_per_cpu_gb=4,
)
dist_executor.run()





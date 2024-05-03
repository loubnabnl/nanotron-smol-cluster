import os


from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader, HuggingFaceDatasetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


N_TASKS = 300
N_WORKERS = None

subset = "stanford_openstax_wiki_repro_2B_mixtral1_sampling"
data_path = "HuggingFaceTB/stanford_openstax_wiki_repro_2B_mixtral1_sampling"
target = "completion"

logs_file = f"/fsx/loubna/logs/tokenization_cosmopedia/tokenize_{subset}"
out_path = f"s3://synthetic-datasets-phi/ablations/{subset}/tokenized/tokenized/"
out_path_merged = f"s3://synthetic-datasets-phi/ablations/{subset}/tokenized/standard/"
local_working_dir = f"/fsx/loubna/data/tokenization/{subset}/"


dist_executor = SlurmPipelineExecutor(
    job_name=f"tokenize-{subset}",
    pipeline=[
        HuggingFaceDatasetReader(
            data_path,
            dataset_options={
                "split": "train"
            },
            text_key=target
        ),
        DocumentTokenizer(
            output_folder=out_path,
            tokenizer_name="lvwerra/the-tokenizer-v1",
            batch_size=10_000,
            local_working_dir=local_working_dir, # shuffle and reorganize tokenized files locally (faster)
        ),
    ],
    tasks=N_TASKS,
    workers=N_WORKERS,
    time="20:00:00",
    partition="hopper-cpu",
    logging_dir=logs_file,
    cpus_per_task=96,
    mem_per_cpu_gb=3,
)
dist_executor.run()


merge_executor = SlurmPipelineExecutor(
    job_name=f"tok-{subset}-merge",
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=out_path,
            output_folder=out_path_merged,
            save_filename=f"{subset}",
        ),
    ],
    tasks=1,
    time="50:00:00",
    partition="hopper-prod",
    logging_dir=logs_file+"_merged",
    mem_per_cpu_gb=20,
    cpus_per_task=96,
    depends=dist_executor,
)

merge_executor.run()


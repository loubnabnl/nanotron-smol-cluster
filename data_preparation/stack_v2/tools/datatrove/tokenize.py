from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


print(f"tokenizing code")
dist_executor = SlurmPipelineExecutor(
    job_name=f"tokenize",
    pipeline=[
        JsonlReader(
            f"s3://bigcode-datasets-us-east-1/stack_v2_train_json/stack_3b/",
            text_key="content",
        ),
        DocumentTokenizer(
            output_folder=f"/fsx/bigcode/data/stack_tokenisze_mistral/stack_3b/",
            tokenizer_name="mistralai/Mistral-7B-v0.1",
            save_filename=f"code_tokenized",
            eos_token="</s>",
        ),
    ],
    tasks=260,
    time="20:00:00",
    partition="hopper-cpu",
    logging_dir=f"/fsx/loubna/logs/tokenization_cpu/",
    cpus_per_task=24,
    #mem_per_cpu_gb=4,
)
dist_executor.run()

merge_executor = SlurmPipelineExecutor(
    job_name=f"tokenize2",
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=f"/fsx/bigcode/data/stack_tokenisze_mistral/stack_3b/",
            output_folder=f"/fsx/bigcode/data/stack_tokenisze_mistral/standard/stack_3b/",
            save_filename=f"{DATASET_NAME}",
        ),
    ],
    tasks=1,
    time="50:00:00",
    partition="hopper-cpu",
    logging_dir=f"/fsx/loubna/logs/tokenization/tokenize_stack_merged",
    mem_per_cpu_gb=12,
    depends=dist_executor,
)
merge_executor.run()

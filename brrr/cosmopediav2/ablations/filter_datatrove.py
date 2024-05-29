import argparse
import os

# python /fsx/loubna/projects/datatrove/examples/filter_edu_fw_slurm.py hf://datasets/HuggingFaceTB/fineweb_edu_ge_2_5/ fineweb_edu_smol --n_tasks 200

parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument(
    "input_dataset",
    type=str,
    help="Path to the data to tokenize.",
    default="hf://datasets/HuggingFaceTB/fineweb_edu_ge_2_5/",
)
parser.add_argument("output_name", type=str, help="Output name.", default="fineweb_edu_smol")
parser.add_argument("--n_tasks", type=int, help="nb of tokenization tasks", default=100)


if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter

    dist_executor = SlurmPipelineExecutor(
        job_name=f"filter-{args.output_name}",
        pipeline=[
            ParquetReader("hf://datasets/HuggingFaceTB/fineweb_edu_ge_2_5/", glob_pattern="**/*.parquet", text_key="text"),
            LambdaFilter(lambda doc: doc.metadata["int_score"] >= 4),
            HuggingFaceDatasetWriter(
                dataset=f"HuggingFaceTB/{args.output_name}",
                private=True,
                local_working_dir=f"/fsx/loubna/data/{args.output_name}",
                output_filename="data/${rank}.parquet",
                cleanup=True,
            ),
        ],
        tasks=args.n_tasks,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=f"/fsx/loubna/logs/filter_fw/{args.output_name}",
        cpus_per_task=12,
        qos="high",
        mem_per_cpu_gb=3,
    )
    dist_executor.run()

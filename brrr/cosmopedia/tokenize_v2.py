import argparse
import os

parser = argparse.ArgumentParser("Quickly launch thom's style of tokenization.")

# python /fsx/loubna/projects/datatrove/examples/edu_fw.py hf://datasets/bigcode/stackoverflow-clean stackoverflow --n_tasks 50 --tokenizer HuggingFaceTB/cosmo2-tokenizer


parser.add_argument("data_path", type=str, help="Path to the data to tokenize.")
parser.add_argument("output_name", type=str, help="Output name.")
parser.add_argument("--n_tasks", type=int, help="nb of tokenization tasks", default=1000)
parser.add_argument("--max_toks", type=int, help="max tokens per file", default=1e9)
parser.add_argument("--tokenizer", type=str, help="tokenizer to use", default="HuggingFaceTB/cosmo2-tokenizer")
parser.add_argument("--text_key", type=str, default="text")
parser.add_argument("--sample", type=float, default=1.0)
parser.add_argument("--email", type=str, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.readers import ParquetReader, JsonlReader
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger

    dist_executor = SlurmPipelineExecutor(
        job_name=f"tok-{args.output_name}",
        pipeline=[
            ParquetReader(
                args.data_path, # read directly from huggingface
                glob_pattern="**/*.parquet", # "**/*.parquet", 
                text_key=args.text_key,
            ),
            #SamplerFilter(rate=0.5),
            DocumentTokenizer(
                output_folder=f"/fsx/{os.getlogin()}/tokenized_for_exps/cosmo2_training_data/{args.output_name}",
                tokenizer_name_or_path=args.tokenizer,
                batch_size=10000,
                max_tokens_per_file=args.max_toks,
                # Max 1 GT per file (i.e. btw 5 et 300 tokenized files per dump et about 100 dump extracts per merged file)
                shuffle=True,
            ),
        ],
        tasks=args.n_tasks,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=f"/fsx/loubna/logs/tokenize_cosmo2/{args.output_name}",
        cpus_per_task=32,
        qos="high",
        mem_per_cpu_gb=3,
        mail_user=args.email,
    )
    dist_executor.run()

    merge_executor = SlurmPipelineExecutor(
        job_name=f"merge-{args.output_name}",
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=f"/fsx/{os.getlogin()}/tokenized_for_exps/cosmo2_training_data/{args.output_name}",
                output_folder=f"/fsx/{os.getlogin()}/tokenized_for_exps/cosmo2_training_data/{args.output_name}_merged",
                save_filename=f"{args.output_name}",
            ),
        ],
        tasks=1,
        time="50:00:00",
        partition="hopper-cpu",
        logging_dir=f"/fsx/loubna/logs/tokenize_cosmo2/{args.output_name}_merged",
        mem_per_cpu_gb=3,
        cpus_per_task=64,
        qos="high",
        depends=dist_executor,
    )

    merge_executor.run()

import csv
import os
from pathlib import Path

import numpy as np
import yaml
from pandas import read_csv


DATA_PATH = "/app/dataset/stack_v2_final_tokenized/tokenized_stack_v2_final"
OUT_FILE = Path("data_args_3b.txt")

# CSV_FILE = "final_weights.csv"
CSV_FILE = "final_weights_3b.csv"

# `stack_full` path for 3B is different
STACK_3B_PATH = "/app/dataset/stack_v2_final_tokenized/tokenized_stack_v2_3b"

# 7B exceptions
# EXCEPTIONS = {"ir_python": 1, "ir_cpp": 1, "ir_rust": 1, "ir_low_resource": 3}
# 3B exceptions
EXCEPTIONS = {"wikipedia": 0, "lhq_data": 1, "arxiv": 0,
              "ir_python": 1, "ir_cpp": 1, "ir_rust": 1, "ir_low_resource": 3}

df = read_csv(CSV_FILE)


def float_representer(dumper, value):
    text = "{0:.2f}".format(value)
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


def get_bin_size(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith(".bin"):
            return os.path.getsize(os.path.join(dir_path, file))
    return 0


def get_num_b_tokens(target_path):
    return float(np.round(get_bin_size(target_path) / (2 * 10**9), 2))


yaml.add_representer(float, float_representer)

# Generate data prefix list
data_prefix = []

for i in range(len(df)):
    source = df.iloc[i]["data_source"]
    data_path = STACK_3B_PATH if source == "stack_3b" else DATA_PATH
    n_shards = int(df.iloc[i]["n_shards"])
    size_0 = get_num_b_tokens(f"{data_path}/{source}/{source}_0")
    if source in EXCEPTIONS:
        if EXCEPTIONS[source] == 0:
            continue
        # these sources have custom weights
        total_weight = sum([
            get_num_b_tokens(f"{data_path}/{source}/{source}_{shard}") for shard in range(n_shards)
        ])
        for shard in range(n_shards):
            target_path = f"{data_path}/{source}/{source}_{shard}"
            # shard size in GB divided by 2 Bytes per token
            sampling_ratio = EXCEPTIONS[source] / total_weight
            shard_weight = float(np.round(get_num_b_tokens(target_path) * sampling_ratio, 2))
            shard_path = f"{target_path}/gpt2-preprocessed_content_document"
            data_prefix.extend([shard_weight, shard_path])
        
        size_0 = float(np.round(size_0 * sampling_ratio, 2))
    else:
        for shard in range(n_shards):
            target_path = f"{data_path}/{source}/{source}_{shard}"
            # shard size in GB divided by 2 Bytes per token
            shard_weight = get_num_b_tokens(target_path)
            shard_path = f"{target_path}/gpt2-preprocessed_content_document"
            data_prefix.extend([shard_weight, shard_path])
    print(f"source: {source}, shard_weight_0: {size_0}, n_shards: {n_shards}")

# Write the data arguments to a file
with open(OUT_FILE, "w") as f:
    f.write(" ".join([str(arg) for arg in data_prefix]))

print(f"Wrote the args to {OUT_FILE} 🎉")

"""
OUTPUT - 7B:
source: pull_requests, shard_weight_0: 1.96, n_shards: 10
source: issues, shard_weight_0: 2.21, n_shards: 5
source: jupyter_structured, shard_weight_0: 2.59, n_shards: 6
source: jupyter_scripts, shard_weight_0: 2.72, n_shards: 6
source: kaggle_scripts, shard_weight_0: 1.68, n_shards: 1
source: documentation, shard_weight_0: 1.6, n_shards: 1
source: owm, shard_weight_0: 2.42, n_shards: 6
source: wikipedia, shard_weight_0: 2.33, n_shards: 3
source: stackoverflow, shard_weight_0: 3.32, n_shards: 3
source: lhq_data, shard_weight_0: 1.45, n_shards: 3
source: arxiv, shard_weight_0: 5.27, n_shards: 6
source: ir_cpp, shard_weight_0: 1, n_shards: 1
source: ir_rust, shard_weight_0: 1, n_shards: 1
source: ir_python, shard_weight_0: 1, n_shards: 1
source: ir_low_resource, shard_weight_0: 3, n_shards: 1
source: stack_full, shard_weight_0: 1.33, n_shards: 512

3B:
source: pull_requests, shard_weight_0: 1.96, n_shards: 10
source: issues, shard_weight_0: 2.21, n_shards: 5
source: jupyter_structured, shard_weight_0: 2.59, n_shards: 6
source: jupyter_scripts, shard_weight_0: 2.72, n_shards: 6
source: kaggle_scripts, shard_weight_0: 1.68, n_shards: 1
source: documentation, shard_weight_0: 1.6, n_shards: 1
source: owm, shard_weight_0: 2.42, n_shards: 6
source: stackoverflow, shard_weight_0: 3.32, n_shards: 3
source: lhq_data, shard_weight_0: 0.25, n_shards: 3
source: ir_cpp, shard_weight_0: 1.0, n_shards: 1
source: ir_rust, shard_weight_0: 1.0, n_shards: 1
source: ir_python, shard_weight_0: 1.0, n_shards: 1
source: ir_low_resource, shard_weight_0: 3.0, n_shards: 1
source: stack_3b, shard_weight_0: 1.95, n_shards: 256
"""

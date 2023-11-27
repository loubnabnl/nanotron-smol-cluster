import csv
import os

import numpy as np
import yaml
from pandas import read_csv


DATA_PATH = "/fsx/anton/data/stack_v2_tokenized"
CSV_FILE = "/fsx/loubna/projects/brrr/examples/starcoder2/configs/final_weights.csv"
YAML_FILE = "/fsx/loubna/projects/brrr/examples/starcoder2/configs/config_final_starcoder2_7b.yaml"
NEW_YAML_FILE = "/fsx/loubna/projects/brrr/examples/starcoder2/configs/config_starcoder2_7b_new.yaml"
EXCEPTIONS = {"ir_python": 1, "ir_cpp": 1, "ir_rust": 1, "ir_low_resource": 3}

df = read_csv(CSV_FILE)


def float_representer(dumper, value):
    text = "{0:.2f}".format(value)
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


def get_bin_size(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith(".bin"):
            return os.path.getsize(os.path.join(dir_path, file))
    return 0

yaml.add_representer(float, float_representer)

# Generate data prefix list
data_prefix = []
for i in range(len(df)):
    source = df.iloc[i]["data_source"]
    n_shards = int(df.iloc[i]["n_shards"])
    size_0 = np.round(get_bin_size(f"{DATA_PATH}/{source}/{source}_0") / (2 * 10**9),2)
    if source in EXCEPTIONS:
        # these sources have custom weights
        data_prefix.extend(
            [
                float(EXCEPTIONS[source]),
                f"{DATA_PATH}/{source}/{source}_0/gpt2-preprocessed_content_document",
            ]
        )
        size_0 = EXCEPTIONS[source]
    print(f"source: {source}, shard_weight_0: {size_0}, n_shards: {n_shards}")
    for shard in range(n_shards):
        target_path = f"{DATA_PATH}/{source}/{source}_{shard}"
        # shard size in GB divided by 2 Bytes per token
        shard_weight = float(np.round(get_bin_size(target_path) / (2 * 10**9), 2))
        shard_path = f"{target_path}/gpt2-preprocessed_content_document"
        data_prefix.extend([shard_weight, shard_path])

# Read the YAML file
with open(YAML_FILE, "r") as file:
    config = yaml.safe_load(file)

# Update the data section
config["data"]["dataset"]["data_prefix"] = data_prefix


# Write the new YAML file
with open(NEW_YAML_FILE, "w") as file:
    yaml.safe_dump(config, file)

print(f"Updated YAML file saved as {NEW_YAML_FILE} 🎉")

"""
OUTPUT:
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
"""

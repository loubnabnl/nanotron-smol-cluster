import os
import shutil
import math


# go inside DATA_PATH and run this script
DATA_PATH = "/fsx/bigcode/bigcode-training/stack_v2/"

# if there are other data sources add them here, the values are number of shards, we don't shard small datasets
# data weight of each shard during: training size or custom_weight divided by number of shards
# DONE:
# "open-web-math": 6,
# "arxiv": 6,
# "stackoverflow": 3,
# "lhq_data": 3,
# "documentation": 1,
# "wikipedia": 3,

folders_dict = {
    # "issues": 3,
    # "jupyter_notebooks": 3,
    # "jupyter_scripts": 3,
    # "kaggle": 1,
    # "pull_requests": ?,
    # "intermediate_rep": ?,
    # "code": ?, for 12 langs we did 40 shards
}

# rename the code dataset into code and each source into the names in folder dict
def split_folder_into_subfolders(folder_base_name, num_of_subfolders, has_data_folder=True):
    """Split folder into subfolders containing equal number of files
    e.g stackoverflow/data with 3 shards will be stackoverflow_0, stackoverflow_1, stackoverflow_2
    under DATA_PATH directly"""
    extra_folder = "/data" if has_data_folder else ""
    folder_name = DATA_PATH + folder_base_name + extra_folder
    files = os.listdir(folder_name)
    total_files = len(files)
    print(f"folder contains {total_files} files total")
    files_per_folder = math.ceil(total_files / num_of_subfolders)

    for i in range(num_of_subfolders):
        new_folder = f'{DATA_PATH + folder_base_name}_{i}'
        os.makedirs(new_folder, exist_ok=True)
        print(f"created folder {new_folder}")
        for j in range(files_per_folder):
            if files:
                current_file = files.pop(0)
                # shutil.move(os.path.join(folder_name, current_file), os.path.join(new_folder, current_file))

    if not os.listdir(folder_name):
        print(f"The move is complete, removing empty folder {folder_name}")
        os.rmdir(folder_name)
        os.rmdir(DATA_PATH + folder_base_name)
        
        

def process_folders(folders_dict):
    for folder, num_of_subfolders in folders_dict.items():
        if num_of_subfolders > 1:
            print(f"sharding {folder} into {num_of_subfolders} folders")
            split_folder_into_subfolders(folder, num_of_subfolders, has_data_folder=True)


process_folders(folders_dict)

#add ["code"=NUMBER_SHARDS] instead of the subset below (e.g ["code"]=80 if we have 80 shards and folder name in LOAD_PATH is code)

declare -A folders_dict=(
    #["jupyter_scripts"]=6
)

# this is where code dataset was saved
LOAD_PATH="/fsx/bigcode/data_v2"
# you have read and write access here
OUTPUT_PATH="/fsx/bigcode/bigcode-training/tokenized_stack_v2_final"

for prefix in "${!folders_dict[@]}"; do
    num_subsets=${folders_dict[$prefix]}
    echo "Tokenizing the $num_subsets shards in $prefix" 

    for i in $(seq 0 $(($num_subsets - 1))); do
        # subfolders from anton data are named code_0, code_1, etc.
        folder_name="code_$i"
        echo "Processing folder: $folder_name"

        # Find the JSON file inside the folder
        json_file=$(ls $LOAD_PATH/$prefix/$folder_name/*.json | head -n 1)
        if [ -z "$json_file" ]; then
            echo "No JSON file found in $folder_name"
            continue
        fi

        echo "Found JSON file: $json_file"
        
        # Constructing the output path
        FULL_OUT_PATH="$OUTPUT_PATH/$prefix/${prefix}_${i}"
        mkdir -p $FULL_OUT_PATH

        echo "Output path: $FULL_OUT_PATH created"
        # Tokenizing the JSON file
        echo "Tokenizing $json_file"
        sbatch -J $prefix-$i ./slurm_files/tokenize_stackv2_anton.slurm $json_file $FULL_OUT_PATH
    done
done

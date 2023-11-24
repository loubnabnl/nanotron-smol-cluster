
# add more sources and their number of shards
declare -A  folders_dict=(
    ["owm"]=6
    ["arxiv"]=6
    ["stackoverflow"]=3
    ["lhq_data"]=3
    ["wikipedia"]=3
    ["documentation"]=1
)

for prefix in "${!folders_dict[@]}"; do
    num_subsets=${folders_dict[$prefix]}
    for i in $(seq 0 $(($num_subsets - 1))); do
        echo "Json creation of shard $i in $prefix"
        # rm -rf /fsx/loubna/data/stack_v2_smol_all_jsonl/$prefix\_$i
        sbatch -J $i-$prefix-shard /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/slurm_files/create_json.slurm $prefix\_$i 
    done
done
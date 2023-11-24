# add more sources and their number of shards
#     ["lhq_data"]=3
declare -A  folders_dict=(
    ["owm"]=6
    ["arxiv"]=6
    ["stackoverflow"]=3
    ["wikipedia"]=3
    ["documentation"]=1
)


for prefix in "${!folders_dict[@]}"; do
    num_subsets=${folders_dict[$prefix]}
    echo "Tokenizing the $num_subsets shards in $prefix" 
    # separate sources with num_subsets = 1
    if [ $num_subsets -eq 1 ]; then
        echo "Tokenizing $prefix"
        sbatch -J $prefix-tokenize /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/slurm_files/tokenize_stackv2.slurm $prefix
    else
        for i in $(seq 0 $(($num_subsets - 1))); do
            echo "Shard: $prefix\_$i"
            rm -rf /fsx/bigcode/bigcode-training/tokenized_stackv2_smol/$prefix\_$i
            sbatch -J $i-$prefix-t /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/slurm_files/tokenize_stackv2.slurm $prefix\_$i
        done
    fi

done


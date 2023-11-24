
# check first 40 tokenized shards for code if their have reasonable size
for i in $(seq 0 39); do
    data=/fsx/bigcode/bigcode-training/tokenized_stack_v2_final/
    # make sure folder not empty and print ok and show size
    if [ "$(ls -A $data\code_$i)" ]; then
        echo "code_$i ok!"
        du -sh $data/code_$i
    else
        echo "Empty"
    fi

done

# find those that failed and restart you might need to shard them further or rebuild the jsonl file it something's off
missing=(8 9 23 32 37)
for i in "${missing[@]}"; do
    echo "Shard: code\_$i"
    rm -rf /fsx/bigcode/bigcode-training/tokenized_stack_v2_final/$prefix\_$i
    sbatch -J $i-code-tokenize /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/slurm_files/tokenize_stackv2.slurm code\_$i
done

sbatch -J 9-1-tokenize /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/tokenize_stackv2.slurm code_9_1

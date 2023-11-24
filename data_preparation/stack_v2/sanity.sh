
for i in $(seq 0 39); do
    data=/fsx/loubna/data/stack_v2_smol_all_jsonl/
    # make sure folder not empty and print ok and show size
    if [ "$(ls -A $data\code_$i)" ]; then
        echo "code_$i ok!"
        du -sh $data/code_$i
    else
        echo "Empty"
    fi

done


missing=(8 9 23 32 37)
for i in "${missing[@]}"; do
    echo "Shard: code\_$i"
    rm -rf /fsx/bigcode/bigcode-training/tokenized_stackv2_smol/$prefix\_$i
    sbatch -J $i-code-tokenize /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/tokenize_stackv2.slurm code\_$i
done

sbatch -J 9-1-tokenize /fsx/loubna/bigcode_2/code/megatron-smol-cluster/data_preparation/stack_v2/tokenize_stackv2.slurm code_9_1
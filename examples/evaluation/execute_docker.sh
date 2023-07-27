#srun --nodes=1 --cpus-per-task=96 --partition=dev-cluster --job-name=eval --time 5:00:00 --pty bash

cd /fsx/loubna/code/bigcode-evaluation-harness
#sudo make DOCKERFILE=Dockerfile-multiple all
#sudo docker pull ghcr.io/bigcode-project/evaluation-harness-multiple
#sudo docker tag ghcr.io/bigcode-project/evaluation-harness-multiple evaluation-harness-multiple

langs=(py js java cpp swift php d jl lua r rkt rb rs)

model=starcoder
org=bigcode
eval_path=/fsx/loubna/code/bigcode-evaluation-harness/multiple_gens

for lang in "${langs[@]}"; do
    suffix=$model-$lang.json
    echo "Evaluation of $model on $lang benchmark, data in $suffix"
    gens_path=$eval_path/$suffix
    sudo docker run -v $gens_path:/app/$suffix:ro -it evaluation-harness-multiple python3 main.py \
        --model $org/$model \
        --tasks multiple-$lang \
        --load_generations_path /app/$suffix \
        --allow_code_execution  \
        --use_auth_token \
        --temperature 0.2 \
        --n_samples 50 | tee -a $eval_path/logs_codegen.txt
    echo "$lang Done"
done

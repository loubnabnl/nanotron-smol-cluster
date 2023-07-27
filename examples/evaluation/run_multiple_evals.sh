langs=(py js java cpp swift php d jl lua r rkt rb rs)
model=starcoder
org=bigcode

for lang in "${langs[@]}"; do
    sbatch -J "eval-$model-$lang" /fsx/loubna/code/bigcode-evaluation-harness/multiple_evals.slurm "$model" "$lang" "$org"
done

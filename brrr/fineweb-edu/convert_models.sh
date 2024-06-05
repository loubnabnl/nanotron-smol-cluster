#!/bin/bash
source ~/.bashrc
conda activate nanotron

start=10000
end=130000
step=10000
ckpt_base_path=/scratch/loubna/fineweb_edu_350B
save_base_path=/fsx/loubna/checkpoints/fineweb_edu_350B_converted

for ((iter=start; iter<=end; iter+=step)); do
  ckpt_path=$ckpt_base_path/$iter
  save_path=$save_base_path/$iter

  mkdir -p $ckpt_path
  mkdir -p $save_path

  s5cmd cp s3://synthetic-project-models/ablations/edu_fw_ablations-1p82G-edu_fineweb_350b_tokens-seed-1-/$iter/* $ckpt_path

  cd /fsx/loubna/projects/main/nanotron/examples/llama
  echo "Converting $ckpt_path to $save_path"
  torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=$ckpt_path --save_path=$save_path --tokenizer_name=gpt2
done

#!/bin/bash

#=== If using local gguf model with nli method
export LLAMA_CPP_PATH="llama.cpp/models/mixtral-instruct-8x7b/ggml-model-q4_0.gguf"
export LLAMA_CPP_PATH="/fsx/loubna/projects/afaik_files/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
python validate.py --method nli --model_path $LLAMA_CPP_PATH --local

#=== if using huggingface model with nli method
# python validate.py --method nli --model_path "mistralai/Mistral-7B-Instruct-v0.2"

#=== if using local gguf model with prompt method
# export LLAMA_CPP_PATH="llama.cpp/models/mixtral-instruct-8x7b/ggml-model-q4_0.gguf"
# python validate.py --method prompt --model_path $LLAMA_CPP_PATH --local

#=== if using local huggingface model with prompt method
# python validate.py --method prompt --model_path "mistralai/Mistral-7B-Instruct-v0.2"
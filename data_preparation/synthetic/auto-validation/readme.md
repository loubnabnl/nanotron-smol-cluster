Code from Gin Jiang
## Requirements
```
transformers
datasets
torch
tqdm
llama-cpp-python
spacy
```

## Quick Start
The code is to check the probability that the generated text contains hallucination effect. To run, use:

```
./run.sh
```

### Notes
* The code is based on the paper: [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)
* Two methods included in `run.sh`: the `nli` and `prompt` method, each support using local quantized gguf model or huggingface model.
* `nli` uses deberta model fine-tuned on NLI tasks to check if there is any contradiction between target passage and sampled passages (with the same prompt), while `prompt` uses generative models with prompt.
* `nli` is much faster, but `prompt` performs slightly better and more stable. (how "slightly" depend on tasks)
* The output score is averaged over all sentences, and the lower the score the better. You can also do downstream filtering or revision based on the score, it's recommended to set threshold to 0.5.
* The code is optimized for readability not performance - multiprocessing would be needed when run in large scale.


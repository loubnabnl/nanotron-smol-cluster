from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

def print_tokens_per_second(checkpoint, device, batch_size=1):
    print(f"Loading model {checkpoint} in fp16...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    inputs = tokenizer(["# torch training loop"], return_tensors="pt").to(device)

    new_tokens = 100
    cumulative_time = 0

    # warmup
    model.generate(
        **inputs, do_sample=False, max_new_tokens=new_tokens, min_new_tokens=new_tokens, num_return_sequences=batch_size
    )

    for _ in range(10):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device=device)
        start_event.record(stream=torch.cuda.Stream(device=device))
        model.generate(
            **inputs, do_sample=False, max_new_tokens=new_tokens, min_new_tokens=new_tokens, num_return_sequences=batch_size
        )
        end_event.record(stream=torch.cuda.Stream(device=device))
        torch.cuda.synchronize(device=device)
        latency_ms = start_event.elapsed_time(end_event)
        cumulative_time += latency_ms / 1e3
    print(f"Tokens per second for {checkpoint} fp16: {new_tokens * batch_size * 10 / cumulative_time:.1f}")

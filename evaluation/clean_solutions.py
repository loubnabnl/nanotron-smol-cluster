import json
import argparse
from datasets import load_dataset


STOP_TOKENS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<filename>", "<file_sep>", "<|endoftext|>"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="generation_nemo.json")
    parser.add_argument("--save_path", type=str, default="generations_3b.json")
    return parser.parse_args()


def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def postprocess_generation(generation, prompt):
    """Defines the postprocessing for a LM generation.
    :param generation: str
        code generation from LM
    :param idx: int
        (not used for Humaneval-Task)
    """
    if not generation.startswith(prompt[:20]):
        print(f"issue with generation: {generation}")
        print(f"origin prompt: {prompt}")
    generation = generation[len(prompt) :]
    return prompt + stop_at_stop_token(generation, STOP_TOKENS)


if __name__ == "__main__":
    args = get_args()
    # load HumanEval to get the prompts
    ds = load_dataset("openai_humaneval", split="test")

    # load generations data
    data = []
    print(f"Loading data from {args.load_path}")
    with open(f"{args.load_path}", "r") as file:
        i = 0
        for line in file:
            i += 1
            data.append(json.loads(line))

    # post-process the generations
    final_generations = []
    for i, problem in enumerate(data):
        # prepare each fo teh 164 problems
        solutions = []
        prompt = ds[i]["prompt"].strip() 
        for sol in problem["completion"]:
            # post-process all 20 solutions
            clean_sol = postprocess_generation(sol, prompt)
            solutions.append(clean_sol)
        final_generations.append(solutions)

    # save new generations
    with open(f"{args.save_path}", "w") as fp:
        json.dump(final_generations, fp)

    print(f"Data saved at {args.save_path} 🎉")
    # sanity checks
    print(f"example 0 solution 5:\n{final_generations[0][5]}")
    print("-" * 60)
    print(f"example 56 solution 13:\n{final_generations[56][13]}")
    print("-" * 60)
    print(f"example 102 solution 0:\n{final_generations[102][0]}")
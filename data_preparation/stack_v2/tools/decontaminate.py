
"""data to filter out of the dataset"""
import json
import itertools
import sys
import os

from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


# HumanEval solutions that are considered simple/generic enough to be kept in the training dataset
HUMAN_EVAL_STRINGS_OK = ['return x + y', 'return len(string)', 'return n**2', 'return ''.join(strings)']


def extract_docstring(prompt: str) -> str:
    if '"""' in prompt:
        if prompt.count('"""') == 2:
            return prompt.split('"""')[1].strip()
        elif prompt.count('"""') == 4:
            return prompt.split('"""')[3].strip()
        else:
            raise ValueError()
    elif '\'\'\'' in prompt:
        assert prompt.count('\'\'\'') == 2
        return prompt.split('\'\'\'')[1].strip()
    else:
        raise ValueError()


def human_eval_docstrings():
    ds = load_dataset("openai_humaneval", split="test")
    docstrings = [extract_docstring(v['prompt']) for v in ds]
    return docstrings


def apps_solutions():
    """
    Solutions column contains a list of strings
    """
    ds = load_dataset("codeparrot/apps", split="test")
    solutions = [sample["solutions"] for sample in ds if len(sample["solutions"]) > 0]
    res = itertools.chain.from_iterable(json.loads(sample) for sample in solutions)
    return list(res)


def load_dataset_column(dataset: str, column: str, split: str, name=None):
    ds = load_dataset(dataset, split=split, name=name)
    res = [sample[column].strip() for sample in ds]
    # Only return non-empty strings
    return [sample for sample in res if len(sample) > 0]


FILTER_OUT = {
    "human_eval_docstrings": human_eval_docstrings(),
    "human_eval_solutions": [
        s for s in load_dataset_column("openai_humaneval", "canonical_solution", "test")
        if s not in HUMAN_EVAL_STRINGS_OK
    ],
}


for benchmark, values in FILTER_OUT.items():
    print(f"num strings from {benchmark}: {len(values)}")


def decontaminate(samples, filter_out=FILTER_OUT):
    """
    filter_out: Dict[str, List[str]] mapping from benchmark name to list of strings that need to be
    filtered-out.
    Return a list where each element is True if the corresponding file should be included in the dataset.
    Otherwise, the element is False.
    """
    output = []

    for content in samples["content"]:
        content = content.lower()
        matched = False
        for benchmark, substrings in filter_out.items():
            for substring in substrings:
                if substring.lower() in content:
                    matched = True
                    break
            if matched:
                break
        # we keep files that are not matched
        output.append(not matched)
        
    return output

dataset = load_dataset("your_data", split="train")
old_size = len(dataset)
dataset = dataset.filter(decontaminate, batched=True, batch_size=10_000, num_proc=64)

print(
    f"Removed {old_size - len(dataset)} files from {old_size} (i.e {(old_size - len(dataset)) * 100 / old_size}%)"
)

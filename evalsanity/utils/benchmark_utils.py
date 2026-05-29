from datasets import load_dataset

BENCHMARK_REGISTRY = {
    "gsm8k": {
        "path": "gsm8k",
        "config": "main",
        "question_key": "question",
        "answer_key": "answer",
        "format": "qa",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "config": "all",
        "question_key": "question",
        "answer_key": "answer",
        "choices_key": "choices",
        "format": "multiple_choice",
    },
    "truthful_qa": {
        "path": "truthful_qa",
        "config": "multiple_choice",
        "question_key": "question",
        "answer_key": "correct_answers",
        "choices_key": "mc1_targets",
        "format": "multiple_choice",
    },
    "hellaswag": {
        "path": "hellaswag",
        "config": None,
        "question_key": "ctx",
        "answer_key": "label",
        "choices_key": "endings",
        "format": "multiple_choice",
    },
    "humaneval": {
        "path": "openai_humaneval",
        "config": None,
        "question_key": "prompt",
        "answer_key": "canonical_solution",
        "format": "code",
    },
}


import re


def load_benchmark(name, split="test", max_samples=None, subject=None):
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    info = dict(BENCHMARK_REGISTRY[name])
    path = info["path"]
    config = subject if subject else info["config"]
    try:
        ds = load_dataset(path, config, split=split, streaming=True)
    except Exception:
        ds = load_dataset(path, config, split=split)
    samples = []
    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        if name == "gsm8k":
            answer_text = example.get("answer", "")
            m = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
            example["parsed_answer"] = m.group(1).replace(",", "") if m else answer_text
        elif name == "mmlu":
            choices = example.get("choices", [])
            correct_idx = example.get("answer", 0)
            if isinstance(correct_idx, int) and correct_idx < len(choices):
                example["parsed_answer"] = choices[correct_idx]
                example["correct_letter"] = chr(65 + correct_idx)
            else:
                example["parsed_answer"] = str(correct_idx)
                example["correct_letter"] = "A"
        samples.append(example)
    return samples, info

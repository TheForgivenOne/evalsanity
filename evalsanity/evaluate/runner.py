import torch
from ..utils.model_utils import load_model
from ..utils.benchmark_utils import load_benchmark
from ..utils.logging import Logger


class Evaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto", quantize="4bit"):
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.log = Logger()

    def _ensure_model(self):
        if self.model is None:
            self.log.info(f"Loading {self.model_name} for evaluation...")
            self.model, self.tokenizer = load_model(
                self.model_name, device=self.device, quantize=self.quantize
            )

    def _format_prompt(self, question, fmt="qa"):
        if fmt == "qa":
            return f"Question: {question}\nAnswer:"
        elif fmt == "multiple_choice":
            return f"Question: {question}\nAnswer the correct option (A, B, C, or D):"
        elif fmt == "code":
            return f"Complete the following code:\n{question}\n"
        return f"Q: {question}\nA:"

    def evaluate(self, benchmark, split="test", max_samples=None, questions=None):
        self._ensure_model()
        if questions is None:
            samples, info = load_benchmark(benchmark, split=split, max_samples=max_samples)
        else:
            info = {"format": "qa"}
        q_key = info.get("question_key", "question")
        a_key = info.get("answer_key", "answer")
        fmt = info.get("format", "qa")

        results = []
        correct = 0
        total = 0
        for sample in samples if questions is None else questions:
            question = sample[q_key] if isinstance(sample, dict) else sample["question"]
            expected = str(sample.get(a_key, "")) if isinstance(sample, dict) else str(sample.get("answer", ""))
            prompt = self._format_prompt(question, fmt)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs.input_ids.to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "question": question[:100],
                "expected": expected[:100],
                "generated": generated.strip()[:100],
            })
        return {"results": results, "total": len(results)}

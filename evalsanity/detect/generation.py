import torch
from ..utils.model_utils import load_model
from ..utils.benchmark_utils import load_benchmark
from ..utils.logging import Logger


class GenerationDetector:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto", quantize="4bit"):
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.log = Logger()

    def _ensure_model(self):
        if self.model is None:
            self.log.info(f"Loading {self.model_name} for generation detection...")
            self.model, self.tokenizer = load_model(
                self.model_name, device=self.device, quantize=self.quantize
            )

    def _completion_accuracy(self, prefix, expected_suffix):
        self._ensure_model()
        inputs = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs.input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        expected_tokens = expected_suffix.strip().lower().split()[:5]
        generated_tokens = generated.strip().lower().split()[:5]
        if not expected_tokens:
            return 0.0
        matches = sum(1 for e, g in zip(expected_tokens, generated_tokens) if e == g)
        return matches / len(expected_tokens)

    def detect(self, benchmark, split="test", max_samples=None):
        samples, info = load_benchmark(benchmark, split=split, max_samples=max_samples)
        q_key = info["question_key"]
        a_key = info.get("answer_key", "")
        results = []
        accuracies = []
        for i, sample in enumerate(samples):
            question = sample[q_key]
            answer = str(sample.get(a_key, "")) if a_key else ""
            prefix_tokens = question.split()[:30]
            prefix = " ".join(prefix_tokens)
            expected = " ".join(question.split()[30:35]) if len(question.split()) > 30 else answer[:50]
            acc = self._completion_accuracy(prefix, expected)
            accuracies.append(acc)
            results.append({
                "index": i,
                "question": question[:100],
                "completion_accuracy": round(acc, 3),
                "flagged": acc > 0.8,
            })
        return {
            "method": "generation",
            "benchmark": benchmark,
            "model": self.model_name,
            "total": len(results),
            "avg_completion_accuracy": round(sum(accuracies) / max(len(accuracies), 1), 3) if accuracies else None,
            "flagged_count": sum(1 for r in results if r["flagged"]),
            "flagged_pct": sum(1 for r in results if r["flagged"]) / max(len(results), 1) * 100,
            "results": results,
        }

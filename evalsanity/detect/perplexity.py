import torch
from ..utils.model_utils import load_model
from ..utils.benchmark_utils import load_benchmark
from ..utils.logging import Logger


class PerplexityDetector:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto", quantize="4bit"):
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.log = Logger()

    def _ensure_model(self):
        if self.model is None:
            self.log.info(f"Loading {self.model_name} for perplexity detection...")
            self.model, self.tokenizer = load_model(
                self.model_name, device=self.device, quantize=self.quantize
            )

    def _perplexity(self, text):
        self._ensure_model()
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def detect(self, benchmark, split="test", max_samples=None):
        samples, info = load_benchmark(benchmark, split=split, max_samples=max_samples)
        q_key = info["question_key"]
        results = []
        perplexities = []
        for i, sample in enumerate(samples):
            question = sample[q_key]
            ppl = self._perplexity(question)
            perplexities.append(ppl)
            results.append({
                "index": i,
                "question": question[:100],
                "perplexity": round(ppl, 2),
                "flagged": False,
            })
        if perplexities:
            avg = sum(perplexities) / len(perplexities)
            std = (sum((p - avg) ** 2 for p in perplexities) / len(perplexities)) ** 0.5
            threshold = avg - 1.5 * std
            for r, p in zip(results, perplexities):
                r["flagged"] = p < threshold
        return {
            "method": "perplexity",
            "benchmark": benchmark,
            "model": self.model_name,
            "total": len(results),
            "avg_perplexity": round(sum(perplexities) / max(len(perplexities), 1), 2) if perplexities else None,
            "flagged_count": sum(1 for r in results if r["flagged"]),
            "flagged_pct": sum(1 for r in results if r["flagged"]) / max(len(results), 1) * 100,
            "results": results,
        }

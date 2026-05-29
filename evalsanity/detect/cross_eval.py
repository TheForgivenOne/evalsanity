import torch
import re
from ..utils.model_utils import load_model
from ..utils.benchmark_utils import load_benchmark
from ..generate.variants import VariantGenerator
from ..utils.logging import Logger


class CrossEvalDetector:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="auto", quantize="4bit",
                 generator_model="Qwen/Qwen2.5-3B-Instruct", generator_device="cuda:1"):
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self.generator_model = generator_model
        self.generator_device = generator_device
        self.model = None
        self.tokenizer = None
        self.variant_gen = None
        self.log = Logger()

    def _ensure_model(self):
        if self.model is None:
            self.log.info(f"Loading evaluator: {self.model_name}")
            self.model, self.tokenizer = load_model(
                self.model_name, device=self.device, quantize=self.quantize
            )

    def _ensure_generator(self):
        if self.variant_gen is None:
            self.log.info(f"Loading generator: {self.generator_model} on {self.generator_device}")
            self.variant_gen = VariantGenerator(
                model_name=self.generator_model,
                device=self.generator_device,
                quantize=self.quantize,
            )

    def _get_answer(self, question, fmt="qa"):
        self._ensure_model()
        if fmt == "qa":
            prompt = f"Question: {question}\nAnswer:"
        elif fmt == "multiple_choice":
            prompt = f"Question: {question}\nAnswer the correct option (A, B, C, or D):"
        else:
            prompt = f"Q: {question}\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
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
        return self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    def _score_answer(self, generated, expected):
        g = generated.strip().lower()
        e = expected.strip().lower()
        nums_gen = set(re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', g.replace(',', '')))
        nums_exp = set(re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', e.replace(',', '')))
        if nums_exp:
            match_count = len(nums_gen & nums_exp)
            return match_count / max(len(nums_exp), 1)
        g_tokens = g.split()[:10]
        e_tokens = e.split()[:10]
        if not e_tokens:
            return 0.0
        matches = sum(1 for gt in g_tokens if gt in e_tokens)
        return matches / len(e_tokens)

    def detect(self, benchmark, split="test", max_samples=None, subject=None):
        samples, info = load_benchmark(benchmark, split=split, max_samples=max_samples, subject=subject)
        q_key = info["question_key"]
        a_key = info.get("answer_key", "answer")
        fmt = info.get("format", "qa")
        parsed_key = "parsed_answer" if (benchmark in ("gsm8k", "mmlu")) else a_key

        self._ensure_generator()
        self.log.info(f"Generating variants for {len(samples)} samples...")

        results = []
        original_scores = []
        variant_scores = []
        performance_drops = []

        for i, sample in enumerate(samples):
            question = sample[q_key]
            expected = str(sample.get(parsed_key, sample.get(a_key, "")))

            variant = self.variant_gen.generate_variant(question, expected)

            orig_answer = self._get_answer(question, fmt)
            var_answer = self._get_answer(variant, fmt)

            orig_score = self._score_answer(orig_answer, expected)
            var_score = self._score_answer(var_answer, expected)

            original_scores.append(orig_score)
            variant_scores.append(var_score)
            drop = orig_score - var_score
            performance_drops.append(drop)

            results.append({
                "index": i,
                "question": question[:80],
                "variant": variant[:80],
                "orig_score": round(orig_score, 3),
                "var_score": round(var_score, 3),
                "drop": round(drop, 3),
                "flagged": drop > 0.3,
            })

            if (i + 1) % 5 == 0:
                self.log.info(f"  Processed {i+1}/{len(samples)}")

        avg_orig = sum(original_scores) / max(len(original_scores), 1)
        avg_var = sum(variant_scores) / max(len(variant_scores), 1)
        avg_drop = sum(performance_drops) / max(len(performance_drops), 1)
        flagged = sum(1 for r in results if r["flagged"])

        return {
            "method": "cross_eval",
            "benchmark": benchmark,
            "model": self.model_name,
            "total": len(results),
            "avg_original_score": round(avg_orig, 3),
            "avg_variant_score": round(avg_var, 3),
            "avg_performance_drop": round(avg_drop, 3),
            "flagged_count": flagged,
            "flagged_pct": flagged / max(len(results), 1) * 100,
            "results": results,
            "contamination_index": round(avg_drop * 100, 1),
        }

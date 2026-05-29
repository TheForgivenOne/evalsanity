import torch
import re
import numpy as np
from ..utils.model_utils import load_model
from ..utils.benchmark_utils import load_benchmark
from ..utils.logging import Logger


REPHRASE_PROMPTS = [
    (
        "Rephrase this question. Change all numbers, names, and entities "
        "while keeping the exact same structure and difficulty.\n"
        "Original: {question}\nRephrased:"
    ),
    (
        "Rewrite this question with different numbers and different wording. "
        "The math or logic must be identical.\n"
        "Original: {question}\nRewritten:"
    ),
]


class VariantGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device="auto", quantize="4bit",
                 _model=None, _tokenizer=None):
        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self.model = _model
        self.tokenizer = _tokenizer
        self._sim_model = None
        self.log = Logger()

    def _ensure_model(self):
        if self.model is None:
            self.log.info(f"Loading generator: {self.model_name}")
            self.model, self.tokenizer = load_model(
                self.model_name, device=self.device, quantize=self.quantize
            )

    def _ensure_sim_model(self):
        if self._sim_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self._sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=dev)

    def _semantic_similarity(self, text1, text2):
        self._ensure_sim_model()
        emb = self._sim_model.encode([text1, text2], show_progress_bar=False)
        return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))

    def _generate_single(self, prompt_template, question):
        self._ensure_model()
        prompt = prompt_template.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated.strip().split("\n")[0].strip()

    def generate_variant(self, question, answer=""):
        candidates = []
        for prompt in REPHRASE_PROMPTS:
            try:
                variant = self._generate_single(prompt, question)
                if variant and variant.lower() != question.lower():
                    sim = self._semantic_similarity(question, variant)
                    candidates.append((variant, sim))
            except Exception:
                continue

        if not candidates:
            return question

        candidates.sort(key=lambda x: x[1], reverse=True)

        for variant, sim in candidates:
            if 0.60 <= sim <= 0.95:
                return variant

        best = candidates[0]
        return best[0] if best[1] > 0.3 else question

    def generate_variants(self, benchmark, split="test", max_samples=None, indices=None):
        samples, info = load_benchmark(benchmark, split=split, max_samples=max_samples)
        q_key = info["question_key"]
        a_key = info.get("answer_key", "")
        if indices is None:
            indices = list(range(len(samples)))
        variants = []
        for i in indices:
            if i >= len(samples):
                continue
            sample = samples[i]
            question = sample[q_key]
            answer = str(sample.get(a_key, "")) if a_key else ""
            variant = self.generate_variant(question, answer)
            sim = self._semantic_similarity(question, variant)
            variants.append({
                "index": i,
                "original": question,
                "variant": variant,
                "similarity": round(sim, 3),
                "answer": answer,
            })
        return variants

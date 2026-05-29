from .detect.ngram import NGramDetector
from .detect.perplexity import PerplexityDetector
from .detect.generation import GenerationDetector
from .detect.cross_eval import CrossEvalDetector
from .generate.variants import VariantGenerator
from .evaluate.runner import Evaluator
from .evaluate.reporter import Reporter


class EvalSanity:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto"):
        self.model_name = model_name
        self.device = device
        self.detectors = {
            "ngram": NGramDetector(),
            "perplexity": PerplexityDetector(model_name=model_name, device=device),
            "generation": GenerationDetector(model_name=model_name, device=device),
            "cross_eval": CrossEvalDetector(model_name=model_name, device=device),
        }
        self.generator = VariantGenerator(model_name=model_name, device=device)
        self.evaluator = Evaluator(model_name=model_name, device=device)
        self.reporter = Reporter()

    def analyze(self, benchmark, split="test", max_samples=None, methods=None):
        if methods is None:
            methods = ["ngram", "perplexity"]
        results = {}
        for method in methods:
            if method in self.detectors:
                results[method] = self.detectors[method].detect(
                    benchmark, split=split, max_samples=max_samples
                )
        return results

    def full_report(self, benchmark, split="test", max_samples=None, methods=None):
        if methods is None:
            methods = ["ngram", "perplexity"]
        contamination = self.analyze(
            benchmark, split=split, max_samples=max_samples, methods=methods
        )
        return self.reporter.generate(contamination, benchmark=benchmark)

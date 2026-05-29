from collections import Counter


class NGramDetector:
    def __init__(self, n=13):
        self.n = n

    def _get_ngrams(self, text, n):
        tokens = text.split()
        if len(tokens) < n:
            return set([" ".join(tokens)])
        return set(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def detect(self, benchmark, split="test", max_samples=None):
        from ..utils.benchmark_utils import load_benchmark
        samples, info = load_benchmark(benchmark, split=split, max_samples=max_samples)
        q_key = info["question_key"]
        questions = [s[q_key] for s in samples]

        ngram_to_questions = {}
        for i, q in enumerate(questions):
            ngrams = self._get_ngrams(q, self.n)
            for ng in ngrams:
                if ng not in ngram_to_questions:
                    ngram_to_questions[ng] = []
                ngram_to_questions[ng].append(i)

        overlapping = {ng: idxs for ng, idxs in ngram_to_questions.items() if len(idxs) > 1}

        intra_question_overlap_scores = []
        for i, q in enumerate(questions):
            ngrams = list(self._get_ngrams(q, self.n))
            matches = sum(1 for ng in ngrams if ng in overlapping and i in overlapping[ng])
            total = max(len(ngrams), 1)
            intra_question_overlap_scores.append(matches / total)

        results = []
        for i, q in enumerate(questions):
            score = intra_question_overlap_scores[i]
            results.append({
                "index": i,
                "question": q[:100],
                "score": round(score, 4),
                "flagged": score > 0.3,
            })

        flagged_count = sum(1 for r in results if r["flagged"])
        return {
            "method": "ngram",
            "benchmark": benchmark,
            "total": len(results),
            "flagged_count": flagged_count,
            "flagged_pct": flagged_count / max(len(results), 1) * 100,
            "avg_score": round(sum(r["score"] for r in results) / max(len(results), 1), 4),
            "shared_ngrams_found": len(overlapping),
            "results": results,
        }

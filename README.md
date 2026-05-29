# EvalSanity 🧪

**Contamination-Resistant ML Evaluation Framework**

[![GitHub](https://img.shields.io/badge/GitHub-TheForgivenOne%2Fevalsanity-181717?logo=github)](https://github.com/TheForgivenOne/evalsanity)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-orange?logo=pypi)](https://pypi.org/project/evalsanity/)

Benchmark contamination is the **#1 most underinvested high-impact problem in ML**. Every major benchmark (MMLU, GSM8K, HumanEval) is compromised by data leakage — models train on web data containing benchmark examples, inflating scores and masking real capability gaps. EvalSanity detects contamination and produces trustworthy evaluation reports.

---

## 🔬 Findings Preview

| Benchmark | Original Acc | Variant Acc | Contamination | Risk |
|-----------|:-----------:|:-----------:|:-----------:|:----:|
| **MMLU Elementary Math** | **87.5%** | **31.2%** | **56.2%** | 🔴 HIGH |
| GSM8K | 37.5% | 12.5% | 25.0% | 🟡 MEDIUM |

> Full findings: [`FINDINGS.md`](FINDINGS.md) · Leaderboard: [theforgivenone.github.io/evalsanity](https://theforgivenone.github.io/evalsanity/)

---

## 🚀 Quick Start

```bash
pip install evalsanity
```

### CLI

```bash
# Analyze GSM8K with n-gram + perplexity detection
evalsanity gsm8k --max-samples 50 --detect-methods ngram,perplexity

# Full cross-evaluation (most powerful method)
evalsanity gsm8k --detect-methods cross_eval --max-samples 20

# List available benchmarks
evalsanity --list-benchmarks
```

### Python API

```python
from evalsanity.detect.cross_eval import CrossEvalDetector

detector = CrossEvalDetector(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    quantize="4bit",
)

result = detector.detect("gsm8k", max_samples=20)
print(f"Original: {result['avg_original_score']:.1%}")
print(f"Variant:  {result['avg_variant_score']:.1%}")
print(f"Contamination: {result['contamination_index']:.1f}%")
```

---

## 🎯 Detection Methods

| Method | Description | Model Required | Speed |
|--------|-------------|:-------------:|:-----:|
| `ngram` | Finds long shared n-grams between questions | ❌ | ⚡ Instant |
| `perplexity` | Flags anomalously low perplexity (memorization signal) | ✅ | 🟢 Fast |
| `generation` | Tests if model completes known benchmark text verbatim | ✅ | 🟡 Medium |
| `cross_eval` | **Most powerful.** Compares accuracy on original vs. semantically-equivalent rewordings | ✅ | 🔴 Slow (best quality) |

### Cross-Evaluation (The Killer Feature)

1. Takes benchmark questions
2. Generates semantically-equivalent rewordings using a dedicated LLM
3. Evaluates the target model on both original and variant questions
4. If performance drops significantly → **contamination detected**

---

## 📦 Supported Benchmarks

| Benchmark | Dataset | Format | Status |
|-----------|---------|:------:|:------:|
| GSM8K | `gsm8k` | Math QA | ✅ |
| MMLU | `cais/mmlu` | Multiple Choice (57 subjects) | ✅ |
| TruthfulQA | `truthful_qa` | Multiple Choice | ✅ |
| HellaSwag | `hellaswag` | Multiple Choice | ✅ |
| HumanEval | `openai_humaneval` | Code Generation | ✅ |

---

## 🧱 Architecture

```
evalsanity/
├── detect/           # Detection methods
│   ├── ngram.py      # N-gram overlap analysis
│   ├── perplexity.py # Perplexity-based detection
│   ├── generation.py # Generation-based verification
│   └── cross_eval.py # Cross-evaluation (flagship)
├── generate/         # Variant generation
│   └── variants.py   # LLM-based question rephrasing with semantic validation
├── evaluate/         # Evaluation & reporting
│   ├── runner.py     # Model inference harness
│   └── reporter.py   # Report generation
├── cli/              # Command-line interface
└── utils/            # Model & benchmark utilities
```

---

## 📊 Methodology

**Cross-evaluation** works because a contaminated model has *memorized* the surface form of benchmark questions. When the same question is reworded — names changed, numbers swapped, phrasing restructured — the model's memorized answer no longer fits, revealing the true reasoning ability.

Key design decisions:
- **Dedicated generator model** (e.g., 3B params) breaks circular dependency between generating and evaluating
- **Semantic similarity validation** (all-MiniLM-L6-v2, threshold 0.60–0.95) rejects poor rephrasings
- **Number-aware scoring** for math benchmarks ensures accurate answer matching
- **Pre-generated variants** shared across models for scientific fairness

---

## 🤝 Contributing

Contributions welcome! This is an active research project. Areas to help:
- Add more benchmarks to the registry
- Improve variant generation quality
- Run evaluations on more models
- Write tests and CI

---

## 📄 License

MIT

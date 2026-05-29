# EvalSanity

**Contamination-Resistant ML Evaluation Framework**

Benchmark contamination is the #1 most underinvested high-impact problem in ML. Every major benchmark (MMLU, GSM8K, HumanEval, etc.) is compromised by data leakage — models train on web data containing benchmark examples, inflating scores and masking real capability gaps. EvalSanity detects contamination and produces trustworthy evaluation reports.

## Quick Start

```bash
# Analyze GSM8K with n-gram and perplexity methods
evalsanity gsm8k --max-samples 50

# Run full cross-evaluation (original vs. generated variants)
evalsanity gsm8k --detect-methods cross_eval --max-samples 20

# List available benchmarks
evalsanity --list-benchmarks
```

## Detection Methods

| Method | Description | Requires Model |
|--------|-------------|----------------|
| `ngram` | Checks for long shared n-grams between questions | No |
| `perplexity` | Flags examples with anomalously low perplexity (suggests memorization) | Yes |
| `generation` | Tests if model can complete known benchmark text | Yes |
| `cross_eval` | **Most powerful.** Compares accuracy on original vs. semantically equivalent variants | Yes |

## Cross-Evaluation (The Killer Feature)

The `cross_eval` method:
1. Takes benchmark questions
2. Generates semantically equivalent rewordings using a local LLM
3. Evaluates the target model on both original and variant questions
4. If performance drops significantly on variants → contamination detected

## Architecture

```
evalsanity/
  detect/          # Detection methods
    ngram.py       # N-gram overlap analysis
    perplexity.py  # Perplexity-based detection
    generation.py  # Generation-based verification
    cross_eval.py  # Cross-evaluation (most powerful)
  generate/        # Variant generation
    variants.py    # LLM-based question rephrasing
  evaluate/        # Evaluation and reporting
    runner.py      # Model inference
    reporter.py    # Report generation
  cli/             # Command-line interface
  utils/           # Model and benchmark utilities
```

## Supported Benchmarks

- GSM8K (grade school math)
- MMLU (massive multitask language understanding)
- TruthfulQA
- HellaSwag
- HumanEval (code generation)

## License

MIT

import argparse
import sys
from .. import EvalSanity
from ..utils.benchmark_utils import load_benchmark, BENCHMARK_REGISTRY
from ..evaluate.reporter import Reporter


def main():
    parser = argparse.ArgumentParser(description="EvalSanity - Contamination-Resistant Evaluation")
    parser.add_argument("benchmark", type=str, help="Benchmark name", nargs="?")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model to test")
    parser.add_argument("--detect-methods", type=str, default="ngram,perplexity",
                        help="Detection methods: ngram,perplexity,generation")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to analyze")
    parser.add_argument("--generate-variants", action="store_true", help="Generate clean variants")
    parser.add_argument("--output", type=str, default="evalsanity_report.json", help="Output path")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available benchmarks")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")

    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available benchmarks:")
        for name, info in BENCHMARK_REGISTRY.items():
            print(f"  {name} -> {info['path']} ({info['format']})")
        return

    if not args.benchmark:
        parser.print_help()
        return

    methods = [m.strip() for m in args.detect_methods.split(",")]

    print(f"EvalSanity: Analyzing benchmark '{args.benchmark}' with model '{args.model}'")
    print(f"Methods: {methods}, Max samples: {args.max_samples}")
    print()

    tool = EvalSanity(model_name=args.model, device=args.device)
    report = tool.full_report(
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        methods=methods,
    )

    reporter = Reporter()
    reporter.print_report(report)
    reporter.save(report, args.output)

    if args.generate_variants:
        print("\nGenerating clean variants of flagged examples...")
        variants = tool.generator.generate_variants(
            args.benchmark, max_samples=args.max_samples
        )
        print(f"Generated {len(variants)} variants")

    print("\nDone.")


if __name__ == "__main__":
    main()

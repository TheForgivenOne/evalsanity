import json
import datetime


class Reporter:
    def generate(self, detection_results, benchmark=""):
        report = {
            "report_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "benchmark": benchmark,
            "generated_at": datetime.datetime.now().isoformat(),
            "methods": {},
            "summary": {},
        }
        all_flagged_indices = set()
        method_scores = {}
        for method_name, result in detection_results.items():
            entry = {
                "total": result["total"],
                "flagged_count": result["flagged_count"],
                "flagged_pct": round(result["flagged_pct"], 2),
                "avg_score": round(result.get("avg_score", result.get("avg_perplexity", 0)), 3),
            }
            if "avg_original_score" in result:
                entry["avg_original_score"] = round(result["avg_original_score"], 3)
                entry["avg_variant_score"] = round(result["avg_variant_score"], 3)
                entry["avg_performance_drop"] = round(result["avg_performance_drop"], 3)
                entry["contamination_index"] = round(result["contamination_index"], 1)
            report["methods"][method_name] = entry
            method_scores[method_name] = result["flagged_pct"]
            for r in result.get("results", []):
                if r.get("flagged"):
                    all_flagged_indices.add(r["index"])

        report["summary"] = {
            "total_samples": max(
                (r["total"] for r in detection_results.values()), default=0
            ),
            "samples_flagged_by_any": len(all_flagged_indices),
            "contamination_risk": self._risk_level(method_scores),
            "verdict": self._verdict(method_scores),
        }
        return report

    def _risk_level(self, method_scores):
        scores = list(method_scores.values())
        if not scores:
            return "unknown"
        avg = sum(scores) / len(scores)
        if avg < 5:
            return "low"
        elif avg < 15:
            return "medium"
        elif avg < 30:
            return "high"
        return "critical"

    def _verdict(self, method_scores):
        scores = list(method_scores.values())
        if not scores:
            return "Insufficient data to determine contamination."
        avg = sum(scores) / len(scores)
        if avg < 5:
            return (
                "Low contamination risk. Benchmark results are likely reliable for this model."
            )
        elif avg < 15:
            return (
                "Moderate contamination risk. Some results may be inflated by memorization. "
                "Consider cross-validating on clean variants."
            )
        elif avg < 30:
            return (
                "High contamination risk. A significant portion of benchmark examples appear "
                "to be memorized. Results should not be taken at face value."
            )
        return (
            "Critical contamination risk. The model has likely seen this benchmark during training. "
            "Reported scores are not trustworthy."
        )

    def print_report(self, report):
        print("=" * 60)
        print(f"  EvalSanity Contamination Report")
        print(f"  Benchmark: {report['benchmark']}")
        print(f"  Generated: {report['generated_at']}")
        print("=" * 60)
        print()
        for method, details in report["methods"].items():
            print(f"  [{method.upper()}]")
            print(f"    Total samples:    {details['total']}")
            print(f"    Flagged:          {details['flagged_count']} ({details['flagged_pct']}%)")
            if 'avg_original_score' in details:
                print(f"    Original acc:     {details['avg_original_score']}")
                print(f"    Variant acc:      {details['avg_variant_score']}")
                print(f"    Performance drop: {details['avg_performance_drop']}")
                print(f"    Contamination:    {details['contamination_index']}%")
            print()
        print("-" * 60)
        s = report["summary"]
        print(f"  Risk Level:       {s['contamination_risk'].upper()}")
        print(f"  Unique Flagged:   {s['samples_flagged_by_any']} / {s['total_samples']}")
        print(f"  Verdict:          {s['verdict']}")
        print("=" * 60)
        return report

    def save(self, report, path="evalsanity_report.json"):
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {path}")
        return path

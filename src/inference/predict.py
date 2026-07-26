#!/usr/bin/env python
"""Single-command submission generator.

Usage:
    python -m inference.predict

Runs the complete, deterministic pipeline (Core Principle 5: a single command generates the
final submission) and prints a summary. Exits non-zero if any validation step fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.pipeline import run_inference  # noqa: E402


def main():
    report = run_inference()

    print(f"Submission written to: {report['output_path']}")
    print(f"SHA-256: {report['output_sha256']}")
    print(f"All validations passed: {report['all_passed']}")
    stats = report["prediction_validation"]["summary_stats"]
    print(f"Prediction summary: mean={stats['mean']:.3f} std={stats['std']:.3f} "
          f"min={stats['min']:.3f} max={stats['max']:.3f}")

    report_path = Path(report["output_path"]).parent / "last_inference_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full inference report: {report_path}")

    if not report["all_passed"]:
        print("VALIDATION FAILED -- see report for details.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

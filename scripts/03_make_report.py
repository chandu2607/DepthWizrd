#!/usr/bin/env python
"""Re-render EXPERIMENT_RESULTS.md from a saved results.json.

Optionally re-computes the GO/MODIFY/ABANDON recommendation with adjusted
thresholds WITHOUT re-running the experiment, e.g.:
    python scripts/03_make_report.py runs/phase1/results.json --min-gain 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.eval.decision import decide, DecisionThresholds
from depthwizard.eval import report as report_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--out", default=None, help="output .md path (default: alongside json)")
    ap.add_argument("--min-gain", type=float, default=None,
                    help="override min relative MAE gain of C over B")
    ap.add_argument("--min-pearson", type=float, default=None)
    args = ap.parse_args()

    results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))

    if args.min_gain is not None or args.min_pearson is not None:
        thr = DecisionThresholds()
        if args.min_gain is not None:
            thr.min_learned_gain = args.min_gain
        if args.min_pearson is not None:
            thr.min_pearson_signal = args.min_pearson
        ev = results.get("meta", {}).get("evidence_valid", False)
        results["decision"] = decide(results, thr=thr, evidence_valid=ev)
        print(f"[decision] recomputed -> {results['decision']['verdict']}")

    out = args.out or str(Path(args.results_json).with_name("EXPERIMENT_RESULTS.md"))
    report_mod.write_report(results, out)


if __name__ == "__main__":
    main()

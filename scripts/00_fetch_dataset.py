#!/usr/bin/env python
"""Standalone dataset resolver / inspector (bounded-effort).

Reports what data is available without running the experiment. Use it to confirm
your DFC2019 tiles are discoverable and correctly city-prefixed before a full run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config
from depthwizard.data import fetch, datasets


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    source, records = fetch.resolve_records(cfg)
    print(f"\nsource resolved: {source}")
    if records:
        cities = datasets.cities_present(records)
        print(f"triplets found : {len(records)}")
        print(f"cities present : {cities}")
        missing = [c for c in (cfg.split.train_cities + cfg.split.test_cities)
                   if c not in cities]
        if missing:
            print(f"[warn] configured cities not found in data: {missing}")
        else:
            print("configured train/test cities are all present. Ready for run_phase1.")
    else:
        print("No real triplets resolved. run_phase1 --smoke will use synthetic data "
              "(NOT evidence). See depthwizard/data/fetch.py for acquisition options.")


if __name__ == "__main__":
    main()

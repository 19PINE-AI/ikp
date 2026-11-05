#!/usr/bin/env python3
"""Build the released convenience file: the cleaned 1,311-probe subset.

All paper results are scored on the cleaned subset of the benchmark
(`final_probe_set_v8.json` minus the 89 probes flagged in `clean_mask.json`
for name-collision or label ambiguity). This script materializes that subset
as a single drop-in file so downstream users don't have to apply the mask
themselves.

Output: data/probes/final_probe_set_clean.json
  A plain JSON list of probe objects (same schema as the source:
  id, question, answer, tier, source_type, domain), tier-sorted by id.

Usage:
  python scripts/build_clean_probe_set.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "probes" / "final_probe_set_v8.json"
MASK = ROOT / "data" / "probes" / "clean_mask.json"
OUT = ROOT / "data" / "probes" / "final_probe_set_clean.json"


def main():
    probes = json.load(open(SRC, encoding="utf-8"))
    clean_ids = set(json.load(open(MASK, encoding="utf-8"))["clean_ids"])
    clean = [p for p in probes if p["id"] in clean_ids]
    clean.sort(key=lambda p: p["id"])
    OUT.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    from collections import Counter
    by_tier = dict(sorted(Counter(p["tier"] for p in clean).items()))
    print(f"Wrote {len(clean)} clean probes -> {OUT.relative_to(ROOT)}")
    print(f"  (from {len(probes)} total, {len(probes) - len(clean)} flagged)")
    print(f"  by tier: {by_tier}")


if __name__ == "__main__":
    main()

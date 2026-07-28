#!/usr/bin/env python3
"""Align IKP knowledge fingerprints with Bruckner's single-token JSD matrix.

Download ``pamela-publish-data.zip`` from DOI 10.5281/zenodo.21278557,
then run:

    python3 scripts/20_single_token_complementarity.py \
        --single-token-artifact /path/to/pamela-publish-data.zip

The external dataset is read in place and is not copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.single_token_complementarity import (
    PINNED_SINGLE_TOKEN_MATRIX_SHA256,
    align_fingerprint_pairs,
    read_verified_distance_matrix,
    render_latex_table,
    summarize_alignment,
)


DEFAULT_JSON = PROJECT_ROOT / "results" / "single_token_complementarity.json"
DEFAULT_TABLE = PROJECT_ROOT / "results" / "tables" / "single_token_complementarity.tex"
DISTANCE_MEMBER = "results/divergence-matrix.csv"
SOURCE_DOI = "10.5281/zenodo.21278557"


def load_model_ids(results_dir: Path) -> dict[str, str]:
    """Map IKP result-file stems to exact served model identifiers."""
    model_ids = {}
    for path in sorted(results_dir.glob("*.json")):
        if "think" in path.stem.lower():
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("model_id"), str):
            model_ids[path.stem] = payload["model_id"]
    return model_ids


def load_external_distances(path: Path):
    """Return the exact, digest-verified source matrix and its SHA-256."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            raw_bytes = archive.read(DISTANCE_MEMBER)
    else:
        raw_bytes = path.read_bytes()
    return read_verified_distance_matrix(
        raw_bytes,
        expected_sha256=PINNED_SINGLE_TOKEN_MATRIX_SHA256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-token-artifact",
        type=Path,
        required=True,
        help="Zenodo data ZIP or extracted results/divergence-matrix.csv",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()

    knowledge_path = PROJECT_ROOT / "results" / "comprehensive_fingerprint_results.json"
    knowledge_bytes = knowledge_path.read_bytes()
    knowledge_pairs = json.loads(knowledge_bytes)["all_pairs"]
    knowledge_sha256 = hashlib.sha256(knowledge_bytes).hexdigest()
    model_ids = load_model_ids(PROJECT_ROOT / "data" / "results")
    distances, distance_sha256 = load_external_distances(args.single_token_artifact)
    aligned = align_fingerprint_pairs(knowledge_pairs, model_ids, distances)
    summary = summarize_alignment(aligned)
    summary["source"] = {
        "single_token_dataset_doi": SOURCE_DOI,
        "single_token_matrix_member": DISTANCE_MEMBER,
        "single_token_matrix_sha256": distance_sha256,
        "ikp_fingerprint_results": str(knowledge_path.relative_to(PROJECT_ROOT)),
        "ikp_fingerprint_results_sha256": knowledge_sha256,
        "alignment": "exact served model identifier; non-thinking IKP runs only",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.output_table.write_text(render_latex_table(summary))

    print(f"Aligned {summary['n_models']} models across {summary['n_pairs']} pairs")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_table}")


if __name__ == "__main__":
    main()

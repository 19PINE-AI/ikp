"""Compare IKP knowledge fingerprints with single-token fingerprints.

The two instruments observe different black-box channels: IKP records which
rare facts a model knows (and which wrong answers it shares), whereas the
single-token instrument records distributions over repeated trivial answers.
This module aligns their pairwise data without collapsing reasoning variants
onto non-reasoning endpoints.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re

from scipy.stats import spearmanr


PINNED_SINGLE_TOKEN_MATRIX_SHA256 = (
    "0eb6821716d6420c814285db73d4edacb6c7104b576079be2500cbcda21d76a5"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def read_distance_matrix(source):
    """Read a symmetric wide CSV and retain each unordered pair once."""
    reader = csv.reader(source)
    header = next(reader)
    models = header[1:]
    distances = {}
    for row in reader:
        model_a = row[0]
        for model_b, value in zip(models, row[1:]):
            if model_a < model_b:
                distances[(model_a, model_b)] = float(value)
    return distances


def read_verified_distance_matrix(
    raw_bytes: bytes,
    expected_sha256: str = PINNED_SINGLE_TOKEN_MATRIX_SHA256,
):
    """Verify and parse the pinned external single-token distance matrix."""
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase 64-character hex digest")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "single-token distance matrix SHA-256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    source = io.StringIO(raw_bytes.decode("utf-8"))
    return read_distance_matrix(source), digest


def _is_thinking_variant(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return "think" in normalized or "reasoning" in normalized


def align_fingerprint_pairs(knowledge_pairs, model_ids, distances):
    """Join pairwise IKP metrics to exact single-token model identifiers."""
    aligned = []
    for pair_key, metrics in knowledge_pairs.items():
        short_a, short_b = pair_key.split("||", 1)
        if _is_thinking_variant(short_a) or _is_thinking_variant(short_b):
            continue
        model_a = model_ids.get(short_a)
        model_b = model_ids.get(short_b)
        if not model_a or not model_b or model_a == model_b:
            continue
        pair = tuple(sorted((model_a, model_b)))
        if pair not in distances:
            continue
        aligned.append({
            "model_a": pair[0],
            "model_b": pair[1],
            "jsd": distances[pair],
            "jaccard": metrics["jaccard"],
            "hss": metrics["hss"],
            "lift": metrics["lift"],
            "both_wrong": metrics["both_wrong"],
        })
    return aligned


def _correlation(rows, metric):
    if len(rows) < 2:
        return {"n": len(rows), "spearman_rho": None}
    result = spearmanr([-row["jsd"] for row in rows], [row[metric] for row in rows])
    return {"n": len(rows), "spearman_rho": round(float(result.statistic), 3)}


def summarize_alignment(aligned, min_joint_wrong=10):
    """Return descriptive correlations for all, within-, and cross-vendor pairs."""
    pair_ids = [(row["model_a"], row["model_b"]) for row in aligned]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("aligned model pairs must be unique")
    models = {model for pair in pair_ids for model in pair}
    expected_pairs = len(models) * (len(models) - 1) // 2
    if len(pair_ids) != expected_pairs:
        raise ValueError(
            "aligned model pairs must form a complete matrix: "
            f"expected {expected_pairs}, got {len(pair_ids)}"
        )

    groups = {
        "all": aligned,
        "same_vendor": [
            row for row in aligned
            if row["model_a"].split("/", 1)[0] == row["model_b"].split("/", 1)[0]
        ],
        "cross_vendor": [
            row for row in aligned
            if row["model_a"].split("/", 1)[0] != row["model_b"].split("/", 1)[0]
        ],
    }
    correlations = {}
    for group_name, rows in groups.items():
        hss_rows = [row for row in rows if row["both_wrong"] >= min_joint_wrong]
        correlations[group_name] = {
            "jaccard": _correlation(rows, "jaccard"),
            "hss": _correlation(hss_rows, "hss"),
        }
    return {
        "n_models": len(models),
        "n_pairs": len(aligned),
        "min_joint_wrong_for_hss": min_joint_wrong,
        "correlations": correlations,
    }


def render_latex_table(summary):
    """Render the descriptive correlations as a compact LaTeX table."""
    labels = {
        "all": "All pairs",
        "same_vendor": "Same vendor",
        "cross_vendor": "Cross vendor",
    }
    metric_labels = {"jaccard": "Jaccard", "hss": "HSS"}
    rows = []
    for group in ("all", "same_vendor", "cross_vendor"):
        for metric in ("jaccard", "hss"):
            result = summary["correlations"][group][metric]
            rho = "---" if result["spearman_rho"] is None else f'{result["spearman_rho"]:.3f}'
            rows.append(
                f'{labels[group]} & {metric_labels[metric]} & {result["n"]} & {rho} \\\\'
            )
    body = "\n".join(rows)
    return f"""% Generated by scripts/20_single_token_complementarity.py.
\\begin{{table}}[H]
    \\centering
    \\small
    \\begin{{tabular}}{{llrr}}
        \\toprule
        Pair subset & IKP metric & $n$ pairs & Spearman $\\rho$ \\\\
        \\midrule
{body}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Descriptive rank association between single-token behavioral similarity ($-\\mathrm{{JSD}}$) and IKP knowledge-fingerprint similarity. HSS rows require at least ten probes on which both models are wrong. Pairwise observations share models, so ordinary significance tests for unrelated pairs do not apply; the correlations quantify signal overlap only.}}
    \\label{{tab:single-token-complementarity}}
\\end{{table}}
"""

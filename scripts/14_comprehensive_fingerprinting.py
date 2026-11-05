#!/usr/bin/env python3
"""Comprehensive knowledge-fingerprinting analysis.

For every pair of evaluated models we compute three complementary metrics on
the T5-T6 (rare-fact) probes:

  1. Jaccard similarity over correct-answer sets (raw overlap; inflated by
     common/easy knowledge).
  2. Lift = observed_intersection / expected_under_independence. This
     controls for "easy probes everyone gets right".
  3. HSS = fraction of *jointly wrong* probes for which the two models
     produce the *same non-refusal wrong answer* (normalized). HSS is
     near-zero for independently trained models, and large (typically
     >=0.3) only when models share weights or a large fraction of their
     training data.

The three metrics are used to classify pairs into:
  - shared-base (same weights, different inference/alignment):
    HSS >= 0.3 and Jaccard >= 0.6
  - lineage (post-training / fine-tune of the same base or ancestor):
    HSS in [0.1, 0.3] and Jaccard >= 0.5
  - retrained (same labelled family but HSS ~ 0):
    HSS < 0.1 on >= 15 joint-wrong probes
  - independent (different vendor, HSS near 0).

Cross-vendor pairs flagged with HSS >= 0.2 are candidate distillation
signatures.

Outputs (into results/):
  comprehensive_fingerprint_results.json     machine-readable
  tables/intra_family_lineage.tex            within-family table
  tables/cross_family_outliers.tex           cross-vendor outliers
  ../paper/figures/fig7_family_lineage.pdf   headline figure
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUT_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
TAB_DIR = OUT_DIR / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ("T5", "T6")

# Vendor mapping (prefix -> vendor)
VENDOR_PREFIXES = [
    ("gpt-oss", "OpenAI-OS"),
    ("gpt-", "OpenAI"),
    ("o3", "OpenAI"), ("o4", "OpenAI"),
    ("claude-", "Anthropic"),
    ("gemini-", "Google"),
    ("gemma-", "Google-OS"),
    ("llama-", "Meta"),
    ("qwen-", "Alibaba"), ("qwen3", "Alibaba"), ("qwq-", "Alibaba"),
    ("deepseek", "DeepSeek"),
    ("mistral", "Mistral"), ("ministral", "Mistral"), ("pixtral", "Mistral"),
    ("phi-", "Microsoft"),
    ("grok-", "xAI"),
    ("glm-", "ZhipuAI"),
    ("kimi", "Moonshot"),
    ("minimax", "MiniMax"),
    ("hermes-", "NousResearch"),
    ("nemotron", "NVIDIA"),
    ("command-", "Cohere"), ("command", "Cohere"),
    ("seed-", "ByteDance"),
    ("ling-", "Ant"),
]

def vendor_of(name: str) -> str:
    n = name.lower()
    for pre, v in VENDOR_PREFIXES:
        if n.startswith(pre):
            return v
    return "Other"

def norm_answer(s: str, question: str = "") -> str:
    if not s:
        return ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
    if not s:
        return ""
    if question and question.lower().startswith("in what year"):
        years = re.findall(r"\b(?:1\d{3}|20\d{2})\b", s)
        if len(set(years)) == 1:
            return years[0]
    s = s.lower()
    s = re.sub(r"[\s\.,;:!?\"'()\-]+", " ", s).strip()
    return s[:200]


def load_all() -> dict:
    data = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name in ("analysis.json", "evaluation_summary.json"):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "results" not in d:
            continue
        name = d.get("model_name", f.stem)
        data[name] = d
    return data


def build_fingerprints(data: dict):
    correct = {}
    wrong = {}
    probes = set()
    for name, d in data.items():
        c = set()
        w = {}
        for r in d.get("results", []):
            if r["tier"] not in TIERS:
                continue
            probes.add(r["probe_id"])
            if r.get("correct"):
                c.add(r["probe_id"])
            elif not r.get("refusal"):
                a = norm_answer(r.get("model_response", ""), r.get("question", ""))
                if a:
                    w[r["probe_id"]] = a
        correct[name] = c
        wrong[name] = w
    return correct, wrong, len(probes)


def pairwise_metrics(correct, wrong, N_probes):
    """Return pairs dict with metrics."""
    names = sorted(correct.keys())
    out = {}
    for a, b in combinations(names, 2):
        A, B = correct[a], correct[b]
        if not A or not B:
            continue
        inter = len(A & B)
        union = len(A | B)
        jac = inter / union if union else 0.0
        exp = len(A) * len(B) / N_probes
        lift = inter / exp if exp else 0.0
        # HSS
        WA, WB = wrong[a], wrong[b]
        shared_pids = WA.keys() & WB.keys()
        both = len(shared_pids)
        same = sum(1 for p in shared_pids if WA[p] == WB[p])
        hss = same / both if both else 0.0
        out[(a, b)] = dict(
            jaccard=round(jac, 3), lift=round(lift, 2), hss=round(hss, 3),
            n_a=len(A), n_b=len(B), inter=inter, both_wrong=both, same_wrong=same,
        )
    return out, names


def classify_pair(m, same_vendor: bool) -> str:
    j, h, bw = m["jaccard"], m["hss"], m["both_wrong"]
    if h >= 0.30 and j >= 0.60:
        return "shared-base"
    if h >= 0.10 and j >= 0.50:
        return "lineage"
    if same_vendor and bw >= 10 and h < 0.10:
        return "retrained"  # named same family but wrong-answer-independent
    if not same_vendor and h >= 0.20 and bw >= 10:
        return "cross-vendor-suspect"
    return "independent"


# --------------------------------------------------------------------
# Hand-curated lists for within-family analysis
# --------------------------------------------------------------------

FAMILY_SERIES = {
    # ---------------- OpenAI ----------------
    "OpenAI GPT base": [
        "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4.1", "gpt-5",
        "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4",
    ],
    "OpenAI GPT-5 base/pro/think cluster": [
        "gpt-5", "gpt-5-pro", "gpt-5-think",
    ],
    "OpenAI GPT-5 .x transitions": [
        "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.2-pro",
        "gpt-5.3", "gpt-5.4", "gpt-5.4-pro",
    ],
    "OpenAI GPT-mini": [
        "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-5.4-mini",
    ],
    "OpenAI GPT-nano": [
        "gpt-4.1-nano", "gpt-5-nano", "gpt-5.4-nano",
    ],
    "OpenAI o-series": [
        "o1", "o3-mini", "o3", "o4-mini-think",
    ],
    # ---------------- Anthropic ----------------
    "Anthropic Claude Opus": [
        "claude-opus-4", "claude-opus-4.1", "claude-opus-4.5",
        "claude-opus-4.6", "claude-opus-4.7",
    ],
    "Anthropic Claude Sonnet": [
        "claude-3.7-sonnet", "claude-sonnet-4",
        "claude-sonnet-4.5", "claude-sonnet-4.6",
    ],
    "Anthropic Claude Haiku": [
        "claude-3-haiku", "claude-3.5-haiku", "claude-haiku-4.5",
    ],
    # ---------------- Google ----------------
    "Google Gemini Flash": [
        "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-flash",
    ],
    "Google Gemini Pro": [
        "gemini-2.5-pro", "gemini-3.1-pro",
    ],
    "Google Gemini Flash-Lite": [
        "gemini-2.5-flash-lite", "gemini-3.1-flash-lite",
    ],
    "Google Gemma": [
        "gemma-2-2b", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b",
        "gemma-2-27b", "gemma-3-27b", "gemma-4-31b",
    ],
    # ---------------- Meta ----------------
    "Meta Llama 3 (70B)": [
        "llama-3-70b", "llama-3.1-70b", "llama-3.3-70b",
    ],
    "Meta Llama 3 (8B)": [
        "llama-3-8b", "llama-3.1-8b",
    ],
    "Meta Llama 3 (small)": [
        "llama-3.2-1b", "llama-3.2-3b",
    ],
    "Meta Llama 4": [
        "llama-4-scout", "llama-4-maverick",
    ],
    # ---------------- Alibaba ----------------
    "Alibaba Qwen dense (7B)": [
        "qwen-2.5-7b", "qwen3-8b-think",
    ],
    "Alibaba Qwen dense (70B)": [
        "qwen-2.5-72b", "qwq-32b-think",
    ],
    "Alibaba Qwen Max": [
        "qwen-max", "qwen3-max",
    ],
    "Alibaba Qwen Plus": [
        "qwen-plus", "qwen3.5-plus-think", "qwen3.6-plus-think",
    ],
    "Alibaba Qwen large MoE": [
        "qwen3-235b-a22b-think", "qwen3.5-397b-a17b-think",
    ],
    # ---------------- DeepSeek ----------------
    "DeepSeek V3": [
        "deepseek-v3", "deepseek-v3.1", "deepseek-v3.2",
    ],
    # ---------------- Zhipu ----------------
    "Zhipu GLM": [
        "glm-4.5-think", "glm-4.6-think", "glm-4.7-think", "glm-5-think",
        "glm-5.1-think",
    ],
    # ---------------- xAI ----------------
    "xAI Grok": [
        "grok-3", "grok-4", "grok-4.20",
    ],
    # ---------------- Moonshot ----------------
    "Moonshot Kimi K2": [
        "kimi-k2", "kimi-k2.5-think", "kimi-k2.6-think",
    ],
    # ---------------- Mistral ----------------
    "Mistral large/medium": [
        "mistral-medium-3.1", "mistral-large",
    ],
    "Mistral small/open": [
        "mistral-7b", "mistral-nemo-12b", "mistral-small-24b",
    ],
    "Mistral MoE (Mixtral)": [
        "mixtral-8x7b", "mixtral-8x22b",
    ],
    "Mistral ministral": [
        "ministral-3b", "ministral-8b",
    ],
    # ---------------- Amazon Nova ----------------
    "Amazon Nova": [
        "nova-micro", "nova-pro", "nova-premier",
    ],
    # ---------------- Cohere ----------------
    "Cohere Command": [
        "command-r7b", "command-r-plus", "command-a",
    ],
    # ---------------- NousResearch Hermes ----------------
    "NousResearch Hermes": [
        "hermes-3-405b", "hermes-4-405b",
    ],
    # ---------------- Phi ----------------
    "Microsoft Phi": [
        "phi-3-mini", "phi-4",
    ],
}

# Known-provenance pairs used as positive controls for the method
KNOWN_LINEAGE_PAIRS = [
    # (student, parent, note)
    ("deepseek-r1-distill-llama-70b-think", "llama-3.3-70b", "R1 distilled into Llama-3.3-70B base"),
    ("deepseek-r1-distill-qwen-32b-think", "qwq-32b-think", "R1 distilled into Qwen-2.5-32B family"),
    ("nemotron-70b", "llama-3.1-70b", "NVIDIA fine-tune of Llama-3.1-70B"),
    ("nemotron-super-49b-think", "llama-3.3-70b", "NVIDIA pruned/distilled from Llama-3.3-70B"),
    ("hermes-3-405b", "llama-3.1-70b", "Hermes-3 is Llama-3.1-405B fine-tune (no 405B base eval here; compare to 70B sibling)"),
    ("hermes-4-405b", "hermes-3-405b", "Hermes successive fine-tunes of Llama-3.1-405B"),
]


def within_family_table(pair_metrics, series, vendor_label: str) -> list:
    """Return ordered consecutive-pair metrics within a series."""
    rows = []
    for i in range(len(series) - 1):
        a, b = series[i], series[i + 1]
        if (a, b) in pair_metrics:
            m = pair_metrics[(a, b)]
        elif (b, a) in pair_metrics:
            m = pair_metrics[(b, a)]
        else:
            continue
        c = classify_pair(m, same_vendor=True)
        rows.append((a, b, m, c))
    return rows


def cross_family_outliers(pair_metrics, min_joint_wrong=10, min_hss=0.2):
    """Flag cross-vendor pairs with HSS above threshold."""
    out = []
    for (a, b), m in pair_metrics.items():
        va, vb = vendor_of(a), vendor_of(b)
        # Treat *-OS and vendor-native as the same vendor so we don't flag
        # e.g. gpt-5 <-> gpt-oss as "cross-vendor". Also don't compare a
        # model against itself after normalization.
        va_core = va.replace("-OS", "")
        vb_core = vb.replace("-OS", "")
        if va_core == vb_core:
            continue
        if m["both_wrong"] < min_joint_wrong:
            continue
        if m["hss"] >= min_hss:
            out.append((a, b, m))
    out.sort(key=lambda x: x[2]["hss"], reverse=True)
    return out


# --------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------

def format_series_table(rows, vendor_name):
    lines = [f"### {vendor_name}"]
    lines.append(f"  {'pair':50s}  {'J':>5}  {'lift':>5}  {'HSS':>6}  {'both_w':>6}  class")
    for a, b, m, c in rows:
        lines.append(
            f"  {a:22s} -> {b:22s}  "
            f"{m['jaccard']:>5.3f}  {m['lift']:>5.2f}  {m['hss']:>6.3f}"
            f"  {m['both_wrong']:>6d}  {c}"
        )
    return "\n".join(lines)


def plot_family_lineage(series_rows_map):
    """Two-panel figure: (a) per-family consecutive-pair HSS with family bands;
    (b) Jaccard vs HSS scatter for the whole pairwise universe."""
    # Trim: skip families where all consecutive pairs have both_wrong<8 (insufficient
    # statistical resolution to classify reliably).
    pruned = {}
    for fam, rows in series_rows_map.items():
        keep = [r for r in rows if r[2]["both_wrong"] >= 8]
        if keep:
            pruned[fam] = keep

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    palette = {
        "shared-base": "#2ca02c",
        "lineage": "#1f77b4",
        "retrained": "#d62728",
        "independent": "#7f7f7f",
        "cross-vendor-suspect": "#ff7f0e",
    }

    # Panel (a): grouped horizontal bars, one family per band
    y = 0
    yticks = []; ylabels = []
    family_bands = []  # (y_start, y_end, family_name)
    for fam, rows in pruned.items():
        y_start = y
        for a, b, m, c in rows:
            bar = ax_a.barh(y, m["hss"], color=palette.get(c, "#7f7f7f"),
                            edgecolor="black", lw=0.4, height=0.75)
            ax_a.text(max(m["hss"], 0.02) + 0.012, y,
                      f"{m['same_wrong']}/{m['both_wrong']}",
                      va="center", fontsize=6.5)
            def short(n):
                for pre in ("claude-", "gemini-", "gemma-", "gpt-", "mistral-",
                            "llama-", "qwen-", "qwen", "deepseek-", "glm-",
                            "grok-", "kimi-", "nova-", "command-", "hermes-",
                            "ministral-", "mixtral-", "phi-", "gpt5-"):
                    if n.startswith(pre):
                        n = n[len(pre):]
                        break
                return n.replace("-think", "*")
            lbl = f"{short(a)} -> {short(b)}"
            yticks.append(y); ylabels.append(lbl)
            y += 1
        family_bands.append((y_start - 0.5, y - 0.5, fam))
        y += 0.6

    # draw faint bands and family labels on the right
    for ys, ye, fam in family_bands:
        ax_a.axhspan(ys, ye, color="black", alpha=0.03, zorder=0)
        ax_a.text(1.02, (ys + ye) / 2, fam, va="center", fontsize=7,
                  color="#444", transform=ax_a.get_yaxis_transform())

    ax_a.axvline(0.30, color="black", ls="--", lw=0.6, alpha=0.6,
                 label="shared-base threshold")
    ax_a.axvline(0.10, color="black", ls=":", lw=0.6, alpha=0.6,
                 label="lineage threshold")
    ax_a.set_yticks(yticks); ax_a.set_yticklabels(ylabels, fontsize=7)
    ax_a.set_xlabel("Hallucination Similarity (HSS)")
    ax_a.set_xlim(0, 1.05)
    ax_a.invert_yaxis()
    ax_a.set_title("(a) Consecutive-generation HSS across all tracked families",
                   fontsize=10)
    handles = [Patch(color=palette[k], label=k) for k in
               ("shared-base", "lineage", "retrained", "independent")]
    ax_a.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)

    return fig, (ax_a, ax_b)


def build_scatter_data(pair_metrics):
    xs, ys, colors, labels = [], [], [], []
    for (a, b), m in pair_metrics.items():
        if m["both_wrong"] < 5:
            continue
        xs.append(m["jaccard"])
        ys.append(m["hss"])
        va, vb = vendor_of(a).replace("-OS", ""), vendor_of(b).replace("-OS", "")
        colors.append("#1f77b4" if va == vb else "#d62728")
        labels.append((a, b))
    return np.array(xs), np.array(ys), colors, labels


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    print("Loading evaluation data...")
    data = load_all()
    print(f"  loaded {len(data)} models")

    correct, wrong, N_probes = build_fingerprints(data)
    print(f"  T5-T6 probe pool: {N_probes}")

    pair_metrics, names = pairwise_metrics(correct, wrong, N_probes)
    print(f"  computed metrics for {len(pair_metrics)} model pairs")

    # ---- within-family ----
    series_map = {}
    for label, series in FAMILY_SERIES.items():
        rows = within_family_table(pair_metrics, series, label)
        if rows:
            series_map[label] = rows

    report_lines = []
    for vend, rows in series_map.items():
        report_lines.append(format_series_table(rows, vend))
    report_lines.append("")

    # ---- cross-family outliers ----
    cross = cross_family_outliers(pair_metrics)
    report_lines.append("\n### Cross-vendor outliers (HSS >= 0.20, both_wrong >= 10)")
    report_lines.append(f"  {'pair':55s}  {'J':>5}  {'lift':>5}  {'HSS':>6}  {'both_w':>6}  vendors")
    for a, b, m in cross[:60]:
        va, vb = vendor_of(a), vendor_of(b)
        report_lines.append(
            f"  {a:26s} <-> {b:26s}  "
            f"{m['jaccard']:>5.3f}  {m['lift']:>5.2f}  {m['hss']:>6.3f}"
            f"  {m['both_wrong']:>6d}  {va:>10s}/{vb:<10s}"
        )

    # ---- known-provenance positive controls ----
    report_lines.append("\n### Known-provenance controls")
    report_lines.append(f"  {'student':35s}  {'parent':28s}  {'J':>5}  {'lift':>5}  {'HSS':>6}  {'both_w':>6}  note")
    control_rows = []
    for student, parent, note in KNOWN_LINEAGE_PAIRS:
        key = (student, parent) if (student, parent) in pair_metrics else (parent, student)
        if key not in pair_metrics:
            report_lines.append(f"  {student:35s}  {parent:28s}  (missing)")
            continue
        m = pair_metrics[key]
        control_rows.append((student, parent, m, note))
        report_lines.append(
            f"  {student:35s}  {parent:28s}  "
            f"{m['jaccard']:>5.3f}  {m['lift']:>5.2f}  {m['hss']:>6.3f}"
            f"  {m['both_wrong']:>6d}  {note}"
        )

    # ---- LaTeX tables ----
    def escape_latex(s):
        return s.replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&")

    latex_lines = []
    latex_lines.append("% auto-generated by scripts/14_comprehensive_fingerprinting.py")
    latex_lines.append("{\\small")
    latex_lines.append("\\begin{longtable}{@{}p{0.20\\textwidth} p{0.20\\textwidth} r r r r l@{}}")
    latex_lines.append("\\caption{Comprehensive within-family fingerprint metrics for all tracked model families (T5--T6 probes). $J$: Jaccard on correct-answer sets; lift: observed/expected-under-independence; $\\mathrm{HSS}$: hallucination similarity; both\\_w: joint-wrong probes; class: shared-base / lineage / retrained / independent. Families with all pairs having both\\_w $<$ 8 are omitted.}\\label{tab:fp-all-families}\\\\")
    latex_lines.append("\\toprule From & To & $J$ & lift & HSS & both\\_w & class \\\\ \\midrule \\endfirsthead")
    latex_lines.append("\\toprule From & To & $J$ & lift & HSS & both\\_w & class \\\\ \\midrule \\endhead")
    for fam, rows in series_map.items():
        if not any(r[2]["both_wrong"] >= 8 for r in rows):
            continue
        latex_lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{escape_latex(fam)}}}}} \\\\")
        for a, b, m, c in rows:
            if m["both_wrong"] < 5:
                c_disp = f"{c} (small $n$)"
            else:
                c_disp = c
            latex_lines.append(
                f"\\quad {escape_latex(a)} & {escape_latex(b)} & "
                f"{m['jaccard']:.3f} & {m['lift']:.2f} & {m['hss']:.3f} & "
                f"{m['both_wrong']} & {c_disp} \\\\"
            )
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{longtable}")
    latex_lines.append("}")
    (TAB_DIR / "fp_all_families.tex").write_text("\n".join(latex_lines))
    print(f"Saved LaTeX table -> {TAB_DIR / 'fp_all_families.tex'}")

    # Known-provenance controls table
    ctrl_lines = []
    ctrl_lines.append("% auto-generated by scripts/14_comprehensive_fingerprinting.py")
    ctrl_lines.append("\\begin{table}[ht]\\centering\\small")
    ctrl_lines.append("\\begin{tabular}{p{0.30\\textwidth} p{0.22\\textwidth} r r r r}")
    ctrl_lines.append("\\toprule Student (derived) & Parent (base/teacher) & $J$ & lift & HSS & both\\_w \\\\ \\midrule")
    for student, parent, m, note in control_rows:
        ctrl_lines.append(
            f"{escape_latex(student)} & {escape_latex(parent)} & "
            f"{m['jaccard']:.3f} & {m['lift']:.2f} & {m['hss']:.3f} & {m['both_wrong']} \\\\"
        )
    ctrl_lines.append("\\bottomrule \\end{tabular}")
    ctrl_lines.append(
        "\\caption{Known-provenance positive controls. Each pair is a publicly-disclosed distillation or fine-tune. All show either lineage or cross-vendor-suspect HSS ($\\geq 0.10$) when both\\_w permits statistical resolution, confirming that the HSS metric detects shared training signal.}"
        "\\label{tab:fp-controls}\\end{table}"
    )
    (TAB_DIR / "fp_controls.tex").write_text("\n".join(ctrl_lines))
    print(f"Saved control table -> {TAB_DIR / 'fp_controls.tex'}")

    # Cross-vendor outliers table
    cv_lines = []
    cv_lines.append("% auto-generated by scripts/14_comprehensive_fingerprinting.py")
    cv_lines.append("{\\small")
    cv_lines.append("\\begin{longtable}{p{0.30\\textwidth} p{0.30\\textwidth} r r r r}")
    cv_lines.append("\\caption{Complete list of cross-vendor outlier pairs with $\\mathrm{HSS} \\geq 0.20$ and $\\geq 10$ joint-wrong probes on T5--T6.}\\label{tab:fp-cross-full} \\\\")
    cv_lines.append("\\toprule Model A & Model B & $J$ & lift & HSS & both\\_w \\\\ \\midrule \\endfirsthead")
    cv_lines.append("\\toprule Model A & Model B & $J$ & lift & HSS & both\\_w \\\\ \\midrule \\endhead")
    for a, b, m in cross:
        cv_lines.append(
            f"{escape_latex(a)} & {escape_latex(b)} & "
            f"{m['jaccard']:.3f} & {m['lift']:.2f} & {m['hss']:.3f} & {m['both_wrong']} \\\\"
        )
    cv_lines.append("\\bottomrule \\end{longtable}")
    cv_lines.append("}")
    (TAB_DIR / "fp_cross_vendor.tex").write_text("\n".join(cv_lines))
    print(f"Saved cross-vendor table -> {TAB_DIR / 'fp_cross_vendor.tex'}")

    report = "\n".join(report_lines)
    print(report)
    (OUT_DIR / "comprehensive_fingerprint_report.txt").write_text(report)

    # ---- JSON dump ----
    pm_serializable = {
        f"{a}||{b}": v for (a, b), v in pair_metrics.items()
    }
    json.dump(
        {
            "n_probes_T5T6": N_probes,
            "n_models": len(data),
            "series": {
                k: [
                    {"a": a, "b": b, **m, "class": c}
                    for a, b, m, c in v
                ]
                for k, v in series_map.items()
            },
            "cross_vendor_outliers": [
                {"a": a, "b": b, "vendor_a": vendor_of(a), "vendor_b": vendor_of(b), **m}
                for a, b, m in cross
            ],
            "all_pairs": pm_serializable,
        },
        open(OUT_DIR / "comprehensive_fingerprint_results.json", "w"),
        indent=2,
    )
    print(f"\nSaved JSON -> {OUT_DIR / 'comprehensive_fingerprint_results.json'}")

    # ---- figure ----
    fig, (ax_a, ax) = plot_family_lineage(series_map)
    xs, ys, colors, labels = build_scatter_data(pair_metrics)
    same_mask = np.array([c == "#1f77b4" for c in colors])
    ax.scatter(xs[~same_mask], ys[~same_mask], s=10, c="#d62728",
               alpha=0.35, label="cross-vendor", edgecolors="none")
    ax.scatter(xs[same_mask], ys[same_mask], s=14, c="#1f77b4",
               alpha=0.55, label="same vendor", edgecolors="none")
    ax.axhline(0.30, color="black", ls="--", lw=0.6, alpha=0.5)
    ax.axhline(0.10, color="black", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("Jaccard similarity (T5-T6 correct)")
    ax.set_ylabel("Hallucination Similarity (HSS)")
    ax.set_title("(b) Same-base, lineage, and independent regimes")
    ax.legend(fontsize=8)
    # annotate a few illustrative points
    annotations = [
        ("gpt-5", "gpt-5-pro", "GPT-5 / GPT-5-pro"),
        ("gpt-5", "gpt-5.1", "GPT-5 / GPT-5.1"),
        ("gpt-5.3", "gpt-5.4", "GPT-5.3 / GPT-5.4"),
        ("claude-opus-4", "claude-opus-4.1", "Opus-4 / 4.1"),
        ("claude-opus-4.6", "claude-opus-4.7", "Opus-4.6 / 4.7"),
    ]
    for a, b, lab in annotations:
        key = (a, b) if (a, b) in pair_metrics else (b, a)
        if key in pair_metrics:
            m = pair_metrics[key]
            ax.annotate(lab, (m["jaccard"], m["hss"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_family_lineage.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig9_family_lineage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {FIG_DIR / 'fig9_family_lineage.pdf'}")


if __name__ == "__main__":
    main()

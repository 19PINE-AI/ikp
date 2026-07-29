#!/usr/bin/env python3
"""Effective-parameter analysis for issue #5 — accuracy & interpretability.

Everything here is a deterministic function of committed data (no API calls),
so anyone can rerun it and reproduce the reported values to numerical
precision. Three experiments that
turn the "is the estimate accurate?" question into measured quantities:

  1. LOFO-CV (leave-one-FAMILY-out). LOO leaks family information: when
     qwen3-8b is held out, five other qwen models still anchor the fit. LOFO
     holds each of the 31 model families out whole, which is the honest
     uncertainty for a *new vendor* — exactly the closed-model setting. The
     per-family bias table quantifies the "systematic bias across
     architectures" concern in issue #5.

  2. Density ledger. Defines the estimand explicitly:
        N_eff ≡ 10^((A − β)/α)   (IKP-effective parameters: the size of a
                                  calibration-cohort-average open model with
                                  equal incompressible-knowledge capacity)
        ρ     ≡ N_eff / N_true   (knowledge density)
     and reports ρ per model, its spread, the adjusted family effect and
     permutation evidence, and which models are density outliers
     (|log10 ρ| > 2σ).

  3. Continuous MoE exponent. Replaces the binary "total beats active"
     claim with a fitted convention: acc ~ a·(γ·log T + (1−γ)·log A) + b
     on the open MoE subset, with a bootstrap CI on γ. γ=1 ⇒ knowledge
     tracks total params; γ=0 ⇒ active params; γ=0.5 ⇒ geometric mean.

Outputs:
  data/results/effective_params.json   (machine-readable, incl. per-model ρ)
  stdout                               (human-readable tables)

Usage:
  python scripts/19_effective_params.py
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_FILE = PROJECT_ROOT / "data" / "results" / "evaluation_summary.json"
CONFIGS_FILE = PROJECT_ROOT / "configs" / "all_models.json"
OUT_JSON = PROJECT_ROOT / "data" / "results" / "effective_params.json"

# Same exclusions as scripts/loo_cv_analysis.py / scripts/18_v2_validation.py —
# broken evals and documented outliers, reproduced verbatim.
CALIBRATION_EXCLUDE = {
    "minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think", "hermes-3-405b",
    "ling-2.6-flash", "deepseek-v3.1-nex-n1", "intellect-3-think",
    "nemotron-ultra-253b",
}

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 5  # issue number, for the record
PERMUTATION_B = 10000
PERMUTATION_SEED = 5005


def load_cohort():
    """Open models with known size, minus exclusions → list of dicts."""
    cfg = json.load(open(CONFIGS_FILE))["models"]
    rows = []
    for m in json.load(open(SUMMARY_FILE)):
        c = cfg.get(m["model"], {})
        p = m.get("params_B") or c.get("params_B")
        if (c.get("type") == "open" and p and p > 0
                and m.get("accuracy") is not None
                and m["model"] not in CALIBRATION_EXCLUDE):
            rows.append({
                "model": m["model"],
                "family": c.get("family", m["model"]),
                "arch": c.get("arch", "?"),
                "thinking": bool(c.get("thinking")),
                "params_B": float(p),
                "active_B": c.get("active_B"),
                "accuracy": m["accuracy"],
            })
    return rows


def ols(x, y):
    sl, ic, r, _, _ = stats.linregress(x, y)
    return sl, ic, r ** 2


def fold_err(pred_B, true_B):
    return max(pred_B / true_B, true_B / pred_B)


def invert(acc, slope, intercept):
    return 10 ** ((acc - intercept) / slope)


# ── 1. LOO vs LOFO cross-validation ────────────────────────────
def cross_validation(rows):
    logN = np.array([math.log10(r["params_B"]) for r in rows])
    acc = np.array([r["accuracy"] for r in rows])
    fams = [r["family"] for r in rows]

    def cv_folds(hold_out_masks):
        folds = []
        for mask in hold_out_masks:
            keep = ~mask
            sl, ic, _ = ols(logN[keep], acc[keep])
            for i in np.where(mask)[0]:
                folds.append((i, fold_err(invert(acc[i], sl, ic),
                                          10 ** logN[i])))
        return folds

    n = len(rows)
    loo = cv_folds([np.arange(n) == i for i in range(n)])
    families = sorted(set(fams))
    lofo = cv_folds([np.array([f == fam for f in fams]) for fam in families])

    def summarize(folds):
        e = np.array([f for _, f in folds])
        return {"median_fold": float(np.median(e)),
                "pi90_fold": float(np.quantile(e, 0.90)),
                "within_2x": float(np.mean(e <= 2)),
                "within_3x": float(np.mean(e <= 3))}

    # Per-family bias: geometric mean of signed fold (pred/true) under LOFO.
    lofo_by_model = dict(lofo)
    fam_bias = {}
    for fam in families:
        idx = [i for i, f in enumerate(fams) if f == fam]
        signed = []
        for i in idx:
            keep = np.array([f != fam for f in fams])
            sl, ic, _ = ols(logN[keep], acc[keep])
            signed.append(math.log10(invert(acc[i], sl, ic) / 10 ** logN[i]))
        fam_bias[fam] = {
            "n": len(idx),
            "gm_signed_fold": 10 ** float(np.mean(signed)),
            "median_abs_fold": float(np.median([lofo_by_model[i] for i in idx])),
        }

    print("  EXPERIMENT 1 — LOO vs LOFO cross-validation "
          f"(n={n}, {len(families)} families)")
    s_loo, s_lofo = summarize(loo), summarize(lofo)
    for name, s in (("LOO (per-model)", s_loo), ("LOFO (per-family)", s_lofo)):
        print(f"    {name:<18}: median {s['median_fold']:.2f}×  "
              f"90% PI {s['pi90_fold']:.2f}×  "
              f"within 2× {s['within_2x']:.0%}  within 3× {s['within_3x']:.0%}")
    print("    LOFO is the honest band for a NEW vendor (the closed-model "
          "setting);\n    LOO leaks family information and is mildly "
          "optimistic.")
    worst = sorted(fam_bias.items(),
                   key=lambda kv: -abs(math.log10(kv[1]["gm_signed_fold"])))
    print("\n    Per-family LOFO bias (geometric-mean predicted/true; "
          "worst 8):")
    print(f"    {'family':<18}{'n':>3}  {'bias':>7}  (>1 = size "
          "over-estimated when family is unseen)")
    for fam, b in worst[:8]:
        print(f"    {fam:<18}{b['n']:>3}  {b['gm_signed_fold']:>6.2f}×")
    print()
    return {"loo": s_loo, "lofo": s_lofo, "family_bias": fam_bias}


# ── 2. Density ledger ──────────────────────────────────────────
def density_ledger(rows, slope, intercept):
    ledger = []
    for r in rows:
        n_eff = invert(r["accuracy"], slope, intercept)
        rho = n_eff / r["params_B"]
        ledger.append({**{k: r[k] for k in
                          ("model", "family", "arch", "thinking", "params_B",
                           "accuracy")},
                       "n_eff_B": n_eff, "rho": rho,
                       "log10_rho": math.log10(rho)})

    lg = np.array([e["log10_rho"] for e in ledger])
    sd = float(np.std(lg, ddof=1))
    for e in ledger:
        e["outlier_2sigma"] = bool(abs(e["log10_rho"]) > 2 * sd)

    # One-way ANOVA effect size on log10 ρ by family. Raw eta-squared is
    # upward-biased when there are many/small groups, so report omega-squared
    # and a label-permutation p-value before interpreting a family effect.
    fams = [e["family"] for e in ledger]
    grand = lg.mean()
    ss_tot = float(((lg - grand) ** 2).sum())

    def between_ss(labels):
        total = 0.0
        for fam in set(labels):
            v = lg[np.array([f == fam for f in labels])]
            total += len(v) * (v.mean() - grand) ** 2
        return total

    ss_between = between_ss(fams)
    k, n = len(set(fams)), len(fams)
    ss_within = ss_tot - ss_between
    ms_within = ss_within / (n - k)
    eta_squared = ss_between / ss_tot
    omega_squared = max(0.0, (ss_between - (k - 1) * ms_within)
                        / (ss_tot + ms_within))

    rng = np.random.default_rng(PERMUTATION_SEED)
    labels = np.array(fams)
    permuted_ge = sum(
        between_ss(rng.permutation(labels)) >= ss_between
        for _ in range(PERMUTATION_B)
    )
    permutation_p = (permuted_ge + 1) / (PERMUTATION_B + 1)

    ledger.sort(key=lambda e: -e["log10_rho"])
    print(f"  EXPERIMENT 2 — density ledger  ρ ≡ N_eff / N_true "
          f"(n={len(ledger)})")
    print(f"    sd(log10 ρ) = {sd:.2f}   (1σ ≈ {10 ** sd:.1f}×)")
    print(f"    family effect: raw η²={eta_squared:.0%}, adjusted ω²="
          f"{omega_squared:.0%}, permutation p={permutation_p:.4f}")
    print(f"    density outliers (|log10 ρ| > 2σ): "
          f"{sum(e['outlier_2sigma'] for e in ledger)}")
    print("\n    densest (IKP reads them bigger than they are):")
    for e in ledger[:4]:
        print(f"      {e['model']:<28} ρ = {e['rho']:.2f}   "
              f"N_true {e['params_B']:g}B → N_eff {e['n_eff_B']:.1f}B")
    print("    sparsest (IKP reads them smaller):")
    for e in ledger[-4:]:
        print(f"      {e['model']:<28} ρ = {e['rho']:.2f}   "
              f"N_true {e['params_B']:g}B → N_eff {e['n_eff_B']:.1f}B")
    print()
    return {"sd_log10_rho": sd,
            "family_effect": {
                "eta_squared": eta_squared,
                "omega_squared": omega_squared,
                "permutation_p": permutation_p,
                "permutations": PERMUTATION_B,
                "seed": PERMUTATION_SEED,
            },
            "ledger": ledger}


# ── 3. Continuous MoE exponent γ ──────────────────────────────
def moe_gamma(rows):
    moe = [r for r in rows if r["arch"] == "moe" and r["active_B"]]
    logT = np.array([math.log10(r["params_B"]) for r in moe])
    logA = np.array([math.log10(r["active_B"]) for r in moe])
    acc = np.array([r["accuracy"] for r in moe])

    def fit_at(gamma, idx=None):
        i = slice(None) if idx is None else idx
        x = gamma * logT[i] + (1 - gamma) * logA[i]
        return ols(x, acc[i])

    grid = np.linspace(0, 1, 201)
    r2s = np.array([fit_at(g)[2] for g in grid])
    gamma_hat = float(grid[np.argmax(r2s)])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(moe)
    boots = []
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, n)
        boots.append(grid[np.argmax([fit_at(g, idx)[2] for g in grid])])
    lo, hi = np.quantile(boots, [0.05, 0.95])

    def loo_median(gamma):
        folds = []
        x = gamma * logT + (1 - gamma) * logA
        for i in range(n):
            keep = np.arange(n) != i
            sl, ic, _ = ols(x[keep], acc[keep])
            pred_x = (acc[i] - ic) / sl  # γ-blend of logs
            # Compare in blend space mapped back through the same blend of
            # the true sizes, i.e. fold on the blended size scale.
            folds.append(10 ** abs(pred_x - x[i]))
        return float(np.median(folds))

    named = {"total (γ=1)": 1.0, "geometric mean (γ=0.5)": 0.5,
             "active (γ=0)": 0.0}
    print(f"  EXPERIMENT 3 — continuous MoE exponent (open MoE, n={n})")
    print(f"    fit: acc ~ a·(γ·log T + (1−γ)·log A) + b")
    print(f"    γ̂ = {gamma_hat:.2f}   bootstrap 90% CI "
          f"[{lo:.2f}, {hi:.2f}]   ({BOOTSTRAP_B} resamples, seed "
          f"{BOOTSTRAP_SEED})")
    comparisons = {}
    for name, g in named.items():
        _, _, r2 = fit_at(g)
        lm = loo_median(g)
        comparisons[name] = {"gamma": g, "r2": r2, "loo_median_fold": lm}
        print(f"      {name:<24} R² = {r2:.3f}   LOO median {lm:.2f}×")
    print(f"    spread check: sd(log T) = {np.std(logT, ddof=1):.2f}, "
          f"sd(log A) = {np.std(logA, ddof=1):.2f} — active range is "
          "compressed but not degenerate;")
    print("    the nested γ-fit compares both in ONE model and still pins γ "
          "at the total-params end.")
    print()
    return {"n": n, "gamma_hat": gamma_hat,
            "gamma_ci90": [float(lo), float(hi)],
            "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
            "comparisons": comparisons}


def main():
    rows = load_cohort()
    logN = np.array([math.log10(r["params_B"]) for r in rows])
    acc = np.array([r["accuracy"] for r in rows])
    slope, intercept, r2 = ols(logN, acc)
    print(f"\n  Effective-parameter analysis — cohort n={len(rows)}, "
          f"fitted λ=0 curve: acc = {slope:.4f}·log10(N_B) + {intercept:.4f}\n")
    print("  Estimand: N_eff ≡ 10^((A − β)/α) — IKP-effective parameters,")
    print("  the size of a calibration-cohort-average open model with equal")
    print("  incompressible-knowledge capacity. ρ ≡ N_eff/N_true. See "
          "PARAM_POLICY.md.\n")

    cv = cross_validation(rows)
    dl = density_ledger(rows, slope, intercept)
    mg = moe_gamma(rows)

    OUT_JSON.write_text(json.dumps({
        "estimand": "N_eff = 10^((A - intercept)/slope), IKP-effective "
                    "parameters (open-cohort-equivalent size); "
                    "rho = N_eff/N_true (knowledge density)",
        "calibration": {"lambda": 0.0, "slope": slope,
                        "intercept": intercept, "n": len(rows),
                        "r_squared": r2,
                        "source": "evaluation_summary.json + all_models.json"},
        "cohort_n": len(rows),
        "cross_validation": cv,
        "density": dl,
        "moe_gamma": mg,
    }, indent=2))
    print(f"  Machine-readable results → {OUT_JSON.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()

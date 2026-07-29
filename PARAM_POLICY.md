# IKP Parameter-Counting & Estimation Policy

This is the normative reference for what IKP estimates, how parameters are
counted on the calibration side, what every term in the pipeline means, and
how to detect when an estimate is likely biased. It exists because the three
questions raised in
[issue #5](https://github.com/19PINE-AI/ikp/issues/5) — penalty-term
justification, MoE accounting, and refusal handling — all reduce to policies
that were implicit in the code but never written down.

Every quantitative claim below is reproducible from committed data with

```bash
python scripts/19_effective_params.py     # LOFO-CV, density ledger, MoE γ-fit
python scripts/18_v2_validation.py        # scoring reproduction + refusal intervals
```

(no API keys needed; both are deterministic functions of `data/results/*`).

---

## 1. The estimand: effective parameters, not true weights

**IKP does not measure a model's true weight count.** The quantity it
measures exactly (up to fit noise) is:

> **N_eff — IKP-effective parameters**: the size of a
> calibration-cohort-average open model with equal incompressible-knowledge
> capacity.
>
> N_eff ≡ 10^((A − β) / α)
>
> where A is λ=0 IKP accuracy and (α, β) are the pooled calibration slope
> and intercept (`data/results/effective_params.json`, calibration object).

This is the same construct as the Densing Law's "effective parameter size"
([arXiv 2412.04315](https://arxiv.org/abs/2412.04315)) with IKP as the
reference task, and Clark et al.'s "Effective Parameter Count" for routed
models ([arXiv 2202.01169](https://arxiv.org/abs/2202.01169)). Two properties
make it a sound unit:

1. **The ruler does not melt.** MMLU-equivalent size inflates ~2× every 3.5
   months (the Densing Law); the paper's densing-falsification section shows
   the IKP time coefficient is ~0, i.e. the IKP density scale is
   time-stationary.
2. **The gap to true size is itself measurable.** Define
   **ρ ≡ N_eff / N_true** (*knowledge density* — deliberately contrasted
   with densing's *capability* density, which measures the compressible
   part). ρ is directly computable for every open model, and
   `scripts/19_effective_params.py` publishes the full ledger.

The causal chain from weights to an IKP score has three model-specific
distortions, and ρ is their product:

```
N_true ──(storage density)──▶ stored bits ──(corpus overlap)──▶ stored IKP-facts ──(elicitation)──▶ accuracy A
```

- *storage density* — bits of factual knowledge per parameter (~2 under
  ideal training per Allen-Zhu & Li,
  [arXiv 2404.05405](https://arxiv.org/abs/2404.05405); lower when
  undertrained, distilled, or quantized aggressively);
- *corpus overlap* — whether training data covers IKP's English/Western-
  leaning fact distribution (the small-Qwen cluster reads ρ ≈ 0.3–0.4,
  consistent with corpus mismatch rather than capacity);
- *elicitation* — refusal policy, thinking mode, prompt format (bounded by
  the v2 interval, §5).

On the 93-model open cohort, sd(log₁₀ ρ) ≈ 0.31 (1σ ≈ 2.0×). Family
labels explain 45% of the in-sample variance before adjustment and 18% by
omega-squared after accounting for the number and size of groups. A
10,000-draw label-permutation test gives p ≈ 0.058, so the family effect is
suggestive rather than established at the conventional 5% threshold.
Extremes span deepseek-v4-flash (ρ ≈ 5.4) to deepseek-v4-pro (ρ ≈ 0.11). So:

- Read an estimate of `N_eff = X` as **X open-cohort-equivalent
  parameters**, not X true weights.
- Converting N_eff to a true weight count requires a prior on ρ; absent
  vendor-specific evidence, the honest conversion band is the LOFO band
  below.

## 2. Uncertainty policy: LOO vs LOFO

Leave-one-out CV leaks family information (when qwen3-8b is held out, five
other qwen models still anchor the fit). For a **new vendor — which is
exactly the closed-model setting** — the honest band is leave-one-family-out
(all 31 families held out whole):

| Cross-validation | median fold | 90% PI | within 2× |
|---|---|---|---|
| LOO (per-model) | 1.48× | 3.13× | 72% |
| **LOFO (per-family)** | **1.56×** | **3.51×** | 70% |

`scripts/ikp_estimate.py` reports the LOFO 90% band by default. Per-family
signed biases (worst: ernie 0.24×, qwen 0.30×, smollm 2.97×) are in
`data/results/effective_params.json` and quantify the "systematic bias
across architectures" concern.

## 3. Parameter-counting conventions (calibration side, N_true)

| Convention | Policy | Rationale |
|---|---|---|
| **What counts as N** | Total weights, **including embeddings** | Matches vendor-reported sizes; keeps the axis auditable. Caveat: at sub-1B scale embeddings are 20–60% of N (gemma-3-270m ≈ 63%) and store vocabulary, not facts — a non-embedding refit is an open validation item. |
| **MoE models** | **Total** parameters, not active | Measured, not assumed: fitting acc ~ a·(γ·log T + (1−γ)·log A) + b on the 41 open MoE models gives **γ̂ = 1.00, bootstrap 90% CI [0.86, 1.00]**. Total-only beats the geometric mean (R² 0.667 vs 0.588; LOO 1.58× vs 1.67×) and active-only (R² 0.412). Theory anchor: Allen-Zhu & Li show knowledge capacity tracks total params in MoE. The nested γ-fit also answers the range-restriction objection: sd(log T)=0.51 vs sd(log A)=0.40 — compressed but not degenerate, and both live in one model. |
| **Shared experts** | Counted once (they are ordinary weights) | Part of total weights. |
| **MTP / speculative heads** | Excluded | Inference accelerators, not knowledge storage. |
| **Vision / audio towers** | Excluded where separable | IKP probes are text-only; multimodal towers store no probeable text facts. |
| **Quantization** | Same N as the fp16 release (a *serving variant*, not a smaller model) | Weights are the same; a sanity check (§6) is that an int4/fp8 serving should return the same N_eff. |
| **Thinking mode** | A covariate (separate roster entry), never a size change | Same weights; elicitation differs. |

## 4. Scoring terms: meaning, rationale, defaults

| Term | Meaning | Default | Rationale |
|---|---|---|---|
| **A (accuracy)** | Mean of the 7 per-tier accuracies (fixed 7-tier average; an empty tier scores 0) | — | Equal tier weighting keeps the score sensitive across the whole size range instead of being dominated by easy tiers. |
| **λ (hallucination penalty)** | Score assigned to a WRONG verdict; probes score {+1, +0.5 weak-correct, 0 refusal, λ wrong} | **λ = 0 (no penalty)** | See below. |
| **α, β (slope, intercept)** | Pooled fit acc = α·log₁₀(N_B) + β on the open cohort (n=93; α ≈ 0.159, β ≈ 0.242, R² ≈ 0.91) | loaded from `effective_params.json` | Single source of truth shared by both estimator CLIs and the issue-5 analysis. |
| **γ (MoE exponent)** | Weight on log-total vs log-active in the MoE size axis | **γ = 1 (total)** | Fitted, with CI — see §3. |
| **Tier flooring** | Whether negative per-tier scores are clipped at 0 | none | Moot at λ = 0; the λ×flooring ablation is in the paper appendix. |

**Why λ = 0 is the only defensible default (not just the simplest).**
Capacity and honesty are different latent axes. P(correct) is a monotone
statistic of stored knowledge alone; P(wrong | not-known) is a post-training
*policy* — the paper's hallucination-signature section shows it varies from
3% to 99% **by vendor at fixed capacity**. Any λ ≠ 0 therefore mixes a
vendor-signature variable into the capacity estimate with vendor-dependent
sign; the λ ablation observes exactly that (estimates swing 2–3× by vendor
while R² stays flat, so goodness-of-fit cannot pick λ). λ = 0 is the only
scoring under which the estimand stays on one axis. Honesty is reported as
its own second axis (the hallucination signature), never folded into size.

Historical note: early versions of the toolkit scored λ = −1 ("penalized
accuracy"). That is retained only as an ablation
(`scripts/lambda_sensitivity.py`, `scripts/lambda_floor_ablation.py`); all
shipped estimates use λ = 0.

## 5. Refusal policy: interval, not model, not filter

Refusals are **not** filtered out, **not** counted as wrong-with-penalty, and
**not** behaviorally modeled. They are propagated into an interval by the
default estimator and IKP v2 (`scripts/ikp_estimate.py`,
`scripts/ikp_estimate_v2.py`, `IKP_V2.md`):

- **Floor** — refusals score 0 (a refusing model proves nothing about
  stored knowledge; this bound is sandbag-proof: refusing more can only
  lower it).
- **Refusal-adjusted reference** — accuracy over attempted probes bounds
  the case where every refusal hid a known fact.
- **Confidence tier** — Reliable (<10% refusals) / Caution (<30%) /
  Low-confidence (≥30%); the interval width grows with refusal rate by
  construction (validated in `scripts/18_v2_validation.py`, CHECK 3).

In N_eff terms, the v2 interval is the bound on the *elicitation* factor of
ρ. The point estimate quoted anywhere is the **floor** — conservative and
strategy-proof.

## 6. Bias-detection checklist (when is an estimate suspect?)

Reproducible checks, roughly in cost order:

1. **Refusal tier** (free, per estimate): if the v2 confidence tier is
   Caution or worse, quote the interval, not the point.
2. **Density-outlier context** (free): compare the target's vendor/family
   against the ledger in `data/results/effective_params.json`; families
   flagged `outlier_2sigma` or with LOFO bias beyond ~3× (ernie, qwen small
   models, smollm) get a directional caveat.
3. **Tier-profile shape** (free, per estimate): T6/T7 accuracy far above
   what the T3–T5-implied size predicts is a contamination signature;
   suppressed easy tiers with intact hard tiers is a sandbagging signature
   (thresholds in `ADVERSARIAL_IKP.md`).
4. **Split-delta** (free, per estimate): the public/private probe-split
   accuracy gap (v2) is a standard contamination indicator — the private
   half is not in any training set.
5. **Quantization invariance** (~$1): score an int4/fp8 serving of the same
   checkpoint; same weights must give the same N_eff within the LOO band.
6. **Recalibration drift** (cheap, after roster changes): rerun
   `scripts/19_effective_params.py` and diff `effective_params.json`; a
   jump in sd(log₁₀ ρ) or in the LOFO PI means the cohort got less
   homogeneous and the quoted bands must widen.

## 7. Open validation items

- **Non-embedding refit** (Kaplan convention) to straighten the sub-1B end.
- **Deprobing**: paired prompts on open models to measure how much
  refused-but-known capacity is recoverable (paper Open Question #3).
- **IRT/Rasch modeling** of the full response matrix (per-probe difficulty,
  per-model ability with SE, misfit as a bias flag, test equating across
  probe-set versions).
- **Fusion with inference economics**: price/throughput constrains *active*
  params; IKP (γ=1) constrains *total-equivalent* — orthogonal instruments
  that jointly pin (active, total) for closed MoE models.

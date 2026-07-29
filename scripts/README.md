# scripts/

## Public entrypoints — use these

| Script | Purpose |
|---|---|
| `ikp_estimate.py` | Score one model against the 1,400-probe set and print an estimated parameter count. See `../TOOLKIT.md`. |
| `ikp_estimate_v2.py` | Gaming-resistant estimate: refusal-robust interval + held-out `--split`. See `../IKP_V2.md`. |
| `ikp_budget.py` | Estimate the $ cost of a run before spending a token (no API key). See `../TOOLKIT.md`. |
| `make_probe_split.py` | Generate the deterministic public/private probe split used by v2. |
| `run_all_models.py` | Batch-score every model in `configs/all_models.json` (idempotent, resumable). |
| `run_evaluation.py` | Score a single model by name; called from `run_all_models.py` but runnable standalone. |

## Dataset pipeline (numbered, reproducible)

Run in order if you want to regenerate the probe set and re-score every
model from scratch. Most users never need to run these — the data
already ships in `data/`.

| Stage | Script | Output |
|---|---|---|
| Probe gen (T1–T4) | `01_generate_probes.py`, `01b_generate_t6_t7.py` | Tier candidates |
| Calibration eval | `02_run_calibration.py` | Per-calibration-model raw answers |
| Calibration fit | `03_fit_calibration.py` | `data/calibration/calibration_fit.json` |
| Target eval | `04_run_targets.py` | Per-target-model raw answers |
| Chinese probes | `08_chinese_probes.py` | Chinese-language subset |
| Researcher probes | `09_researcher_probes.py` | T4–T7 researcher subset |
| Web-grounded | `10_web_grounded_probes.py` | Web-frequency grounded probes |
| Corpus-grounded | `11_corpus_grounded_t5t7.py` | T5–T7 corpus-grounded probes |
| Fingerprint probes | `12_fingerprint_probes.py` | Extended-phrasing subset for §D |
| Distillation det. | `13_distillation_detection.py` | Early fingerprint analysis |
| Fingerprinting | `14_comprehensive_fingerprinting.py` | Fig 9 + `results/tables/fp_*.tex` |
| Densing analysis | `15_densing_law_analysis.py` | `data/densing_analysis_data.csv` + appendix stats |
| Dataset assembly | `assemble_final_dataset.py` | `data/probes/final_probe_set_v8.json` |
| Metadata | `add_release_dates.py` | Augments `configs/all_models.json` |

## Analysis / post-hoc

| Script | Purpose |
|---|---|
| `analyze_results.py` | Aggregates per-tier accuracy, builds `data/results/evaluation_summary.json`, fits scaling curves. |
| `loo_cv_analysis.py` | Leave-one-out CV + writes Fig 7 (`paper/figures/fig7_loo_validation.pdf`). |
| `17_adversarial_robustness.py` | Quantifies how cheaply a black-box operator can game its IKP estimate (sandbagging vs. contamination) across the full roster; writes `data/results/adversarial_ikp.json` + `paper/figures/adversarial_ikp.png`. Findings in `../ADVERSARIAL_IKP.md`. |
| `18_v2_validation.py` | Validates IKP v2: reproduces the repo's own accuracy (0 diff), refits the paper's calibration, checks the refusal-interval behavior. Writes `data/results/ikp_v2_validation.json` + `paper/figures/ikp_v2_intervals.png`. |
| `19_effective_params.py` | Issue #5 accuracy/interpretability harness: leave-one-family-out CV (the honest band for new vendors), per-model knowledge-density ledger ρ = N_eff/N_true, and the continuous MoE exponent γ-fit with bootstrap CI. Deterministic, no API calls. Writes `data/results/effective_params.json`. Policy in `../PARAM_POLICY.md`. |
| `show_progress.py` | Quick text progress dump across `data/results/`. |

## legacy/

Older one-off scripts from earlier dataset iterations, batch retries,
and superseded analyses. None are required to reproduce any figure or
table in the paper; they are kept for audit/reference. `PROJECT_ROOT`
in every file under `legacy/` is written as `.parent.parent.parent`
(three `..`), so the scripts still find `data/`, `configs/`, `src/`,
etc. after the move.

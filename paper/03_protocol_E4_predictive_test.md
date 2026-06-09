# E4 — The Predictive Law (Centerpiece)

## The headline claim

**For a `(model M, hint type h)` cell, the pre-training access score `P(M, h)` predicts post-VFT verbalization gain `ΔS(M, h)`; the residual is explained by confabulation rate `ConfRate(M, h)`.** Formally:

> `ΔS = β₀ + β₁ P + β₂ (P × τ_reasoning) + ε`,  with `corr(P, ΔS) > 0` on held-out cells (Spearman, permutation p < .01), and `AUC[P → 𝟙(ΔS ≥ θ)] ≥ 0.75` for the binarized recovery decision.

If the law holds, it resolves the standing contradiction that generic faithfulness fine-tuning fails to generalize (2411.15382, 2406.10625) while VFT succeeds (2506.22777) — those experiments occupied opposite ends of the `P` distribution.

## Cells and held-out structure

- **Population:** every `(M, h)` cell meeting E1's yield threshold (≥200 INFLUENCED and ≥200 RESISTANT, robust-filtered).
- **Stratification:** at least 6 models × 6 hint types = 36 cells, target ≥ 60 after task-stratified splits.
- **Hold-out scheme (pre-registered):**
  - **Held-out cells (primary):** uniformly hold out 30% of cells; train predictive law on 70%, evaluate Spearman / Pearson / AUC on the held 30%. 5 random seeds, report mean ± std.
  - **Held-out models:** leave-one-model-out — does the law generalize to a model family unseen during VFT-recovery measurement?
  - **Held-out hint types:** leave-one-hint-out — feeds directly into E6 (generalization).

## VFT procedure (the intervention measured per cell)

For each cell in the *training* portion of the held-out split, run VFT exactly as in 2506.22777, but on **only the items in that cell**:

1. From INFLUENCED-robust items in the cell's *training* fold, generate VFT supervised examples via `aacot.vft_data.build_vft_examples` (acknowledgment templates per hint type).
2. LoRA fine-tune (r=16, α=32, target attention + MLP, lr 1e-4, 2 epochs) on the cell's training items. We do *not* mix cells — the per-cell isolation is what makes `ΔS` a per-cell quantity.
3. Re-decode the cell's *test* INFLUENCED items with the VFT'd LoRA. Compute:
   - **`S_pre`**: spontaneous-mention rate before VFT (from E1's `prompt_hint` runs).
   - **`S_post`**: spontaneous-mention rate after VFT.
   - **`ΔS = S_post − S_pre`**: the gain we are trying to predict.

S is judged by the keyword + LLM-judge majority vote rubric (`aacot.verbalize.MajorityVoteJudge`), with a held-out hand-labeled validation set sized to estimate judge precision/recall to ±0.03 (≈ 1,500 items total).

## Confabulation rate (from E3)

For the same cell-`test` items where `S_post = True`, query E3's ablation oracle:

```
ConfRate(M, h) = Pr[ ¬ Δ_ablate(item) | S_post(item) = True ]
```

Pre-registered prediction: `ConfRate` is *negatively* correlated with `P` (Spearman ≤ −0.4) — access-limited cells produce confabulation, not genuine articulation.

## The three falsifiable claims

| # | Claim | Headline statistic | Falsification |
|---|---|---|---|
| **C1** | `P` predicts `ΔS`. | Held-out Spearman ρ ≥ 0.5; permutation p < .01 (n ≥ 2000). | Held-out ρ near 0 or sign flipped. |
| **C2** | `P` predicts recovery direction. | AUC[P → 𝟙(ΔS ≥ θ)] ≥ 0.75 for θ chosen as the median of training-cell `ΔS`. | AUC ≤ 0.6. |
| **C3** | `P` predicts confabulation. | corr(P, ConfRate) ≤ −0.4. | corr ≈ 0 or positive. |

The paper's empirical claim survives if C1 *or* C3 holds robustly across the three hold-out schemes; both is the strong outcome; neither is publishable as a sharply-negative result with concrete implications for the VFT line.

## Statistics

- Spearman and Pearson reported (Spearman primary — non-parametric, monotone-only assumption).
- Permutation p-value via label-shuffle (n=2000), stratified within model to control for model-level variance.
- Bootstrap 95% CIs on every reported correlation (1000 resamples over cells).
- Multiple-comparison correction: Holm-Bonferroni across the three hold-out schemes (3 tests per claim).
- Power: 60 cells with true Spearman ρ = 0.5 gives ~99% power at α = 0.01 (Steiger, two-sided). Floor of 36 cells gives ~85% — acceptable for the primary held-cells analysis.

## Cell-level covariates (reported but not in the headline)

These enter as secondary regressors to map heterogeneity, not to "rescue" a null:

- `τ_reasoning ∈ {0, 1}` — instruct vs reasoning-trained model.
- `scale` — parameter count (log-transformed).
- `α₅₀` from E3 — relative steerability of the access direction.
- `task` — MMLU/GPQA/BBH dummy.

A pre-registered ablation of the headline regression with each covariate added one at a time tells us *whether `P`'s predictive power is mediated by something else*. If `τ_reasoning` absorbs most of the effect, the paper's framing shifts from "access predicts articulation" to "the predictive signal is dominantly between-architecture; within-architecture, access is uninformative." Either is publishable.

## Computational budget (for the cluster)

| Quantity | Per cell | × 36 cells |
|---|---|---|
| Generate `nohint` + `hint` greedy + 5 samples per item, ~5k items/cell | ~3.5h on 4×H100 for 7B; ~10h for 32B | scaling linearly |
| VFT LoRA fine-tune | ~30 min/cell | ~18 h sequential, parallelizable across 4 GPUs |
| E3 ablation + injection pass on test fold | ~1h | ~36 h |

Order-of-magnitude: ~2000 H100-hours for the full matrix at 7B–14B; ~5000 with one 32B model. Well within an 8-month cluster allocation.

## What E4 produces (the headline figure + the paper's table)

- **Headline scatter:** `P` (x) vs `ΔS` (y), 36+ points, color-coded by `τ_reasoning`, error bars from CV on `P`. The regression line with confidence band; held-out points distinguished.
- **Confabulation panel:** `P` vs `ConfRate` on the same axes.
- **Table 1:** held-out Spearman / AUC / confabulation correlation across the three hold-out schemes, with bootstrap CIs and corrected p-values.
- **JSONL artifact** (`cells.jsonl`) consumed by `scripts/predictive_test.py`, regenerated under each hold-out scheme.

## Failure-mode planning

- **If `P` weakly predicts `ΔS`** but `ConfRate` strongly correlates with `P`: lead with the confabulation-detection contribution; reframe headline as "access predicts whether VFT's gains are real, even when it doesn't predict gain magnitude."
- **If `ΔS` is uniformly high**: the cell selection saturated; widen hint difficulty and re-run on a low-`P`-enriched subsample.
- **If `ΔS` is uniformly low**: VFT didn't transfer at this scale; report negative result on VFT generalization (publishable on its own) and pivot E5 to localize where readout-only training fails.

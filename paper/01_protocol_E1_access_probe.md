# E1 — Counterfactual-Influence Access Probe

## Goal

For every `(model M, hint type h, item q)` triple, produce an **access score** quantifying how strongly the model's hidden states represent the *counterfactual influence* of the hint on its answer. This score is the predictor variable for E4's predictive law and the population on which E3 will run causal validation.

## What "access" means here

A probe trained INFLUENCED-vs-RESISTANT decodes whether the hint *changed the answer to the target*, not whether the hint is *lexically present* — hint presence is held constant across both classes (§ Dataset Spec, §1). High decodability is therefore a *lower bound* on the model's internal representation of the counterfactual influence; we do **not** claim it implies the model could verbalize, monitor, or use that representation.

## Inputs

- A stimulus set built via `aacot.build_stimuli` with the primary (wrong-target) arm. The random-target arm runs as a robustness control (§ E1.6).
- A list of models `M ∈ 𝓜` (see § Models).
- Per `(M, stim)`: greedy decodes for `prompt_nohint` and `prompt_hint`, the N=5 sampled hint continuations (for the robustness filter), and captured activations on both runs.

## Procedure

### E1.1 Cell labeling (denoised)

Run `compute_influence(a_nohint, a_hint_greedy, target, samples, k_of_n=0.8)` from `aacot.cells`. Keep only items with `robust_influence ∈ {True, False}` (i.e., flips that survive the N=5 / ≥4 sample filter) so the INFLUENCED class is not polluted by stochastic pseudo-flips. RESISTANT requires `a_hint == a_nohint ≠ target` on greedy *and* `flip_prob ≤ 1 − k_of_n` (symmetric robustness). Items not surviving filters are excluded from probe training and reported as a per-cell yield metric.

### E1.2 Activation capture

Capture residual-stream vectors at the five positions defined in `aacot.activations.POSITIONS` (`prompt_end`, `cot_25`, `cot_50`, `cot_75`, `last`) for *both* runs of every retained item. Two probe-input variants are evaluated independently:

- **`hint`**: raw `prompt_hint` activations.
- **`diff`**: matched `hint − nohint` activations — the hint's representational footprint, with question-level structure subtracted out. Pre-registered prediction: `diff` reaches comparable AUC at lower layers than `hint` because it strips shared content.

### E1.3 Probe training and the AUC sweep

For each cell, position `p`, and layer `ℓ`:

1. Standardize features (`StandardScaler`).
2. Fit a logistic probe with balanced class weights (`aacot.probes.evaluate_probe`, `cfg.kind="logreg"`). Stratified 5-fold CV. Report mean ± std AUC.
3. Replicate with the difference-of-means probe (`cfg.kind="diffmeans"`); we use logreg as primary, diffmeans as the steering direction for E3.

### E1.4 Controls (anti-circularity)

Two mandatory controls per cell:

- **Shuffled-label floor.** `n_perm = 20` label permutations recompute CV-AUC; report mean + std. The headline metric is `above_chance = (auc - shuffled_auc) / shuffled_std` (σ above floor).
- **Hint-presence probe.** Train a probe on the same activations with labels = {with-hint, no-hint}. Expected AUC ≈ 1.0 — trivially high because hint tokens are in context. This is *not a baseline to beat*; it is a positive control verifying the activation pipeline isn't degenerate, while simultaneously demonstrating that the INFLUENCED-vs-RESISTANT probe solves a strictly harder problem (both classes have the hint).
- **Placebo probe.** Train INFLUENCED-vs-RESISTANT using `placebo_prompt` activations *as if they were `prompt_hint`*. Should give chance AUC — confirms the signal in the real probe is hint-specific, not insertion-generic.

### E1.5 Aggregation: the access score `P`

Per cell:

```
P(M, h) = max_{ℓ, p} CV-AUC_{logreg}(ℓ, p)   over diff features by default
σ(M, h) = above_chance at that argmax
(ℓ*, p*)(M, h) = argmax  -- recorded for E3 and E5
```

Pre-register that we report **both** the per-cell best AUC and a *fixed-(ℓ, p)* version that uses the population-level best layer (mid-network) for cross-cell comparability — to defuse the "you cherry-picked the layer per cell" attack.

### E1.6 Robustness arms (cross-validation of the construct)

- **Random-target arm.** Re-run E1.1–E1.5 with `random_target_arm=True`. The `P` distribution should be similar (decouples influence from error-induction).
- **Hint-position ablation.** Re-run with `HintPosition ∈ {PRE_QUESTION, POST_QUESTION, SYSTEM}`. Expect direction-of-effect preservation; quantitative differences inform downstream framing.
- **Sample budget.** Sub-sample n to 256, 512, 1024 per cell to estimate variance of `P` under realistic data budgets.

## Statistics and power

- Minimum per cell: ≥ 200 INFLUENCED and ≥ 200 RESISTANT after robustness filtering. Cells below threshold are reported but excluded from E4.
- Cells per model: 6 (hint types) × ≥3 tasks (MMLU-Redux / GPQA / BBH) = ≥ 18 logical groupings; we report at the (model × hint) granularity and use task as a within-cell covariate.
- Total cells for E4: ≥ 36 (6 models × 6 hints) with target ≥ 60 after stratification by task.

## What E1 produces (the artifact handed to E3/E4/E5)

For every cell:

- `P(M, h)`, its σ-margin, and `(ℓ*, p*)`.
- The cached probe (model + scaler) and the diff-of-means direction `w*(M, h, ℓ*, p*)`.
- Per-item INFLUENCED/RESISTANT assignments and activation tensors keyed by `stim_id`.

These are the exact inputs `aacot.steering.direction_from_acts` (E3) and `scripts/predictive_test.py` (E4) consume.

## Threats addressed

| Reviewer attack | Defense built into E1 |
|---|---|
| "Probe just detects hint tokens." | Held constant across INFLUENCED/RESISTANT; placebo probe must give chance AUC. |
| "Flips are stochastic noise." | N=5 robustness filter on both classes. |
| "You cherry-picked the layer." | Report both per-cell best and population-best fixed (ℓ, p). |
| "Influence ≠ error induction." | Random-target arm reports same `P` shape. |
| "Linear decodability ≠ causal use." | Acknowledged explicitly; load-bearing access claim is E3 (steering causality), not E1 alone. |

# E3 — Causal Validation: Steering / Ablation as Confabulation Ground Truth

## Goal

Turn E1's correlational `P` into a *causal* statement, and — critically — produce the **confabulation ground truth** that E4 uses to distinguish genuine articulation recovery from a model that has been trained to claim influence it does not in fact possess.

A high access score `P` is, by itself, only evidence that *something separable* lives in the residual stream. E3 asks: does that direction *cause* the behavior, and can we use it as an oracle for whether a post-VFT model's stated reason is genuine?

## Two distinct uses (both needed)

| Use | Question | Where it goes |
|---|---|---|
| **Causality of the access representation** | Does ablating / injecting the probe direction change behavior in the predicted direction? | Validates `P` as access, not artifact. |
| **Confabulation oracle** | For a post-VFT model that *says* it was influenced by the hint, would ablating the influence direction actually remove the behavioral effect? | The ground truth in E4 — separates real articulation gain from confabulated acknowledgment. |

## Inputs from E1

Per cell `(M, h)`: `(ℓ*, p*)`, the diff-of-means direction `w` (from `direction_from_acts(...)`), the per-item activations, and the residual-stream norm scale.

## E3.1 Direction construction and norm-normalization

`w(M, h, ℓ*) = mean(X_INFLUENCED) − mean(X_RESISTANT)` at layer `ℓ*`, computed on the **probe-training fold only** (no leakage into causal evaluation). Steering strength is reported as a unitless `α`, internally scaled so that the added vector has norm `α × ⟨‖h‖⟩` (`aacot.steering.normalized_alpha`). All cross-cell, cross-model comparisons use this normalized scale.

## E3.2 Ablation (the primary causal test)

For every retained INFLUENCED item:

1. Re-run the model on `prompt_hint` with a forward hook (`attach_steering`) that *projects out* the `w` direction at layer `ℓ*` for all positions ≥ prompt-end. (Equivalent to setting the component along `w` to zero; implementation: subtract `(h · ŵ) ŵ` from the residual stream.)
2. Re-extract the answer. Define `Δ_ablate = 𝟙[a_ablated ≠ target]` — flip-away rate.
3. Headline: across INFLUENCED items in the cell, what fraction *lose* the hint effect under ablation?

Prediction (access-real): `mean(Δ_ablate) > 0.5` in cells where `P` is high; near 0 in low-`P` cells. The correlation `corr(P_cell, mean(Δ_ablate)_cell)` is itself reported and contributes to the access claim independently of E4.

**Ablation control: a random orthogonal direction `w⊥`** (same norm, projected out of `w`'s span) must produce significantly smaller `Δ`. Without this, "any large perturbation flips things" is the obvious null.

## E3.3 Injection (the inverse test)

For every retained RESISTANT item (hint present, answer unchanged), add `+α w` at `ℓ*` and sweep `α ∈ {0.5, 1.0, 1.5, 2.0, 3.0}`.

- `Δ_inject(α) = 𝟙[a_steered = target]` — flip-toward rate.
- Report `α₅₀`: smallest α with `Δ_inject ≥ 0.5`.
- Control: matched-norm random direction must give near-chance `Δ_inject`.

Together, ablation and injection are *bidirectional* causal evidence for the same direction — a stronger claim than either alone.

## E3.4 Reasoning-model caveat (and what to do about it)

Pre-CoT Probes (2603.01437) reports that steering *fails* on reasoning models even when their answers are decodable. We reproduce this on at least one R1-distill model. **We do not treat the failure as a flaw of our access claim**; we treat it as a known phenomenon (the "representation-behavior gap"). Concrete mitigations:

- Report `Δ_ablate` and `α₅₀` separately for instruct vs reasoning models.
- For reasoning models where steering effect is small, fall back to **per-token sustained steering across the full CoT generation** (apply the hook to every generated token, not just prompt-end), and report whether this recovers a behavioral signal.
- If `Δ_ablate` is small but the *correlation* `corr(P_cell, Δ_ablate)` persists, that is itself a publishable finding: the representation-behavior gap is graded, not binary, and `P` predicts it.

## E3.5 The confabulation oracle (feeds E4)

After VFT (run in E4): for each INFLUENCED item where the **post-VFT** CoT acknowledges the hint (`S_post = True`), test whether that acknowledgment is *causally grounded*:

```
confabulated(item) := S_post(item) ∧ ¬ Δ_ablate(item)
```

Read: the model *says* it was influenced, but ablating the influence direction does *not* change its behavior — the stated reason is dissociated from the causal driver. Then per cell:

```
ConfRate(M, h) = |{confabulated items}| / |{S_post=True items}|
```

`ConfRate` is E4's third axis. Predicted by `P`: low pre-training `P` (access-limited cells) → high `ConfRate` after VFT, because training a missing representation only teaches the *form* of acknowledgment, not its content.

## E3.6 Pre-registered specifications (defends against multiple-comparison attacks)

- Steering applied **only** at `ℓ*` from E1 — chosen on training folds, never re-tuned on E3 outcomes.
- α-sweep grid fixed in advance.
- One primary endpoint per cell per direction (ablation flip-rate, injection α₅₀, confabulation rate). Secondary endpoints reported but explicitly marked.

## What E3 produces

- Per cell: `(meanΔ_ablate, α₅₀, ConfRate)`.
- Per item: a binary causal-grounding flag used by E4 and E5.
- The validated diction `w(M, h, ℓ*)` — re-used for E5's mechanism check (pre/post-VFT probe invariance is computed *along this direction*).

## Threats addressed

| Reviewer attack | Defense |
|---|---|
| "Steering is unreliable / direction-specific." | Bidirectional (ablate + inject), matched-norm random-direction control, pre-registered α grid. |
| "Reasoning models don't steer." | Explicitly modeled; reported separately; correlational fallback. |
| "Confabulation detection is post-hoc rationalization." | Pre-registered: confabulation only flagged when S_post and ¬Δ_ablate co-occur on the *same item*. |
| "Ablation just degrades the model." | Random-direction control must reproduce any general-degradation effect; the *direction-specific* gap is what counts. |

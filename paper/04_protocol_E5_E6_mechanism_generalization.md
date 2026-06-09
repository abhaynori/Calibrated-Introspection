# E5 / E6 — Mechanism and Generalization

E4 establishes *that* `P` predicts recovery; E5 explains *how* VFT works when it works (and why it confabulates when it doesn't); E6 establishes *how far* the recovered articulation transfers. Together they convert the predictive law into a mechanistic theory.

---

## E5 — Mechanism: VFT moves the readout, not the representation

### The falsifiable claim

When access pre-exists (`P` high), VFT changes the *readout* from the existing access representation while leaving the representation itself approximately invariant. When access is absent (`P` low), VFT either fails to find a readout (low `ΔS`) or installs a new readout disconnected from the true causal driver — i.e., confabulation, detected via E3.

Operationally:

```
Δw(M, h) = ‖ w_post − w_pre ‖ / ‖ w_pre ‖
cos(M, h) = cos(w_post, w_pre)
```

where `w_pre` and `w_post` are diff-of-means access directions from E1 / re-run E1 on the post-VFT model.

**Predicted, per cell:** in articulation-limited cells (`P` high), `cos(w_post, w_pre) ≥ 0.85` and `Δw ≤ 0.25`; in access-limited cells (`P` low but VFT applied), the post-VFT direction is either unstable (`cos` ~ 0) or shifts substantially (`Δw` large), with the latter accompanied by high confabulation.

### Procedure

1. **Pre-state.** From E1: per cell, the access direction `w_pre(M, h, ℓ*, p*)`, and per-layer/per-position AUC table.
2. **Post-state.** After E4's VFT, re-extract activations on the *same* test items using the LoRA-fine-tuned model, at the same `(ℓ*, p*)`. Recompute the diff-of-means direction `w_post` on the post-VFT activations.
3. **Invariance metrics.** Compute `Δw`, `cos(w_post, w_pre)`, and the *cosine ascent through layers*: does cos remain high across layers, or only at `ℓ*`?
4. **Layer-wise localization (the "where does VFT change things" plot).** For each layer ℓ, compute the L2 norm of activation deltas `‖h_post(ℓ) − h_pre(ℓ)‖` averaged over INFLUENCED items. Predicted shape in articulation-limited cells: a *late-layer spike* (the readout) with mid-layer flatness (the representation untouched). Access-limited cells: a diffuse shift with no late-layer peak.
5. **LoRA-weight inspection.** Project the LoRA delta `ΔW` (rank-r decomposition) onto the residual-stream basis at each layer; report the layer at which projection energy peaks. Same prediction as (4).

### Causal cross-check (re-uses E3)

After computing `w_post`, re-run E3's ablation on the post-VFT model, ablating `w_pre` (the *original* direction). If, in articulation-limited cells, ablating `w_pre` *still* removes the behavioral effect (and removes the post-VFT verbalization), this is decisive evidence that the pre-existing representation remained the causal driver and VFT only added a readout to surface it. If ablating `w_pre` no longer affects behavior, the representation was overwritten — a different, also-publishable mechanism.

### Sanity controls

- **Capability check.** Per cell, post-VFT accuracy on a held-out *non-hinted* benchmark slice must not drop more than 1 percentage point. If it does, VFT was harmful regardless of articulation gain, and the mechanism conclusions need a caveat.
- **Random-direction null.** Repeat invariance metrics for a random direction at `ℓ*` (matched norm). The `cos`/`Δw` distributions for `w_pre` must be distinguishable from this null.

---

## E6 — Generalization: cross-hint transfer and the predicted boundary

### The falsifiable claim

VFT trained on a set of hint types transfers to a held-out hint type *iff that held-out type is articulation-limited in the recipient model*. Cross-hint transfer in access-limited held-out cells produces confabulation, not faithful articulation. This is the mechanism-level resolution of "fine-tuning fails to generalize" (2411.15382) vs "VFT succeeds" (2506.22777): both are correct, on different cells.

### Three transfer regimes (pre-registered)

| Regime | Training cells | Held-out cell `P` | Predicted outcome |
|---|---|---|---|
| **A — Articulation→Articulation** | 5 hint types, all high-`P` | High `P` | High `ΔS`, low ConfRate. Generalizes. |
| **B — Articulation→Access-limited** | 5 hint types, all high-`P` | Low `P` | Low `ΔS` *or* high ConfRate. Fails / confabulates. |
| **C — Mixed→All** | Random 5 hint types | Any | `ΔS` predictable from held-out cell `P` alone (covariate test). |

### Procedure

1. **Hold-out by hint type.** For each `M ∈ 𝓜` and each `h_out ∈ HintType`, train VFT on all cells `(M, h ≠ h_out)` *and matched-yield subsamples*, then evaluate on `(M, h_out)`. This yields 6 × 6 = 36 transfer measurements per model.
2. **Per-transfer measurement.** Compute `ΔS(transfer)` and `ConfRate(transfer)` on `(M, h_out)` test fold.
3. **Predictive law on transfer.** Re-run E4's regression with `P(M, h_out)` as predictor and `ΔS(transfer)` as response. Predicted: same regression slope, same AUC for binarized recovery.

### Cross-model transfer (lower priority, larger claim)

VFT trained on model `M_train` applied via direction-transfer (not weight-transfer; we project the access direction `w(M_train, h)` into `M_eval`'s residual basis via Procrustes alignment on shared anchor concepts) and used as a readout addition on `M_eval`. Reported as an exploratory analysis; only ship if `P` similarly predicts cross-model transfer outcomes.

### Statistics

- Headline: paired-cell test — for each `(M, h_out)`, compare predicted `ΔS` (from in-cell `P` via E4's law) vs measured transfer `ΔS`. Wilcoxon signed-rank on residuals; report mean absolute residual.
- Power: 36 transfer cells per model × ≥3 models ≥ 100 paired comparisons — adequate for Wilcoxon.

---

## What E5 / E6 produce (paper-ready)

- **E5 figure: invariance.** Two-panel: (left) scatter of `P` (x) vs `cos(w_pre, w_post)` (y), one point per cell — predicted upward trend. (right) layer-wise activation-delta curves grouped by access-limited vs articulation-limited cells — predicted late-spike vs diffuse.
- **E5 ablation table.** Per cell: `cos`, `Δw`, late-layer-delta peak layer, post-VFT capability delta, post-VFT-direction ablation outcome.
- **E6 figure: regime separation.** Three scatter panels (A, B, C above), each with predicted-vs-actual `ΔS` overlaid.
- **E6 table.** Per `(M, h_out)`: in-cell `P`, predicted `ΔS`, observed transfer `ΔS`, ConfRate, residual.

## Reviewer attacks addressed

| Attack | Defense |
|---|---|
| "Probe-invariance is a tautology — you used the same activations." | We use *re-extracted* post-VFT activations and recompute the direction from scratch. |
| "LoRA isn't representative of full fine-tuning." | One full-fine-tune cell as a sanity check; report cos/Δw on full-FT separately. |
| "Cross-hint transfer experiments at this scale will have heterogeneous confounds." | Pre-registered regimes A/B/C explicitly model the predicted heterogeneity. |
| "Cross-model transfer via Procrustes is exotic." | Exploratory only; not in the headline claim. |

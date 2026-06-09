# Novelty Crawl — Findings & Defensible Differentiator

Crawl date: 2026-06-02. Goal: confirm the reframed contribution (a pre-training **access probe** that predicts **per-cell** post-VFT verbalization gain, validated against causal-steering as a confabulation ground truth) is unclaimed.

## Verdict

**Medium confidence the specific reframed contribution is open.** The room is denser than the May 26 pass found: at least four additional adjacent papers exist (one a December 2025 preprint on faithfulness-without-verbalization, two on faithful-CoT training, plus the established "probes predict fine-tuning" line). None operationalizes the same construct, but the related-work section must thread carefully or a knowledgeable reviewer will collapse the contribution into existing categories.

## Adjacent prior art the paper must address

| Paper | What it does | How it differs from our contribution |
|---|---|---|
| **CoT Can Be Faithful without Hint Verbalization** (2512.23032, Dec 2025) | Uses causal mediation to argue verbalization-only faithfulness metrics are too strong; proposes `faithful@k`; shows non-verbalized hints can causally mediate via the CoT. | Evaluation-methodology critique; no per-cell predictive diagnostic, no probe-based access score, no training intervention. **Closest conceptual neighbor — we cite it as motivation but our claim is constructive (predict + recover), not corrective.** |
| **Do Models Know Why They Changed Their Mind?** (2605.27773) | Knowledge-conflict setting; decomposes CoT into "decision-invariant knowledge display (~96%) + thin confidence layer." 8 models, 200 questions, behavioral only. | Different bias source (document-vs-prior conflict, not injected hints); no hidden-state probes; different decomposition. We use injected counterfactual hints and probe-based access. |
| **Faithfulness as Information Flow** (2605.24286, ICLR-26 workshop) | NLDD metric (CoT→answer causal flow); structural-intervention RL to make CoT mediate the answer. | Different causal axis: theirs is **CoT→answer**; ours is **bias→answer via internal representation**. Their training target is mediation; ours is articulation of an already-present representation. |
| **Counterfactual Simulation Training** (2602.20710) | Training intervention for CoT faithfulness using counterfactual simulations. | A faithfulness-training method, not a *predictive theory of when training works*. Likely a baseline / comparison for E5, not a competitor for the predictive law. |
| **Pre-training Indicators Predict Fine-tuning Outcomes** (2504.12491); **Predicting Fine-tuning with Probing** (2210.07352) | Probes / pre-training metrics predict downstream fine-tuning performance on general tasks. | Establishes the *general pattern*. Our specific instance — pre-training access probe predicting **VFT-style verbalization recovery on a per (model × hint-type) cell**, with confabulation as the failure mode — is a domain transfer, not a category invention. Honest framing softens the novelty claim from "new idea" to "the right instance of a known idea on a problem nobody applied it to." |
| **Mechanistic Evidence for Faithfulness Decay** (2602.11201) | Mechanistic study of faithfulness decay over CoT length. | Different axis (length, not training); cite as related mechanism. |
| **Lie to Me** (2603.22582), **VFT** (2506.22777), **Pre-CoT Probes** (2603.01437), **Anthropic** (2505.05410), **Reward-hack monitoring** (2603.04069, 2604.01476) | Re-confirmed positions from the May 26 pass. No new follow-ups directly preempting the predictive-diagnostic + confabulation-grounded angle. | (See prior memory.) |

## Where the differentiator now sits

The specific four-part claim that survives the crawl:

1. **Construct.** A *counterfactual-influence* probe (INFLUENCED-vs-RESISTANT, hint presence held constant) at the (model × hint-type) cell level. Distinct from answer-identity probes (Pre-CoT Probes), correctness probes (*Reasoning Models Know When They're Right*), and reward-hack-shortcut probes (the monitoring line).
2. **Predictive law.** Pre-training access `P` predicts post-VFT verbalization gain `ΔS` across held-out cells. Adapts the "probes predict fine-tuning" pattern (2504.12491, 2210.07352) to faithfulness training — *the specific instance is new even if the pattern is not*.
3. **Confabulation ground truth.** Causal steering/ablation on the influence direction distinguishes genuine articulation recovery from confabulated post-VFT acknowledgments. **Not in any adjacent paper.**
4. **Mechanism check.** Pre/post-VFT probe-invariance: VFT moves the readout, not the representation, *when access pre-exists* — a falsifiable mechanistic claim that explains both VFT's success and prior faithfulness fine-tuning's failure to generalize (2411.15382, 2406.10625).

## Required framing moves in the paper

- **Drop the bold version** "knows but doesn't say" — taken (Lie to Me; 2512.23032's near-claim). Lead with *"access vs articulation, and a predictive law for verbalization training."*
- **Foreground confabulation detection** as the load-bearing methodological contribution. It is the cleanest unclaimed move and the rigor floor for the predictive law.
- **Position 2512.23032 as motivation, not threat.** They argue verbalization metrics under-count faithfulness; we provide the diagnostic that says, per cell, whether the under-counting is recoverable by training. Complementary.
- **Acknowledge the "probes predict fine-tuning" lineage** (2504.12491, 2210.07352) upfront. Pretending it doesn't exist invites a hostile reviewer; citing it positions us as the right specialization.
- **Compare against 2602.20710 and 2605.24286** as baselines in E5 (different faithfulness-training methods). Show the predictive law holds across these too — turns them into validation of our diagnostic, not competitors.

## Residual risks

- A still-unreleased preprint specifically operationalizing access-vs-articulation with per-cell predictive testing would compress the contribution to a confabulation-detector + mechanism paper. Mitigation: monitor arxiv-sanity weekly under the relevant tags through submission; if scooped, pivot to leading with the confabulation-detection methodology, which remains uniquely ours.
- The "predictive law" leg may not hold empirically (the P→ΔS correlation could be weak). Mitigation: the design (§§ E1–E5) yields publishable findings under every outcome — null result becomes "access alone doesn't predict articulation recovery; here's what does."

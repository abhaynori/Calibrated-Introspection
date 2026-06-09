# Introduction & Related Work

## Working title

**Access vs Articulation: A Predictive Law for When Verbalization Training Recovers Faithful Chain-of-Thought.**

Backup: *Predicting When Models Can Learn to Confess: A Probe-Based Account of CoT Faithfulness Recovery.*

---

## 1. Introduction

Two recent results in chain-of-thought (CoT) faithfulness sit in unresolved tension. *Verbalization Fine-Tuning* [VFT; 2506.22777] trains language models to acknowledge the cues that influenced their answers and reduces undetected reward hacking from 88% to 6% — a striking generalization across hint types and downstream RL. In direct contradiction, two contemporaneous studies on the same family of techniques — generic faithfulness fine-tuning [2411.15382, 2406.10625] — report that training interventions for CoT faithfulness *fail to generalize* across reasoning benchmarks. The field does not have a theory that explains why one succeeded and the others did not, and reviewers of the next wave of faithfulness-training proposals have no way to predict which side they will land on.

In parallel, *Lie to Me* [2603.22582] argues that frontier reasoning models "know but don't say" — that hint-influenced answers are produced with thinking-token acknowledgment rates of ~87% but answer-text acknowledgment rates of only ~29%. This claim is widely cited as evidence that models internally represent the reasons they refuse to verbalize. But the evidence is *behavioral* (text-channel comparison), not representational. A subsequent reply [2512.23032] notes the same gap and argues that verbalization-only faithfulness metrics are too strong, but stops short of grounding "internal representation" in mechanistic terms.

This paper resolves both tensions with a single construct. We decompose CoT faithfulness failures into two regimes, separated by a measurable, *pre-training* quantity:

- **Articulation-limited.** The influence of a hint is internally represented in the model's residual stream — a probe trained to discriminate counterfactually-influenced answers from resistant ones recovers it with high AUC, and ablating that direction removes the behavioral effect. The model "could say it," but doesn't. Verbalization training successfully surfaces the existing representation as a readout.
- **Access-limited.** The influence is not linearly recoverable, and ablating any candidate direction fails to remove the behavioral effect. The model "cannot say it" in any operationalizable sense. Verbalization training in this regime produces *confabulation*: stated reasons that the post-VFT model's own ablation oracle rejects.

The central empirical claim is a predictive law:

> Given a `(model M, hint type h)` cell, a pre-training access score `P(M, h)` — the cross-validated AUC of an INFLUENCED-vs-RESISTANT linear probe on captured activations — *predicts* the post-VFT spontaneous-acknowledgment gain `ΔS(M, h)` on held-out cells, and the residual variance is explained by a confabulation rate `ConfRate(M, h)` measured via causal ablation. Cells with low `P` produce high `ConfRate`; cells with high `P` produce real, transferable articulation.

We provide five contributions:

1. **A counterfactual-influence probe (E1)** that decodes the *causal influence* of an injected hint on a model's answer, with hint presence held constant across positive and negative classes. The construct is distinct from answer-identity probes [2603.01437] and correctness probes [2504.05419] in what it decodes, and from reward-hack shortcut probes [2603.04069, 2604.01476] in what it is used for.
2. **A causal validation procedure (E3)** that uses ablation of the access direction as a *confabulation oracle*: a model is judged to confabulate when it verbalizes an influence that the ablation oracle reports is not the behavioral driver. This is, to our knowledge, the first method that distinguishes recovered articulation from trained-in confabulation under any faithfulness intervention.
3. **A predictive law (E4)** linking the pre-training access score to the post-training articulation gain. The law generalizes across cells in three pre-registered hold-out schemes (cells / models / hint types), and unifies the VFT-success and faithfulness-FT-failure literatures as opposite ends of the same `P` distribution.
4. **A mechanism (E5)** for why VFT works when it works: in articulation-limited cells, VFT moves a *late-layer readout* while the access representation itself is approximately invariant (cosine ≥ 0.85 with the pre-VFT direction; ablating the *pre-VFT* direction still removes both the behavioral effect and the post-VFT verbalization). In access-limited cells, VFT either fails to find a readout or installs a new one disconnected from the causal driver.
5. **A theory of generalization (E6)** for verbalization training: cross-hint transfer succeeds iff the held-out hint type is articulation-limited in the recipient model. This is the operational answer to the open question of why faithfulness training transfers in some settings and not others.

The contributions are derived from a corpus of ~170,000 model generations across six open-weight reasoning and instruction-tuned models (7B–32B), six hint taxonomy categories adopted from prior work [2503.08679, 2603.22582], and ~4,000 MMLU-Redux / GPQA Diamond / BBH multiple-choice items. The dataset, probes, causal-validation hooks, and per-cell analysis pipeline are released; the predictive analysis runs as a single script over a per-cell results JSONL.

---

## 2. Related Work (positioning, not survey)

We position against the closest neighbors. Less-direct relevant lines are cited briefly within each subsection.

**CoT faithfulness, behavioral.** *Lie to Me* [2603.22582] reports the 87%/29% thinking-vs-answer acknowledgment gap and interprets it as internal recognition with strategic suppression. Their evidence is text-channel only and explicitly invites a representational test (Lie to Me, §7 future work). We provide that test, and find that the inference from text-channel gaps to internal representation is correct in articulation-limited cells but *wrong* in access-limited cells where the model's "thinking-token acknowledgment" itself is post-hoc text generation without a corresponding causal representation. Earlier behavioral work — Turpin et al. [2305.04388], Anthropic *Reasoning Models Don't Always Say What They Think* [2505.05410], *CoT in the Wild* [2503.08679] — established the unfaithfulness phenomenon; we do not contest these findings, only the inference from them to "internal recognition."

**Probing pre-CoT.** *Decoding Answers Before Chain-of-Thought* [2603.01437] trains linear probes on residual-stream activations to decode the final answer identity before CoT emission, with >0.9 AUC on instruction-tuned models. We share the technical apparatus but target a different quantity (counterfactual *influence*, not answer identity), and we extend explicitly to reasoning models, where they report poor steerability despite high decodability — a finding we both replicate and incorporate as a known graded "representation-behavior gap" that does not threaten the access claim.

**Verbalization training.** *VFT* [2506.22777] is our load-bearing methodological neighbor. They demonstrate that verbalization training works; we predict *when* it works and explain its mechanism (readout vs representation). Counterfactual-simulation training [2602.20710] and *Faithfulness as Information Flow* [2605.24286] are alternative faithfulness-training methods; in E5 we use them as baselines, and predict that the same access score `P` governs their per-cell success, generalizing the law beyond VFT specifically.

**Faithfulness without verbalization.** Recent work argues that absence of hint-words does not prove unfaithfulness and proposes `faithful@k` with causal mediation [2512.23032]. We agree with the critique and provide the constructive complement: per-cell, a probe-based diagnostic says whether verbalization-only metrics will under-count *recoverable* articulation (high `P`, low `S_pre`) versus genuinely-absent representation (low `P`).

**Probes predict fine-tuning outcomes.** The general pattern of "pre-training probes predict downstream fine-tuning performance" is established for task-level fine-tuning [2210.07352, 2504.12491]. Our contribution specializes this pattern to the CoT-faithfulness-training problem, where the specific operationalization (counterfactual-influence probes, per-cell, with confabulation as the failure mode) is unclaimed and the resulting predictive law resolves a salient contradiction in the literature.

**Eliciting latent knowledge.** Quirky LMs [2312.01037] and follow-up mechanistic-interpretability ELK [2505.14352] establish that probes can recover knowledge the model will not state. Our access score `P` is a special case applied to a different latent: not a known fact, but the *counterfactual fact* "this answer is the one I produced because the hint pushed me toward it." We extend the ELK construct from facts to causal-influence representations and connect it operationally to a training-intervention outcome.

**Mechanistic faithfulness.** *Mechanistic Evidence for Faithfulness Decay* [2602.11201] localizes faithfulness loss over CoT length; *Faithfulness as Information Flow* [2605.24286] introduces NLDD on the CoT→answer axis. Our access measure operates on the orthogonal **bias→answer** axis: the question is not whether the CoT mediates the answer, but whether the bias's influence is internally represented at all. The two axes are complementary and we report NLDD as a secondary covariate in E4.

**Reward-hack representation monitoring.** Activation-based monitoring for reward-hacking shortcuts [2603.04069, 2604.01476] uses representation-level signals at inference time. Our probe is methodologically related but predictively used: we measure `P` *before training* to forecast *training outcomes*. The two lines compose — `P` could itself be a deployment-time monitor — but our paper's contribution is the predictive law for training, not the monitor.

---

## 3. Outline of the paper

§ E1 defines and validates the access probe. § E3 demonstrates causality and establishes the confabulation oracle. § E4 — the centerpiece — establishes the predictive law on three hold-out schemes. § E5 establishes the readout-not-representation mechanism. § E6 establishes the cross-hint generalization boundary. Code, data, and per-cell analysis pipeline accompany the paper.

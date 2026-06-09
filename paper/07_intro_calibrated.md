# Introduction & Related Work — Calibrated Introspection

Working title: **Calibrated Introspection: Do Language Models Know How Much They Were Swayed?**

This document supersedes `paper/05_intro_related_work.md` as the active intro / related-work draft. The former is preserved for the supporting access-vs-articulation methodology in the appendix.

---

## 1. Introduction

Recent work has converged on a striking but qualitative claim: language models display a form of *behavioral self-awareness* about their own biases. Models fine-tuned to exhibit hidden policies later describe those policies without prompting [2602.14777]. Frontier reasoning models acknowledge cue influence in thinking-tokens at ~87% while denying it in answer-text at ~29% [2603.22582]. Probes recover knowledge that models refuse to state [2312.01037, 2505.14352]. Each result establishes that introspective access in language models is *non-zero*.

What none of these papers measure is *how accurate* the access is. Models that report some influence at some rate are not models whose self-reports can be trusted as numbers. The CoT-monitorability and AI-safety lines that depend on self-report require a *calibration curve*: given that a model says it was influenced with confidence `c`, how often was it actually influenced? Without that curve, the field has been arguing about a qualitative phenomenon when the operationalization is unmistakably quantitative.

This paper provides that curve. We construct two scalar measurements per item:

- **`P_internal`** — a continuous, cross-validated score from a counterfactual-influence probe (INFLUENCED-vs-RESISTANT, hint presence held constant) trained on the model's own hidden states. The probe decodes "this answer was, in fact, pulled toward the target by the hint."
- **`P_reported`** — the model's own intensity-elicited score on a 0–10 scale for the same item, asked as a follow-up turn.

The headline empirical object is the joint distribution of these two scalars, summarized by the reliability diagram, ECE, signed miscalibration, slope, and a miscalibration typology. We characterize this object across six open-weight model families spanning 0.5B–32B parameters, three multi-choice benchmarks, and six hint categories adopted from the prior taxonomy [2503.08679, 2603.22582].

We make four contributions:

1. **A construct.** Introspective accuracy for bias influence is a *measurable, scalar* property — not a binary "can-they" — and we operationalize it as the calibration of `P_reported` against `P_internal` on counterfactually-influenced items.
2. **A characterization.** Across the model matrix, *which* of three a priori plausible patterns holds: systematic under-confidence in independence, a calibration improvement with scale, or a per-hint calibration ceiling.
3. **A training-dynamics result.** On open RL-checkpoint series (DeepSeek-R1-Distill at multiple depths; OLMo-2 across SFT/DPO stages), we identify *which* training transitions move the calibration curve — separating the gains from instruction-tuning from any losses introduced by outcome-RL.
4. **A mechanistic claim.** The residual `P_internal − P_reported` is itself decodable from activations at non-trivial accuracy: when models miscalibrate, the miscalibration is *internally represented*. A causal cross-check via direction-ablation closes the loop.

The infrastructure required for these measurements — the counterfactual-influence corpus, the access probe, the matched no-hint / hint / placebo construction, the elicitation prompts, and the causal hooks — was built for an earlier framing of this work (access vs articulation; see appendix). The pivot is to use that infrastructure to answer a different and, we argue, more fundamental question.

---

## 2. Why calibration is the right metric

Calibration as a measurement target has three properties that survive the next several years of follow-up work:

- **It composes with any introspection protocol.** Whether the elicitation is a 0–10 scale, a binary admission, a verbal acknowledgment in CoT, or a separate inner-speech token, the calibration of that signal against a probe-ground-truth is well-defined. Subsequent methods that improve elicitation can be measured on the same axis without re-inventing the construct.
- **It separates two questions the literature currently conflates.** *Can* a model report being influenced (rate of non-zero `P_reported`) is independent of *how accurately* it reports (the slope of `P_reported` against `P_internal`). A model can have high coverage and low calibration, or vice versa. The reliability diagram makes both visible.
- **It naturally yields a scaling law and a training-dynamics result.** Once `ECE(M, h)` is defined, asking how it scales with parameters and moves under SFT/DPO/RL is straightforward; the resulting curves are the kind of artifact other researchers cite for years.

---

## 3. Related work — positioning, not survey

**Behavioral self-awareness in LLMs.** *Emergently Misaligned LMs Show Behavioral Self-Awareness* [2602.14777] shows models can describe behaviors they were fine-tuned into. *Reasoning Models Don't Always Say What They Think* [2505.05410] and *Lie to Me* [2603.22582] document gaps between thinking-token acknowledgment and answer-text acknowledgment. *Do Models Know Why They Changed Their Mind?* [2605.27773] decomposes CoT under knowledge conflict into a decision-invariant display and a thin confidence layer. Each provides existence proofs for non-trivial introspection in some setting. *None* provides a per-item calibration metric of self-reported influence against a probe-defined ground truth, which is our object of study.

**Calibration of confidence in correctness.** *Reasoning Models Know When They're Right* [2504.05419] establishes that linear probes on hidden states yield well-calibrated correctness predictions. We take this paper's calibration-of-internal-state idea and extend it to a different latent: the *counterfactual influence* of a bias on the answer, calibrated against the model's *self-report* rather than against correctness. The technical machinery is shared; the substantive target is different.

**Persona vectors and behavior-level steering.** Per-persona steering vectors [2605.21006] show that aggregate behavioral states are linearly recoverable and steerable. Persona is a coarser representational construct than item-level counterfactual influence; calibration of *per-item* `P_reported` against `P_internal` could not be reduced to persona-level statistics, because the question is whether the model knows for *this answer* how much it was pushed.

**Probing latent knowledge / ELK.** [2312.01037, 2505.14352] show probes can read out knowledge the model will not state. Our contribution specializes ELK from facts to causal-influence representations and *measures*, rather than asserts, the gap between internal state and external report.

**Activation steering and self-monitoring.** Recent work documents that models can detect when their own activations are perturbed [2602.06941] and that steering vectors operate as cross-task abstractions [2604.02608]. We use steering in the methods section as the causal validation of the access probe and as the ablation primitive for the miscalibration subspace check in §4 of the paper; the calibration claim does not depend on steering being itself reliable.

**Faithfulness-without-verbalization** [2512.23032] argues that absence of hint-words does not prove unfaithfulness and proposes causal-mediation metrics. We agree with the critique. The calibration framing is its constructive complement: the field has been arguing about a qualitative gap when the right object is the quantitative curve relating internal state to report.

**Verbalization training.** *VFT* [2506.22777] demonstrates that supervised training can substantially increase verbalization of hint influence. From the calibration standpoint, VFT is an *intervention on the curve*: does it move the slope toward 1, shift the intercept toward 0, or merely raise coverage without improving calibration? The original protocols of this project (now in the appendix) lay out exactly the experiment to answer this; we report it as a secondary intervention result, not the headline.

---

## 4. Pre-registered predictions

Independent of which outcome the data deliver, we will report:

- The reliability diagram per `(model, hint type)`.
- ECE, signed miscalibration, slope, intercept, type — with bootstrap 95% CIs.
- The scaling regression `ECE ~ log(parameters)` within model family.
- The training-dynamics result across at least two checkpoint series.
- The mechanistic-subspace result: does activation-decoding of `P_internal − P_reported` succeed above chance, and does ablating its direction move `P_reported` toward `P_internal`?

We pre-register three a priori plausible outcome patterns, each of which we will treat as the lead result if it holds robustly: (A) systematic under-confidence in independence, (B) scale improves calibration but RL stages reverse the gain, (C) a per-hint calibration ceiling exists. If none of (A)–(C) holds, we expect the data to surface a structured pattern we can characterize; a true null — `P_internal` and `P_reported` independent — would itself be a striking finding about introspective inaccessibility and is reported as such.

---

## 5. Outline

§ 2 of the paper defines the construct precisely. § 3 describes the corpus, models, and the elicitation prompt. § 4 reports the per-cell reliability diagrams (the headline figure). § 5 reports the scaling law. § 6 reports the training-dynamics result. § 7 reports the mechanistic subspace check. The appendix contains the access-vs-articulation methodology and an intervention experiment using VFT.

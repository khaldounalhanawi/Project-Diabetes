# Evaluation of the three strategies for the diabetes project

> Refers to the strategies in `01`–`03` and the summary in `00-summary.md`: A "Screening Net", B "Data Detective", C "Cost Matrix & Dose-Response". A judgment with weights is a stipulation. That is why there are two weightings and, at the end, a check on whether the ranking flips.

## 0. Caveats about the evaluation

- **I evaluate the execution, not the idea.** I gave the agents the guiding concept, metric direction and focus of each strategy. What is evaluated is how well they carried out the assignment.
- **A shared error is partly due to my briefing.** All three took Glucose for a target leak (correction K1). The accompanying paper was not on their mandatory list, but they could have found it in the repo. The error costs each strategy exactly one point (rule in section 3), it does not shift the ranking. *In the first version this sentence was already here, but the points did not follow it. The Codex verifier found this, see section 6.*
- **There is no independent judge.** I wrote the assignments and evaluate the results. A second opinion (Codex verifier) would be the clean cross-check.

## 1. Criteria

| Criterion | Question | Weight "Submission" | Weight "Learning" |
|:--|:--|--:|--:|
| K-1 Task fit | Does the strategy cover all 9 steps of the assignment, and does the metric fit its own rationale? | 20 % | 15 % |
| K-2 Course fit | Only methods from W2D1–W2D4? Is everything beyond that marked? | 15 % | 10 % |
| K-3 Correctness | Are the statistics, formulas and domain statements right? | 20 % | 20 % |
| K-4 Pre-registration and leakage discipline | Is it fixed what is settled before looking at the data? Does the test set stay untouched? | 15 % | 15 % |
| K-5 Feasibility | Can this be done in one project day, if need be alone? | 15 % | 10 % |
| K-6 Learning value and defensibility | What does one understand better afterwards, and does it withstand a coach's follow-up question? | 15 % | 30 % |

Scale 1–5 (5 = very good). The points are in section 3, the reasoning in section 2.

## 2. Evaluation per strategy

### A – Screening Net

**Strengths**
- **Closest to the assignment.** Glucose, BMI and Age are exactly the hypotheses the assignment suggests ("based on what you know about diabetes"). The variable names match the specification (`dummy`, `logreg`, `logreg_balanced`).
- **The most valuable single finding of all three texts:** F2 of "all sick" is 5p/(4p+1). Without this baseline every model looks good at F2. No other agent said this so clearly.
- **The two imbalance levers are separated** (`logreg_thr` without weights, `logreg_balanced` without threshold). That way one can answer which lever contributes what. That would be the coach's first question.
- **The threshold selection is good craftsmanship:** grid, guardrail, "middle of the plateau instead of the edge", rounding. This reduces overfitting to the training data without needing week 3 methods.
- **The leakage trap "class-specific median" is justified twice:** impossible on the test set, a target leak in the training set. This is the best explanation of this trap in all three texts.

**Weaknesses**
- K1 three times: in the hypothesis rationale, as "circularity" among the weaknesses and as coach question 2. The rationale for `logreg_cheap` is built on it too. The variant remains sensible (screening without a blood draw), the rationale does not.
- **Low information gain from the hypotheses.** H1 (Glucose) will almost certainly be supported. Three predictable yes answers show little about whether the tests were understood. A sees this itself and points to "found along the way".
- **The worked example p = 0.06** comes from the clinic slide, not from the project, and can be misleading. In the Pima data the imbalance is much weaker.
- **Everything one-tailed.** This is justified, but then "contradicted" is only possible descriptively. A says so itself. A coach could still ask why not simply two-tailed.
- **The precision guardrail is set too low** (Codex verifier). `P_min` = 0.33 is supposed to lie, according to A, "clearly above the positive share". With a share of sick patients around 0.35, however, even "all sick" reaches 0.35 and would pass. The guardrail thus contradicts its own condition.
- **"Cross-check with the other test"** opens a second evaluation per hypothesis although m = 3 applies. It is clean only as a sensitivity check without a verdict.

**Judgment:** the safest strategy for a good submission. Technically solid, little risk, moderate learning gain.

### B – Data Detective

**Strengths**
- **The best data craftsmanship.**
  - A decision table per column, with a reasoned zero-plausibility fixed before looking at the data.
  - The median is computed **after** `0 → NaN` (an easy mistake to make).
  - The tests run on the NaN version.
  - `apply_cleaning(df, cfg)` with `assert` afterwards.
  - The test set's indicator columns come from its own zeros, which is correct because nothing is learned in the process.
- **Most consistent pre-registration discipline.** The band boundaries are fixed beforehand, merging happens only by row counts, "never by outcome shares or p". The main model is named in advance, variants serve robustness and not the search for a winner. That is exactly what the p-hacking slide from W2D2 says.
- **The only one with a schedule:** a priority order for short time. `test_size=0.25` is justified (more positive cases in the test set).
- **Honest self-criticism.** B itself names that F1 does not fit its own rationale "FN is worse", and that H1 can never establish MCAR.

**Weaknesses**
- **The metric contradicts its own rationale.** F1 weights FP and FN equally, B argues "FN is worse". Naming the contradiction is good, leaving it standing is a vulnerable core in step 1 ("choose your metric … and explain why"). Consistent would be F2.
- **The hypotheses yield little result.** H1 (missing insulin value × Outcome) has no direction and can never be "contradicted". H3 (blood pressure) is deliberately designed for "cannot tell yet". In the unfavorable case, at the end one of three hypotheses has a clear verdict. Honest in substance, but thin for the project goal.
- **H1 is a hypothesis about the measurement process, not about the patients.** That fits the assignment only halfway ("the research question states what you want to find out about the patients").
- **Self-invented thresholds** (5 % / 40 %) are correctly marked as a rule of thumb, but are not from the course.
- **The time risk is the largest:** drop/impute × plain/balanced × threshold × indicator. Hardly achievable alone in one day.
- K1 (suspected target leak) and coach question 1 with a trap (K2), which, however, works well as a practice question.

**Judgment:** the best craftsmanship and the best discipline. Inconsistent in the metric, thin on results in the hypotheses, too broad for one day. More valuable as a **toolbox** than as a whole strategy.

### C – Cost Matrix & Dose-Response

**Strengths**
- **The only strategy whose choice of metric is justified by calculation.** k → β ≈ √k → β = 2 → threshold ≈ 1/(1+k). I double-checked all formulas, they are correct. C itself warns against false precision ("2.24 would be false precision") and against equating F-beta with the cost function.
- **Honest handling of an arbitrary number.** The sensitivity k ∈ {2, 5, 10} is **reported, not used for selection**. That is methodologically more mature than anything else in the three texts.
- **Cost anchors as baselines:** "all negative" costs k·p, "all positive" costs 1−p. A model is worthwhile only below that. This corresponds to A's F2 finding, only expressed in costs.
- **Greatest learning value:** the comparison of scaled coefficients with the test verdicts, with the distinction "marginal (test) versus conditional (coefficient)" and the example of Pregnancies being "explained away" by Age. This connects W2D2 and W2D3, and confounding becomes visible in real data.
- **Hypotheses on less obvious columns** (DPF, Pregnancies) with an open outcome. The choice of test is justified with the course table, including the note on why Spearman against a 0/1 column is only a detour.
- Committing the hypotheses before `read_csv` to have a timestamp as proof. That is a good, simple idea.

**Weaknesses**
- **Furthest beyond the course so far.** The derivations are correctly marked as "beyond the course so far". Still, the risk rises of presenting something that one cannot derive freely under a coach's follow-up question.
- **Two target quantities** (report F2, minimize cost) can yield different winners. C names this but does not resolve it. The assignment wants **one** chosen metric.
- K1, plus the weakness "scenario consistency", which builds on K1.
- **More computational work:** four to five model variants, cost curves for three k values. Medium time risk.
- **H3 (Age) is almost identical to A-H3.** For a standalone strategy that is little that is new.
- **H2 promises more than χ² can show** (Codex verifier). "The more pregnancies, the higher" is tested by χ², and the direction is pulled into the "supported" verdict via "highest band > lowest". But χ² only shows that the shares differ, and the comparison of two bands does not show a continuous rise.

**Judgment:** in substance the strongest strategy and the one with the most learning material. But it requires confidence in derivations that were not taught.

## 3. Points

| Criterion | A | B | C |
|:--|:-:|:-:|:-:|
| K-1 Task fit | 5 | 4 | 4 |
| K-2 Course fit | 5 | 4 | 3 |
| K-3 Correctness | 3 | 4 | 3 |
| K-4 Pre-registration and leakage discipline | 4 | 5 | 5 |
| K-5 Feasible in one day | 5 | 2 | 3 |
| K-6 Learning value and defensibility | 3 | 4 | 5 |
| **Weighted "Submission"** | **4.15** | **3.85** | **3.80** |
| **Weighted "Learning"** | **3.85** | **3.95** | **4.05** |

**Rule for K-3** (reformulated after the Codex verifier, see section 6): Start at 5. K1 costs **each** strategy exactly one point, no matter in how many places the error appears. Every further substantive error costs one point. Imprecisions that all three share (e.g. presenting Welch and Mann-Whitney as interchangeable) or that the text itself labels as a rule of thumb cost nothing.
- A: K1 and `P_min` = 0.33 (the guardrail would let "all sick" through as soon as the share of sick patients is above 0.33) → 3.
- B: only K1 → 4. The metric contradiction now counts only under K-1, not twice.
- C: K1 and the direction criterion for χ² ("highest band > lowest" as part of a "supported" verdict) → 3. The `class_weight` statement sits in C under "rules of thumb, not identities" and therefore costs nothing.

**Does the ranking flip? Yes, depending on the weighting.** Whoever wants a safe submission is ahead with A, whoever wants to learn the most, with C. **After the correction, B is in 2nd place in both weightings** (previously 3rd place). The gaps are small (0.05 to 0.35 points), and the points are my stipulation, not a measurement. Read the table as a direction, not as a judgment down to the decimal place.

## 4. What none of the three strategies delivered

1. **Used the paper as a source.** That would have made K1, the predictive nature of `Outcome` and the reference value 0.76 apparent on their own. Lesson for the project day: read the sources in the repo first, then write the hypotheses.
2. **Addressed the selection bias.** According to the paper, only women with a normal glucose test are selected, and diagnoses in the first year are excluded. So the model says something about "currently normal" women, not about all women. That belongs in the conclusion and answers the question of whether the model can be used everywhere.
3. **Planned the conclusion.** Step 9 demands "explain what your best model gets right, and where it still makes mistakes". All three deliver the table, none sketches the three to five sentences that should end the report.
4. **Probabilities after `class_weight` are no longer risks.** Only C mentions this. Anyone who writes "the model says 40 % risk" in the conclusion is wrong after `balanced`.

## 5. Consequence for the recommendation

The recommendation in `00-summary.md` (section 5: framework A, protocol B, k sentence C) stands. The evaluation supports it: it takes the best feasibility (A), the best discipline (B) and the best metric justification (C) and avoids B's weakness in the metric.

Two additions from the evaluation:
- **First buffer task** is the coefficient comparison from C. It has the highest learning value per minute.
- **The conclusion should include** the selection bias (section 4, point 2) and the sentence that `balanced` probabilities are not risks.

In addition, from the Codex verifier: `P_min` = 0.5 instead of 0.33, H3 phrased as a non-directional χ² hypothesis, and the note that Welch and Mann-Whitney test for different kinds of difference. All three points have been incorporated into `00-summary.md`.

## 6. Cross-check by Codex (Oct 8, evening)

Codex (`codex exec`, read access only, all sources embedded: both documents, assignment text, paper excerpt, slides W2D1–W2D4) returned the verdict **PASS WITH COMMENTS**. It confirmed all formulas, the Bonferroni limit, the statements on `class_weight`, AP and ROC, and the weighted sums of the first version. I checked each finding against the file by keyword search. All ten are correct.

| Severity | Finding | Implemented |
|:--|:--|:--|
| high | Diagnosis wording in the agent texts despite K1 | central reading note in the strategy files, from `00-summary.md` |
| high | χ² on bands phrased directionally (recommendation H3, C-H2) | H3 non-directional, direction only descriptive. Added to C as a weakness, costs one point in K-3 |
| medium | `P_min` = 0.33 is below p ≈ 0.35 | `P_min` = 0.5 proposed, with a check against `p_train`. Added to A as a weakness, costs one point in K-3 |
| medium | Welch and Mann-Whitney test different null hypotheses | Section 4, point 6 of `00-summary.md` |
| medium | "Cross-check with the other test" increases the number of tests | marked as a sensitivity check without a verdict |
| medium | K1 is the paper's reading, the CSV is unchecked; 0.76 is sensitivity = specificity | K1 qualified, check of the CSV on the project day added |
| medium | `class_weight` shifts the odds only approximately by k | reading note in the strategy files; no point deduction because C itself presents it as a rule of thumb |
| medium | **This evaluation:** K1 penalized unequally, B's metric deducted twice | K-3 rule reformulated, points and ranking corrected (B now 2nd place) |
| low | K2 does not appear on the slide in these words | marked as a derivation from the slide's criterion, with the condition "list fixed beforehand" |
| low | Tests on the NaN version may hit a selected subgroup | state case counts per outcome, state the limitation in the verdict |

The coaches have the last word: the questions in section 7 of `00-summary.md` remain open.

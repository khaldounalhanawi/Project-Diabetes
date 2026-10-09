# Diabetes Project Day: Three Strategies – Summary and Recommendation

> **How this came about:** On the evening of Oct 8, three agents (Sonnet) each drafted one strategy, all with the same setup: the assignment from `01_diabetes_project.ipynb`, using only methods from the slides W2D1 to W2D4. Claude (Opus) checked the texts against the slides and the paper, corrected errors (section 1) and merged them. **Nobody opened `diabetes.csv` in the process.** So the hypotheses come before any look at the data, as the assignment requires, and no verdict has been anticipated.
>
> **How to use this:** The hypotheses here are proposals for discussion, not a finished solution. In the project we will formulate them together in our own words, and we should both be able to explain every decision.

## 1. Fact check: corrections to the agent texts

**K1 – Glucose is not target leakage (all three agents were wrong).** All three assumed that the 2-hour glucose value was "practically the diagnostic criterion" for `Outcome`. The accompanying paper (`paper_on_diabetes_mellitus_data_set.pdf`, section 2.3 "Case Selection") says otherwise:

- For each woman, **one examination with a non-diabetic glucose tolerance test** (GTT) was selected.
- `Outcome = 1` means: **diabetes was diagnosed 1 to 5 years after this examination.** `Outcome = 0` means: a GTT five or more years later was normal.
- Cases diagnosed within one year were excluded, "to remove … those cases that were potentially easier to forecast".

Consequences for project day:
- Glucose is therefore a **legitimate predictor** (known at the time, not computed from the target, per the W2D3 leakage checklist). The objections about "circularity" in A and C drop out. *Caveat (Codex verifier):* This is the paper's reading for its own 768 examinations. Whether the Kaggle CSV contains exactly these cases has not been checked yet. On project day, check after loading: 768 rows, same columns, Glucose range (see below).
- The scenario is a **prognosis, not a diagnosis:** "Who will develop diabetes in the next five years?" A false negative (FN) is therefore a **missed chance for prevention** (lifestyle, close monitoring), a false positive (FP) is an **unnecessary check-up or counselling plus worry**. The cost logic "FN is worse" stays, but it is told more honestly. This wording fits step 1 better than "confirmation test".
- Checkable as "found along the way": Because of the selection rule, the dataset should contain no Glucose value in the diabetic range (the paper gives the limit as 200 mg/dl). A look at `df_train["Glucose"].max()` shows whether the Kaggle version matches.
- **Benchmark from the paper:** The ADAP network reached sensitivity = specificity **0.76** at the crossover point on 192 test cases. That is a nice yardstick for the discussion. Note: the number comes from a different split and a different model, so it is not a direct comparison.

**K2 – Replacing zeros with NaN may happen before the split.** In control question 1, Strategy B asks what would have happened "if you had already replaced the zeros before the split". Answer, derived from the criterion on the W2D3 slide "Clean after the split" (the slide does not name `0 → NaN` literally): nothing bad, **as long as the column list and the rule are fixed beforehand**. Then `0 → NaN` learns nothing from the data ("Learns nothing: fine before the split"). If a look at all the data decided which zeros get replaced, this would no longer hold. Leakage arises from **filling in** with a median that was computed from all rows. A good trick question from the coach.

**Spot check passed:** All quoted slide titles really exist ("A test that does nothing", "When one mistake matters more", "Frequency is not cost", "metric chosen before the results are in", "Try the alternative"). The formulas check out as well:
- F2 of "all positive" is 5p/(4p+1).
- The cost-optimal threshold is 1/(1+k).
- F-beta weights an FN β² times as heavily as an FP.
- The random baseline costs (k+1)·p·(1−p) per patient.

## 2. The shared framework: what you have learned and where it is needed on project day

| Project step | Learned on | Core rule |
|:--|:--|:--|
| 1 Metric and hypotheses | W2D4, W2D2 | Choose the metric **before** the results. Justify FP/FN costs, "it depends on the situation". Hypothesis: what changes, how, through what. Decide one-tailed or two-tailed before looking at the data. |
| 3 Split | W2D3 | Split first, `stratify`, `random_state`. Explore after the split: every plot of the full data is silent leakage. |
| 4 EDA and cleaning | W2D1 | MCAR/MAR/MNAR. Median before mean when the data are skewed. IQR gives candidates, not deletion orders. "Try the alternative, and see whether the conclusion moves." |
| 4 Tests | W2D2 | Test table: two groups × number → t-test (Welch) or Mann-Whitney U. Category × category → χ². Effect size next to every p. Bonferroni for 3 tests: 0.05/3 ≈ 0.0167. "Fail to reject ≠ accept." |
| 5 Baselines | W2D3 | Random (`DummyClassifier`) and a rule with a cutoff **on the training set**. A model only counts to the extent that it beats the baseline. |
| 6–7 Model and imbalance | W2D3, W2D4 | Fit `StandardScaler` on train only. `class_weight="balanced"` changes the training, the threshold only changes the decision. Take the threshold from the train probabilities. After the class weights, choose the threshold again (coach, Oct 8). |
| 8 Evaluation | W2D4 | One train/test table per model. Confusion matrix: rows = truth, columns = prediction, 0 before 1. `.ravel()` returns TN, FP, FN, TP. |

## 3. The three strategies at a glance

| | **A – Screening Net** | **B – Data Detective** | **C – Cost Matrix & Dose-Response** |
|:--|:--|:--|:--|
| Guiding idea | Overlook no future case, if possible | First trust the measurement, then model | Put a number on the FN/FP trade-off (k) |
| Main metric | F2 with a precision floor `P_min` | F1 (F2 and AP alongside) | F2 from β ≈ √k, plus cost per patient |
| H1 / H2 / H3 | Glucose / BMI / Age | Insulin missing × Outcome / age bands / blood pressure | DPF / pregnancy bands / Age |
| Test types | 3 × Shapiro → Welch or MWU, one-tailed | 2 × χ² + Cramér's V, 1 × Welch/MWU, two-tailed | 2 × MWU (one-tailed), 1 × χ² |
| Cleaning | 0 → NaN, train median, variant without Insulin/SkinThickness | Decision rule per missing share, indicator columns, drop-versus-impute variant | brief, train median, drop as a cross-check |
| Model variants | logreg, balanced, balanced + threshold | M1–M3, V-Drop, V-Ind | unweighted, balanced, {0:1, 1:k}, threshold from cost minimum |
| Strength | Clearest story, closest to the assignment | Best data understanding, cleanest protocol | Choice of metric defensible by calculation |
| Risk | F2 flatters (see 5), few surprises | Many variants, F1 does not quite fit its own rationale | k is an assumption, most calculation effort |
| Time needed | low | high | medium to high |

Interesting: **Age appears in all three strategies**, but is tested in three different ways (MWU one-tailed, χ² on bands, MWU plus bands as a plot). Whoever understands why all three routes are legitimate and what each one costs has grasped the core of W2D2.

## 4. Where all three agree (the robust base)

1. Write the research question, hypotheses, metric and (for C) k **before `read_csv`** into a Markdown cell, and commit locally (timestamp as evidence, no push).
2. `0 → NaN` in Glucose, BloodPressure, SkinThickness, Insulin and BMI. **Pregnancies = 0 is real.** Fix the list beforehand.
3. **Hypothesis tests on the NaN version** (drop missing values per test), not on imputed data, because median filling blurs differences. The price: if measurements are missing systematically, you are testing on a selected subgroup. So state for each test how many cases per Outcome drop out, and mention the limitation in the verdict. *Check this reading of "cleaned training set" with the coach.*
4. Train medians in a dict, one function `apply_cleaning(df, cfg)` for train and test, then an `assert` for no NaN and identical columns.
5. Bonferroni α = 0.0167. Shapiro and Levene are assumption checks and do not count as tests. Additional real tests count.
6. Shapiro per group; with n in the hundreds, read it together with a histogram and `skew()`. A Shapiro p ≥ 0.05 does not prove normality, it only means that nothing clearly speaks against it. Welch if normal, otherwise Mann-Whitney U. **The two do not test the same null hypothesis** (Codex verifier): Welch tests means, Mann-Whitney tests whether values of one group tend to be larger (rank order), not simply the medians. So in the verdict say *which* difference was tested, mean or median difference, with η alongside as a description. If the other test is also computed as a "cross-check" (Strategy A), then only as an openly labelled sensitivity check without a verdict of its own, otherwise there are more than three tests.
7. Threshold from `predict_proba(X_train)`. It is slightly optimistic (in-sample). `cross_val_predict` only comes in week 3.
8. Accuracy appears in the table only as a warning light. The same model with a different threshold has **identical** AP and ROC AUC.
9. Small test set (a few dozen positive cases): do not over-interpret differences of a few hundredths.

## 5. Recommendation (sparring): a hybrid instead of a pure strategy

**A as the backbone, the protocol from B, one sentence from C.** Rationale: It is a single project day. A is closest to the assignment. B supplies the craft that the assignment explicitly demands ("keep a record of every change"). C makes β=2 justifiable instead of a gut feeling.

- **Metric:** F2 (positive = diabetes within 5 years) with a precision floor `P_min` set in advance. Rationale from C: "A missed case weighs about four times as much as an unnecessary check-up. F-beta weights an FN β² times, so β = 2."
- **Do not take over `P_min` from A** (Codex verifier): A proposes 0.33 ("at most 2 false alarms per case found"). If the share of positive cases is about 0.35, even "all positive" reaches a precision of 0.35 and would pass this guardrail. Proposal: `P_min = 0.5`, i.e. "at most one unnecessary check-up per future case found". Justify it on domain grounds, fix it before looking at the data, and after the split check that `P_min` is clearly above `p_train`.
- **Proposed hypotheses**, deliberately with two types of test so that the test table from W2D2 becomes visible:

| | Hypothesis (proposal) | Column | Test |
|:--|:--|:--|:--|
| H1 | The higher the glucose in the glucose tolerance test, the more likely a woman is to develop diabetes in the following years. | `Glucose` | Shapiro → Welch or MWU, one-tailed. You need this anyway for the rule-based baseline. |
| H2 | The stronger the family history, the more likely she is to develop diabetes. | `DiabetesPedigreeFunction` | Shapiro → (probably) MWU, one-tailed. Less obvious; "cannot tell yet" is possible and allowed. |
| H3 | If you compare women by age band (21–29 / 30–39 / 40–49 / 50+), the share of later diabetes cases differs between the bands. | `Age` → bands | `chi2_contingency` + Cramér's V. χ² is **non-directional**: the verdict applies only to "differ". Whether the share rises with age, you describe separately using the band shares, without a verdict of its own. Fix the band boundaries now, not after looking at the data. |
| Found along the way | Is the Insulin value missing more often in one of the two groups? | `Insulin == 0` | descriptive (shares) or as a 4th test with α = 0.05/4 |

**Weaknesses of this recommendation (please weigh honestly):**
- **F2 is generous.** "All positive" reaches F2 = 5p/(4p+1). On project day, work this out with your p from `y_train`: with p = 0.35 it is already **0.73**. That is why `DummyClassifier(strategy="constant", constant=1)` has to go into the table as a third baseline, otherwise a mediocre model looks brilliant. This is the single most valuable tip from Strategy A.
- **One-tailed tests** give you more power, but can never establish the opposite direction ("contradicted" only descriptively). Alternative: everything two-tailed, which is stricter but easier to defend. Decide before looking at the data.
- **χ² on bands** loses information compared with MWU on the continuous age and cannot establish a direction. On the other hand, it practises the "category × category" row of the test table. A deliberate choice, not a requirement. Anyone who wants to formally test "the older, the more frequent" uses MWU one-tailed on `Age` instead.
- If you would rather run **one pure strategy**: A if time is short, B if you want to practise data quality, C if the threshold-cost curve appeals to you.

**Time frame (rough, from project start):**

| Block | Duration | Content |
|:--|:--|:--|
| 1 | 30 min | Step 1 in writing (metric, k sentence, research question, H1–H3, band boundaries, one- or two-tailed), commit |
| 2 | 15 min | Load (size and columns only), split |
| 3 | 90 min | EDA, `0 → NaN`, missing shares, `cfg` and `apply_cleaning`, three tests each with plot, number and verdict |
| 4 | 45 min | X/y, three baselines, rule cutoff on train |
| 5 | 45 min | logreg, balanced, threshold from train |
| 6 | 45 min | Test set through `apply_cleaning`, one table, confusion matrix of the best model, conclusion (mentioning the selection bias according to the paper; `balanced` probabilities are not risks) |
| Buffer | remainder | first the coefficient comparison from C (scaled `coef_` next to the verdicts), then variants from B/C (without Insulin/SkinThickness, own weights), quiz each other: Can we explain every step? |

Evaluation of the three strategies: `04-evaluation.md`.

Emergency version if time gets tight: Block 3 with only two plots, all three tests, no comparison of variants. Block 5 only balanced + threshold.

## 6. Pitfall checklist for project day

- [ ] Hypotheses, metric and `P_min` are written down **before** the first `head()`
- [ ] Checked after the split: `P_min` is clearly above `p_train`, otherwise "all positive" would pass the guardrail
- [ ] CSV checked against the paper: 768 rows, Glucose range (K1)
- [ ] For each test, stated which difference was tested (mean for Welch, rank order for Mann-Whitney, shares for χ²)
- [ ] `df_test` not touched before step 8, not even a "quick" `describe()`
- [ ] Median computed **after** `0 → NaN` (otherwise too small), from `df_train` only
- [ ] No class-specific median (impossible on the test set because the `Outcome` of new patients is unknown)
- [ ] Band boundaries set in advance; merging bands only based on row counts, never based on shares or p
- [ ] Every additional real test counted (Bonferroni)
- [ ] Verdicts worded as "supported / contradicted / cannot tell yet", never "proven"
- [ ] Scaler and threshold from train only, no rows dropped in the test set
- [ ] Table contains train **and** test, Δ commented on, "all positive" baseline included
- [ ] Confusion matrix read in absolute numbers: how many future cases missed, how many unnecessary check-ups

## 7. Questions for the coaches

1. Hypothesis tests on the version with NaN (without imputation) or on the imputed training set? The assignment says "cleaned training set".
2. Are one-tailed tests wanted in the project, or is two-tailed the standard?
3. Does a test under "found along the way" count towards Bonferroni?
4. (From the laundry list, Oct 8) "precision-recall curve (balanced data only)" is the wrong way round according to slides 31/32. Has that been corrected?

## Files in this folder

- `00-summary.md` – this file: corrections, comparison, recommendation, checklist
- `01-strategy-A-screening-net.md`, `02-strategy-B-data-detective.md`, `03-strategy-C-cost-matrix.md` – the three agent drafts verbatim, with correction notes
- `04-evaluation.md` – evaluation of the three strategies with scores and the cross-check by Codex

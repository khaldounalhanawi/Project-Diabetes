# Strategy B – Data Detective

> Draft by an AI agent (Claude Sonnet) for the diabetes project day, verbatim. Corrections after the fact check (paper, course slides) and the Codex cross-check appear as ⚠ notes in the text and below in the reading notes. Summary and recommendation: `00-summary.md`.

**Reading notes after the Codex verifier (Oct 8, evening).** They apply to all three strategy files, even without their own ⚠ marker:

- **Diagnosis wording (on K1):** Wherever the agents speak of "diagnosis", "untreated diabetes", "positive diabetes test", "has diabetes" or "confirmation test", what is meant is: **later onset of diabetes within 1–5 years**. A FN is then a missed opportunity for earlier follow-up or prevention, a FP an unnecessary check-up. Present it this way to the coach, not as a diagnosis today.
- **A, `P_min` = 0.33:** too low if the share of sick patients is above it. See `00-summary.md`, section 5.
- **A and C, test chains:** Welch and Mann-Whitney test different null hypotheses. The "cross-check with the other test" is only a sensitivity check. See `00-summary.md`, section 4, point 6.
- **B, H2 and C, H2 (χ² on bands):** χ² only shows that "proportions differ", not a rising trend. C's criterion "highest band > lowest" also does not show a consistent increase, it only describes.
- **C, "`class_weight={0:1, 1:k}` shifts the odds by a factor of k":** only a rough picture, not an identity. With regularized sklearn regression, weights can also change the slopes. `1/(1+k)` is the cost-optimal threshold only for well-calibrated probabilities. The actual threshold is still chosen on the training data.

---

## 1. Core idea
Before we model, we have to trust the measurement: in several columns missing values are probably recorded as 0. The common thread is to find them and to assess their mechanism (MCAR/MAR/MNAR) with justification. In addition there is a deliberate decision between drop and impute, and testing hypotheses across groups and categories (chi-square) as well, not only across means.
**Choose it if** the team wants to learn from data quality and document every cleaning decision (change log). **Rather not** if the day is short or pure model quality counts: The strategy produces many variants (see 9).

## 2. FP vs. FN and choice of metric
- **FN** (diabetes missed): stays untreated, long-term damage (eyes, kidneys, vessels) continues unnoticed. That is exactly what screening is for. **FP** (false alarm): confirmation test, worry, costs, occupied capacity.
- Judgment: FN is worse, but not without limit. "Flag everyone" gives recall 100% at precision = share of positives (w02d4). That costs resources (follow-up tests) and creates alarm fatigue, i.e. a loss of trust in the screening.
- **Main metric F1** (positive class, `f1_score`): The harmonic mean punishes sacrificing precision. Honest catch: F1 weights FP and FN equally. We get the asymmetry via `class_weight` and threshold, and F2 checks whether the ranking of the models flips.
- **Secondary metrics:** Recall and precision individually, F2 (`fbeta_score(beta=2)`), **average precision** (`average_precision_score` on `predict_proba`, threshold-independent, reference = share of positives), confusion matrix. ROC AUC at most as an addition (too optimistic under imbalance).
- **Not accuracy:** A model that always says "healthy" has high accuracy with zero cases found. Accuracy only as a control column.

## 3. Research question
Which measurements and characteristics distinguish women with a positive diabetes test from those without, and does the mere fact that a measurement is missing carry information about the outcome?

## 4. Three hypotheses (set before looking at the data)
| | Hypothesis | Column(s) | H0 / H1 | Tails | Test chain | Effect size | Plot |
|:--|:--|:--|:--|:--|:--|:--|:--|
| H1 | **If** the 2-h insulin value is missing (value 0), **then** the diabetes share differs from that of patients with a measured value. | `Insulin` (helper column `insulin_fehlt`), `Outcome` | H0: Missingness and Outcome are independent (compatible with MCAR with respect to Outcome). H1: dependent (indication of MAR/MNAR). | two-tailed: Chi² captures every deviation from independence, there is no solid reason for a direction | binary, no band limits needed → `pd.crosstab` → check `expected` ≥ 5 (rule of thumb) → `stats.chi2_contingency` (choose the Yates correction deliberately, and note it). Fallback if a cell < 5: `fisher_exact` (beyond the course so far, optional) | Cramér's V | Bars of the diabetes share per group (axis from 0, n on the bar, no red-green) + crosstab heatmap |
| H2 | **If** patients fall into different age bands (21–29, 30–39, 40–49, 50+), **then** the diabetes share differs; expected: higher in the higher bands. | `Age` → `AgeBand`, `Outcome` | H0: Age band and Outcome are independent. H1: dependent. | two-tailed (Chi² is undirected); the expected direction is only described, not tested | `pd.cut` with fixed limits → `crosstab` → check `expected` → `chi2_contingency`. Fallback: merge bands by a rule fixed in advance (see below) | Cramér's V | Bars of the share per band (n on the bar) + normalized crosstab |
| H3 | **The higher** the diastolic blood pressure, **the more likely** diabetes: Outcome 1 has a higher `BloodPressure` than Outcome 0. | `BloodPressure` (measured values > 0 only), `Outcome` | H0: same location in both groups. H1: location differs (expected: higher for Outcome 1). | two-tailed: direction not established (age, medication). One-tailed would bring power, but is exactly the temptation we rule out in advance | Histogram + `skew()`, `stats.shapiro` per group (with large n Shapiro triggers early: read it together with the histogram). Both p ≥ 0.05 → `ttest_ind(equal_var=False)` (Welch; `levene` then only for documentation), otherwise `mannwhitneyu` | Correlation ratio η + difference in mmHg | Boxplot per Outcome (n labeled) + overlaid histograms |

- **Bonferroni:** α = 0.05 / 3 ≈ **0.0167** per test. Keep a counter `n_tests`: Every further test (also from "found along the way") lowers the threshold again. The MAR comparisons in section 5 remain descriptive (effect size only, no p-values).
- **Verdict:** *supported* = p < 0.0167 and direction as expected ("significant evidence for …, p = …, effect small/medium/large"). *contradicted* = p < 0.0167, but direction opposite to the expectation (impossible for H1, which has no direction). *cannot tell yet* = everything else, including 0.0167 ≤ p < 0.05 (do not reinterpret after the fact). Never "proven", never "H0 accepted" (fail to reject ≠ accept).
- **Band limits in advance:** Decades are a convention unrelated to the data. Merge only if a band has < 10% of the training rows or `expected` is < 5, and only by row totals, never by Outcome shares or p. Why: Anyone who sets limits after looking at the shares unconsciously optimizes the Chi². That is p-hacking, the finding does not replicate (w02d2).
- **H1, domain view:** 2-h serum insulin is a laborious lab determination. Gaps can have protocol or lab reasons (MCAR/MAR) or be caused by the measurement itself (MNAR). That is open and the point of the hypothesis. *Forecast (not binding):* very uncertain; a non-significant result is no evidence for MCAR.
- **H2, domain view:** Type 2 risk rises with age. *Forecast:* rather supported. Uncertain are the thinly populated top band and the entanglement of age ↔ pregnancies.
- **H3, domain view:** Hypertension and diabetes occur together more often (metabolic syndrome), but the diastolic value varies widely and depends on age. *Forecast:* small effect, "cannot tell yet" is a realistic, intended result.

## 5. EDA & cleaning plan (centerpiece)
Order: **split first** (`train_test_split(df, test_size=0.25, stratify=df["Outcome"], random_state=42)`, the target stays in the frame; 0.25 instead of 0.2 for more positive cases in the test set), then everything only on `df_train`. `df_test` stays closed, not even a "quick" `describe()`.

| Column | 0 is … (domain knowledge only, before looking at the data) | Missing share (guess) | Mechanism (guess) |
|:--|:--|:--|:--|
| `Glucose` | impossible (plasma glucose 0 is incompatible with life); core measurement of the OGTT | low | MCAR (recording error) |
| `BloodPressure` | impossible (diastolic 0 mmHg = circulatory arrest) | low to medium | MCAR/MAR |
| `BMI` | impossible (weight / height²) | low | MCAR |
| `SkinThickness` | implausible (a skin fold has a measurable thickness); caliper measurement is error-prone, difficult with severe overweight | high | MAR/MNAR (may depend on BMI itself) |
| `Insulin` | implausible for a 2-h serum value; laborious lab determination | very high | MAR, MNAR possible |
| `Pregnancies` | **legitimate** (never been pregnant), count variable | – | do not touch |
| `Age`, `DiabetesPedigreeFunction` | `Age` < 21 would contradict the description; DPF is a computed index, 0 not expected | – | validity check only |

1. **Audit on `df_train`:** `describe()`, share `(spalte == 0).mean()` per candidate, histogram + `skew()`, boxplot/Tukey fence. Outliers are candidates, not a deletion order: data errors (note domain limits in advance) vs. real extremes (keep, w02d1).
2. **0 → NaN** only in the five listed columns (list fixed in advance, do not extend it after looking at the data). `Pregnancies` stays untouched.
3. **Mechanism diagnosis (detective part, train only):** (a) share per column; (b) coupling: are Insulin and SkinThickness missing together (`crosstab` of the missingness indicators)? Strong coupling points to a common cause in the measurement process. (c) MAR check: compare BMI, Glucose, Age by missingness group (boxplot + η). (d) H1. MNAR can never be proven from the data, only made plausible by a domain argument.
4. **Decision rule** (in advance; the thresholds are rules of thumb, not from the slides; shares from `df_train`). **Rows are never dropped:** Dropping rows changes the sample (w02d1), the test set must not shrink, and all variants must be compared on the same test rows. Per column:
   - < 5%: median impute.
   - 5–40%: median impute, plus indicator column `<spalte>_fehlt` if step 3 shows dependence.
   - > 40%: the main model imputes (+ indicator if dependent), the drop variant of the value column runs alongside. Which one is reported is decided by the domain value of the column and the indicator finding, never by the test score.
5. **Median instead of mean:** Insulin/SkinThickness are probably right-skewed (`skew()` > 1 as evidence), the mean is pulled by the tail (w02d1). One rule for all columns means fewer degrees of freedom. Compute the median **after** 0 → NaN, with the zeros it would be too small.
6. **Tests run on the NaN version** (omit missing values per test), not on imputed data: median filling narrows the spread and pulls correlations toward 0 (w02d1). That is our reading of "cleaned training set", imputation applies only to the model. Cross-check this reading with the coach.
7. **Change log:** a dict `cfg` (`zero_cols`, `missing_share_train`, `train_medians`, `drop_cols`, `age_bands`, `threshold`) plus a list `log` (step, columns, rule, value, number of affected rows). Exactly one function `apply_cleaning(df, cfg)` for train and test.
8. **Test transfer (task step 8):** `apply_cleaning(df_test, cfg)` with `fillna(cfg["train_medians"])` and `drop(columns=cfg["drop_cols"])`. Indicators come from the test set's own zeros (that learns nothing). Then `assert`: no NaN, same columns in the same order as train.
9. **Leakage traps:** Median, `StandardScaler` (`fit` on train only) and threshold from train only. Choose nothing (variant, threshold, band limit) after the test score, name the main model in advance. Age bands serve only the hypothesis and the plot, the model gets `Age` numerically. **Target-leak suspicion `Glucose`:** To the best of my knowledge, the 2-h plasma glucose value is part of the diabetes criterion that defines `Outcome`. Check in the attached paper. If so, a good model is less impressive than it looks, and that belongs in the discussion ("known in time? computed without the target?", w02d3).

> ⚠ **Correction K1:** According to the paper, `Outcome` is the onset of diabetes 1–5 years after an examination with a normal glucose test. Glucose is a legitimate predictor, not target leakage and not a diagnostic criterion for `Outcome`.


## 6. Baselines
- **Random:** `DummyClassifier(strategy="stratified", random_state=42)`. Expectation (arithmetic identity, not a finding from the data): precision = recall = share of positives, so F1 ≈ that share, likewise the AP reference. A single random draw fluctuates, so fix the seed and do not over-interpret. Optional teaching row: `strategy="most_frequent"` (accuracy high, F1 = 0).
- **Rule-based:** The column is the η winner on `df_train`. Domain knowledge suspects `Glucose`, but the ranking decides, not the wish. Rule "value > c ⇒ diabetes". c is chosen on train: grid of quantiles (`np.percentile`), c = argmax F1(train), fill NaN beforehand with the train median. For AP the column value itself serves as the score. A cross-check against the clinical reference limit (domain knowledge, check the unit) serves only for plausibility.

## 7. Models
Extend the import cell: `StandardScaler`, `fbeta_score`, `average_precision_score`, `precision_recall_curve`. Everything runs after `apply_cleaning`: `X`/`y` from `df_train`, `StandardScaler().fit(X_train)`, then `transform` on train and test. L2 regularization is scale-dependent, and with scaling the coefficients become comparable.
- **M1** `LogisticRegression()`, all columns (impute), threshold 0.5. **M2** `LogisticRegression(class_weight="balanced")`, threshold 0.5.
- **M3** = M2 + threshold on train: `predict_proba(X_train)[:, 1]` → `precision_recall_curve` → F1 per threshold → argmax into `cfg["threshold"]`. On the test set `proba >= thr` applies. Catch: The threshold is chosen on the same data on which the model was fitted, which is slightly optimistic. Beyond the course so far, optional: `cross_val_predict(method="predict_proba")`.
- **V-Drop** = M2 and M3 without the columns that trigger the 40% rule (presumably Insulin, SkinThickness; the share decides, not the guess). **V-Ind** (optional, only if H1/MAR check shows dependence): impute + indicator columns.
- **Main model in advance: M3 (impute).** The variants answer "does the cleaning choice move the result?" (w02d1: try the alternative), they do not serve to search for a winner.

## 8. Results table (sketch) and confusion matrix
| Model | Cleaning | Threshold | F1 train | F1 test | Δ train−test | Recall test | Precision test | F2 test | AP train / test | Accuracy test (control) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|

Rows: Dummy stratified · (Dummy most_frequent) · Rule `Glucose > c` · M1 · M2 · M3 · V-Drop (M2/M3) · (V-Ind). The AP is identical for M2 and M3 (same model, only a different threshold), that is expected.
- **Reading the matrix:** Rows = truth, columns = prediction, 0 before 1 (TN FP / FN TP). Row "truth 1": How many cases were missed (FN)? Column "prediction 1": How many alarms are false (FP, determines precision)? M2 → M3 only shifts cases between FN/TP and TN/FP, the model stays the same.
- **Detective add-on:** Break down FN/FP of the best model by the group "Insulin imputed yes/no". Only describe, change nothing afterwards; the groups are small.
- A large Δ train−test points to overfitting. Differences of a few hundredths of F1 lie within the noise of a test set with only a few dozen positive cases.

## 9. Considerations and weaknesses (sparring)
- **Median impute with 40–50% gaps distorts:** Half of the column consists of a constant, spread and relationship with `Outcome` are diluted, the coefficient is unreliable. Drop, on the other hand, loses real information. Both are an assumption, which is why we show both variants. Better, but beyond the course so far (optional): predict missing values from other columns.
- **H1 can never show MCAR:** Not significant means "cannot tell yet", MNAR remains a domain argument. The centerpiece of the strategy rests partly on something untestable. H1 can also never be *contradicted*.
- **Categorizing costs information:** 29 and 30 years land in different bands, and the limits are arbitrary. The verdict could flip with other limits. Chi² is undirected, "the older, the more" remains a description. Better would be `spearmanr` on continuous `Age` (one more test, α recomputed).
- **Small cells:** The top age band (and H1 after the split) can become thin, check `expected`. The Yates correction changes V for 2×2.
- **Time budget (one day):** The variants multiply (drop/impute × plain/balanced × threshold × indicator). Order when time is short: split + `apply_cleaning` + log → H1–H3 → baselines + M1–M3 → table. V-Drop/V-Ind only with spare time.
- **Metric contradiction:** F1 weights FP = FN, our justification says FN > FP. Anyone who wants that consistently takes F2. We keep F1 because precision should not be sacrificed, and we report F2 as well.
- **Strength of the conclusions:** The test set is small, the difference between drop and impute may lie within the noise. "The cleaning choice changes little" is also a result. Confounding (age ↔ pregnancies ↔ blood pressure) and the Glucose target-leak suspicion qualify any praise of a model.
- **Decision, approval please:** We deliberately do not test the most obvious column (`Glucose`) formally, it appears in the EDA (η) and in the rule-based baseline. Should H2 (age) be swapped for a `Glucose` hypothesis instead? That would be possible without consequences for the rest. Can be adopted regardless of strategy: `apply_cleaning` together with the log, and fixing the band limits in advance.

## 10. Control questions from the coach
1. Show me the place in the code where a value from `df_test` is read for the first time. Which number was not computed from the test set in the process, and what would have happened if you had already replaced the zeros before the split?

> ⚠ **Note K2:** `0 → NaN` before the split is harmless (learns nothing, slide W2D3). Leakage only arises when filling in with a median from all rows.

2. Your H1 test is not significant. Are you allowed to write that the insulin gaps are random? What changes in your imputation if it had been significant?
3. You justify F1 by saying that a missed case is worse than a false alarm, but F1 weights both equally. How many false alarms are you willing to accept per case found, and does F2 change the ranking of your models?

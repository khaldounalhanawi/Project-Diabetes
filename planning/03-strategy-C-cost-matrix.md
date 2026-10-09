# Strategy C – Cost Matrix & Dose-Response

> Draft by an AI agent (Claude Sonnet) for the diabetes project day, verbatim. Corrections after the fact check (paper, course slides) and the Codex cross-check appear as ⚠ notes in the text and below in the reading notes. Summary and recommendation: `00-summary.md`.

**Reading notes after the Codex verifier (Oct 8, evening).** They apply to all three strategy files, even without their own ⚠ marker:

- **Diagnosis wording (on K1):** Wherever the agents speak of "diagnosis", "untreated diabetes", "positive diabetes test", "has diabetes" or "confirmation test", what is meant is: **later onset of diabetes within 1–5 years**. An FN is then a missed opportunity for earlier follow-up or prevention, an FP an unnecessary check-up. Word it this way for the coach, not as a current diagnosis.
- **A, `P_min` = 0.33:** too low if the share of sick patients is above it. See `00-summary.md`, section 5.
- **A and C, test chains:** Welch and Mann-Whitney test different null hypotheses. The "cross-check with the other test" is only a sensitivity check. See `00-summary.md`, section 4, point 6.
- **B, H2 and C, H2 (χ² on bands):** χ² only shows that "proportions differ", not a rising trend. C's criterion "highest band > lowest" does not show a continuous rise either, it only describes.
- **C, "`class_weight={0:1, 1:k}` shifts the odds by a factor of k":** only a rough picture, not an identity. With regularized sklearn regression, weights can also change the slopes. `1/(1+k)` is the cost-optimal threshold only for well-calibrated probabilities. The actual threshold is still chosen on the training set.

---

## 1. Core idea
The trade-off "FN is worse than FP" is put into numbers: an assumption fixed in advance, "1 FN costs as much as k FP". From this one number follow the metric (F-beta), the class weights and the threshold. The hypotheses test dose-response ("the more …, the more …") on less obvious columns.
**Choose this if** the choice of metric should be defensible by calculation and you enjoy threshold curves. **Rather not if** k cannot be justified or time is short (emergency version: only F2 plus one threshold).

## 2. FP vs. FN – cost trade-off and choice of metric
- **Assumption, not a measurement:** FP = confirmation test (second glucose test, some effort and worry). FN = untreated diabetes with late complications (eyes, kidneys, nerves, heart). The resulting damage is likely many times more costly. Because an FP also means worry and follow-up tests, though, we deliberately set a cautious **k = 5** (sensitivity: 2 and 10). k is a stipulation without a data source. Before submission, either find a source or leave it openly as a stipulation.
- k means the *additional* cost compared with the correct decision. TP and TN count as 0 (simplification).
- **Why not accuracy:** A model that never says "diabetes" has accuracy = share of healthy people and finds no case (slide "A test that does nothing"). Accuracy counts FN and FP equally.
- **Main metric F2** (`fbeta_score(..., beta=2)`). **Decision measure cost per patient** = (k·FN + FP)/n. Secondary metrics: recall, precision, average precision (more informative under imbalance), ROC AUC (only as ranking quality, independent of threshold and k), confusion matrix.
- **Relationship between k, beta, threshold** (rules of thumb, not identities):
  - Threshold: With calibrated probabilities, "positive from p ≥ 1/(1+k)" would be cost-optimal, so about 0.17 for k=5 (k=2: 0.33; k=10: 0.09). *Beyond the course so far, optional* (derivation). Orientation only: LogReg is only roughly calibrated, the training is in-sample. t* is chosen empirically.
  - beta: F_β = (1+β²)·TP / ((1+β²)·TP + β²·FN + FP). This is pure algebra, *beyond the course so far, optional*. An FN thus counts β² times as much as an FP, so β ≈ √k ≈ 2.2. We round to β = 2 (corresponds to k ≈ 4), "2.24" would be false precision. F-beta is still not the cost function (it divides by TP), the optima can differ.
  - Weights: `class_weight={0:1, 1:k}` shifts the odds by a factor of k and acts roughly like the threshold 1/(1+k) on the unweighted model.
- **Why k must be fixed before the results:** The slide "When one mistake matters more" shows two fixed models whose winner changes depending on β (F1: default ahead, F2: lowered threshold ahead). Whoever chooses k or β after the result is thereby choosing the winner (slide: "metric chosen before the results are in"). In practice: write k, β and the hypotheses into a Markdown cell before `read_csv` and commit locally (timestamp, no push).

## 3. Research question
Among Pima women aged 21 and over, does the probability of a positive diabetes result rise with family history, number of pregnancies and age, and can cases be found with them so that the expected costs fall at k = 5?

## 4. Three hypotheses (before looking at the data, all tests on the training set only)
| Hypothesis | Column(s) | H0 / H1 | Tails | Test chain | Effect size | Plot |
|---|---|---|---|---|---|---|
| **H1** The higher the family history burden, the more likely a positive result | DiabetesPedigreeFunction × Outcome | H0: DPF equally distributed in both groups · H1: higher in Outcome 1 | one-tailed (`greater`): heredity as a risk factor is domain knowledge, direction set before looking at the data | `shapiro` per group + histogram/skewness → normal: `levene` → Welch `ttest_ind`; otherwise `mannwhitneyu`. Expected to be right-skewed, so MWU | η, plus median per group | Boxplot per Outcome; bars "share with diabetes per DPF tercile" |
| **H2** The more pregnancies, the higher the share with diabetes | Pregnancies in bands (0 / 1–2 / 3–5 / ≥6) × Outcome | H0: share equal in all bands · H1: shares differ, expected to rise | χ² is non-directional. Read the direction as ordered shares (criterion fixed in advance: highest band > lowest) | `chi2_contingency`; expected counts ≥ 5 (`expected`), otherwise merge the upper bands (rule fixed in advance). Alternative: MWU on the raw count (more power, many ties) | Cramér's V | Bars of share per band, n per bar, y starting at 0 |
| **H3** The older, the higher the share with diabetes | Age × Outcome (descriptive: age bands 21–29 / 30–39 / 40–49 / ≥50) | H0: Age equally distributed in both groups · H1: higher in Outcome 1 | one-tailed (`greater`): age risk is domain knowledge | as H1 (expected to be right-skewed, so MWU) | η, median per group | Boxplot/violin; bars of share per age band |

- **Test choice according to the course table:** Outcome is binary, so "two groups × numeric" means t-test/MWU. Spearman/Pearson are number × number and would be only a detour with a 0/1 column. Spearman makes sense only for Pregnancies ↔ Age, as a descriptive confounding check (ρ as effect size, no verdict p-value). Bands show the dose clearly but cost information, and the boundaries must be fixed in advance.
- **Bonferroni:** 3 hypothesis tests, so α* = 0.05/3 ≈ 0.0167 per test. Shapiro/Levene are assumption checks and do not count. Every further test (robustness counter-test, p for Pregnancies–Age) is counted too or marked as "no verdict test".
- **Verdicts:** *supported* = p ≤ α* and direction as predicted (always report the effect size: significant ≠ important). *contradicted* = direction clearly reversed (median/shares fall, one-tailed p close to 1). *cannot tell yet* = everything else (p > α*, assumption violated, thin bands). Wording: "The data support / contradict / leave open …", never "proven"; fail to reject ≠ accept. Price of one-tailedness: a counter-effect can then only be described, not tested.
- **H1, rationale and uncertainty:** DPF summarizes diabetes in relatives by degree of relationship. The direction is plausible. The effect is likely small to medium a priori because the index is crudely constructed, and after Bonferroni it is not firmly established.
- **H2, rationale and uncertainty:** Multiple pregnancies (and gestational diabetes) are associated with later type 2 risk. But Pregnancies grows with Age: "supported" does not mean "independent of age". Upper bands may be thinly populated.
- **H3, rationale and uncertainty:** Type 2 risk rises with age, a priori this is the safest direction of the three. The sample contains only women aged 21 and over from one population group, so the trend may flatten. MWU tests "tends to be older", not the stepwise shape. H2 and H3 are not independent, Bonferroni remains conservative.

## 5. EDA & cleaning plan
- Split first (`train_test_split`, `stratify=Outcome`, fixed `random_state`). EDA and tests only on `df_train`.
- **Hidden zeros (domain knowledge):** Glucose, BloodPressure (diastolic), SkinThickness and BMI = 0 are physiologically impossible. Insulin = 0 (2-h value) is highly implausible. These zeros are treated as NaN (`replace(0, np.nan)`). **Pregnancies = 0 and Outcome = 0 are valid.** How many zeros there are is only shown by `(df_train == 0).sum()`. DPF, Pregnancies and Age have no hidden zeros, so the tests do not depend on imputation.
- Missingness mechanism (MCAR/MAR/MNAR) is unknown, for Insulin/SkinThickness probably not MCAR. Imputation: train median per column. Cross-check ("Try the alternative"): drop rows. Does a verdict flip as a result?
- **Change log:** One table/dict with `fill_values` (train medians), number of values replaced per column, band boundaries, k, β, t*, rule cutoff c. The test set receives exactly these values.
- IQR/boxplot yields candidates, not deletion orders. Genuine extremes stay, only data errors are removed, with a justification.
- **Leakage traps:** computing the median or scaler on all data; looking into `df_test` before step 8; changing k, band boundaries or t* after looking at test or p-values; imputation before the split. Glucose comes from the glucose tolerance test and is close to the diagnostic criterion (suspected near-target leakage). Check against domain knowledge/paper and name it in the report.

> ⚠ **Correction K1:** According to the paper, `Outcome` is the onset of diabetes 1–5 years after an examination with a normal glucose test. Glucose is a legitimate predictor, not a target leak and not a diagnostic criterion for `Outcome`.


## 6. Baselines
- **Random:** `DummyClassifier(strategy="stratified", random_state=42)`. In expectation, recall ≈ precision ≈ share p₁ of positives. Sanity check for the cost per patient: (k+1)·p₁·(1−p₁).
- **Cost anchors (addition):** "all negative" costs k·p₁, "all positive" costs 1−p₁ per patient (`DummyClassifier(strategy="constant", constant=…)`). A model is worthwhile only below min(k·p₁, 1−p₁).
- **Rule-based:** Column with the largest effect (η) from the training EDA (selection rule fixed in advance, a selection and not a test). Rule "value ≥ c ⇒ positive", c = minimum of k·FN+FP **on the training set** over a candidate list (quantiles or unique values). c goes into the change log. Same optimism warning as for the model threshold.

## 7. Models
- **Features:** all 8 after cleaning. `StandardScaler().fit(X_train)`, then transform both sets. Scaling is necessary because raw coefficients depend on units (insulin in µU/ml vs. DPF) and only scaled coefficients ("log-odds per 1 SD") are comparable. In addition, the default L2 regularization depends on the scale.
- **Variants**, all `fit` on train: (a) unweighted, threshold 0.5 · (b) `class_weight="balanced"`, 0.5 · (c) `class_weight={0:1, 1:k}`, 0.5 · (d) unweighted plus t* (cost minimum on train). Optional (e) = (c) plus t* as a consistency check: Is t* close to 0.5?
- **balanced vs. {0:1, 1:k}:** "balanced" weights by frequency (n/(2·n_c), ratio n₀/n₁). It reflects the costs only if k happens to be ≈ n₀/n₁ ("Frequency is not cost"). Custom weights set the ratio directly to k. Both change the *training*: probabilities shift upward and can no longer be read as risk. The threshold changes only the *decision*.
- **t*:** `predict_proba(X_train_s)[:, 1]`, grid over t, cost(t) = k·FN + FP, `argmin`. On ties, take the higher threshold (rule fixed in advance). Cost curves for k = 2/5/10 in one figure: a flat minimum means the choice is uncritical.
- **After step 9, comparing model and tests:** Put the scaled `model.coef_` per column next to the verdicts (sign and rough size, sklearn provides no p-values).

## 8. Results table sketch (placeholder "…", k = 5 fixed in advance)
| Model | Threshold | F2 train | F2 test | Recall test | Precision test | AP / AUC test | Acc test | Cost/pat. train | Cost/pat. test |
|---|---|---|---|---|---|---|---|---|---|
| Random (stratified) | – | … | … | … | … | … | … | … | … |
| Rule-based (column, c) | c | … | … | … | … | … | … | … | … |
| LogReg (a) unweighted | 0.5 | … | … | … | … | … | … | … | … |
| LogReg (b) balanced | 0.5 | … | … | … | … | … | … | … | … |
| LogReg (c) {0:1, 1:k} | 0.5 | … | … | … | … | … | … | … | … |
| LogReg (d) unweighted + t* | t* | … | … | … | … | … | … | … | … |
| Anchors: all negative / all positive | – | … | … | … | … | … | … | … | … |

- **Reading the matrix:** `cm = confusion_matrix(y, y_pred)` gives `[[TN, FP], [FN, TP]]`, cost = `k*cm[1,0] + cm[0,1]`, divided by n (train and test differ in size). From (a) to (d), FN falls and FP rises. That is worthwhile only if k·ΔFN > ΔFP.
- (a) and (d) are the same model, so AP/AUC are identical. (b) and (c) are different models, so AP/AUC differ slightly. Accuracy is in the table only as a warning.
- Residual errors: Which cases remain FN (profile, imputed values)? Describe them, do not retune on the test set. Comment on the train–test gap.

## 9. Considerations & weaknesses (sparring)
- **k is arbitrary.** Sensitivity table for k ∈ {2, 5, 10} (t*(k), cost, winning model). The k = 5 fixed in advance remains the main result. The sensitivity is only reported, otherwise k ends up being chosen after the fact. An honest statement is a range statement ("(d) wins for k from … to …").
- **t* on the training set is over-optimistic** (in-sample, noisy cost curve). Test cost ≥ train cost is expected. Optional, *beyond the course so far* (week 3): `cross_val_predict` on the training set.
- **Small sample:** few positive test cases, one more FN costs k points. Do not over-interpret small differences. Confidence intervals (bootstrap) would be *beyond the course so far, optional*.
- **Coefficients ≠ causality.** Correlated features (Age–Pregnancies, BMI–SkinThickness) make sign and size unstable. The univariate test (marginal) and the coefficient (conditional) answer different questions. Three readings: consistent · "explained away" (test yes, coefficient ≈ 0, e.g. Pregnancies by Age) · unclear. The comparison is a plausibility check and not a fourth test.
- **Scenario consistency:** "FP = confirmation test" fits only halfway because Glucose itself comes from the glucose tolerance test. Name it as a mental model (prioritizing further follow-up), not as an actual screening process. A constant k for everyone and TP/TN = 0 are simplifications.

> ⚠ **Correction K1:** According to the paper, `Outcome` is the onset of diabetes 1–5 years after an examination with a normal glucose test. Glucose is a legitimate predictor, not a target leak and not a diagnostic criterion for `Outcome`.

- **Two target quantities** (report F2, optimize cost for t*) can favor different models. Report this openly, do not silently pick one.

## 10. Coach's check questions
1. Where does k = 5 come from, and what changes about β, weights and threshold at k = 20? Until when were you allowed to fix k, and why not afterwards?
2. Why is t* on the training set permissible and the test cost per patient nevertheless an honest estimate? What would be different if you had adjusted k or t* after looking at the test values?
3. The test for Pregnancies is significant, the scaled coefficient is close to zero. Is that a contradiction, and how would you clarify whether pregnancies are associated with diabetes independently of age?

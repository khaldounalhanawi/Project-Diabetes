# Strategy A – Screening Net

> Draft by an AI agent (Claude Sonnet) for the diabetes project day, verbatim. Corrections after the fact check (paper, course slides) and the Codex cross-check appear as ⚠ notes in the text and below in the reading notes. Summary and recommendation: `00-summary.md`.

**Reading notes after the Codex verifier (Oct 8, evening).** They apply to all three strategy files, even without their own ⚠ marker:

- **Diagnosis wording (on K1):** Wherever the agents speak of "diagnosis", "untreated diabetes", "positive diabetes test", "has diabetes" or "confirmation test", what is meant is: **later onset of diabetes within 1–5 years**. A FN is then a missed opportunity for earlier follow-up or prevention, a FP an unnecessary check-up. Present it this way to the coach, not as a diagnosis today.
- **A, `P_min` = 0.33:** too low if the share of sick patients is above it. See `00-summary.md`, section 5.
- **A and C, test chains:** Welch and Mann-Whitney test different null hypotheses. The "cross-check with the other test" is only a sensitivity check. See `00-summary.md`, section 4, point 6.
- **B, H2 and C, H2 (χ² on bands):** χ² only shows that "proportions differ", not a rising trend. C's criterion "highest band > lowest" also does not show a consistent increase, it only describes.
- **C, "`class_weight={0:1, 1:k}` shifts the odds by a factor of k":** only a rough picture, not an identity. With regularized sklearn regression, weights can also change the slopes. `1/(1+k)` is the cost-optimal threshold only for well-calibrated probabilities. The actual threshold is still chosen on the training data.

---

## 1. Core idea
The model is an initial screening in the GP practice. It should miss as few diabetes cases as possible and may accept false alarms for that, because they only cost a cheap confirmation test (e.g. HbA1c). The hypotheses test the classic, strong risk markers (Glucose, BMI, Age). The model is a `LogisticRegression(class_weight="balanced")` with a threshold chosen on the training set, measured by F2 with a precision guardrail.
**Choose it if** you want a clear, medically tellable metric story and prefer to test well-founded hypotheses cleanly rather than hunt for surprises. **Rather not if** false alarms would be really expensive (see 9).

## 2. FP vs. FN and choice of metric
- **FN (missed case):** The diabetes stays untreated for years, long-term damage to eyes, kidneys, nerves and blood vessels accumulates. Serious and gradual.
- **FP (false alarm):** A healthy person takes a confirmation test and has an uneasy week. Mild, but the volume matters: too many false alarms burden the practice and trust.
- **Main metric F2** (`fbeta_score(beta=2)`, positive class = diabetes): weights recall more than precision, without ignoring precision entirely. Pure recall would reach 100% with "everyone sick". β=2 is an assumption about the cost ratio, not a law of nature ("Frequency is not cost").
- **Guardrail:** Precision ≥ `P_min`, fixed before the results are in (proposal: "at most 2 false alarms per finding", i.e. ≥ 0.33). `P_min` must be clearly above the positive share `p` from `y_train`, otherwise screening would be no better than rolling dice.
- **Secondary metrics:** Recall (target quantity), precision, F1 (shows the price), average precision / PR curve (threshold-independent, suits the imbalance), confusion matrix. ROC AUC only on the side, because it flatters when the positive class is rare.
- **Why not accuracy:** According to the task, most patients do not have diabetes. An "everyone healthy" model would have high accuracy and find zero cases. The majority class decides accuracy and, without `class_weight`, training as well. How strong the imbalance is should be counted from `y_train` first.

## 3. Research question
Which routinely measurable risk markers (Glucose, BMI, Age) separate patients with diabetes from those without, and are they enough to detect as many cases as possible without false alarms getting out of hand?

## 4. Three hypotheses (set before looking at the data)
| | Hypothesis | Column(s) | H0 / H1 | Tail | Test chain | Effect size | Plot |
|---|---|---|---|---|---|---|---|
| H1 | The higher the plasma glucose in the 2-h glucose tolerance test, the more likely a patient has diabetes. | `Glucose` ~ `Outcome` | H0: Glucose has the same distribution in both groups. H1: higher for Outcome 1 | one-tailed (greater): direction is part of the diagnostic criterion | Shapiro per group. Both p ≥ 0.05: Levene (info only), Welch with `alternative="greater"`. Otherwise `mannwhitneyu(alternative="greater")`. Cross-check with the other test | eta (for 0/1 groups = absolute value of r) plus difference of medians/means | Boxplot per Outcome; diabetes share per Glucose quartile (`pd.qcut`) |
| H2 | If BMI is higher, then diabetes is more frequent. | `BMI` ~ `Outcome` | analogous: H1 = BMI higher for Outcome 1 | one-tailed: overweight leads to insulin resistance, direction established | as H1 | as H1 | Boxplot; diabetes share per BMI quartile |
| H3 | The older the patient, the more likely she has diabetes. | `Age` ~ `Outcome` | analogous: H1 = Age higher for Outcome 1 | one-tailed: risk rises with age | as H1; Mann-Whitney more likely (age probably right-skewed, many ties) | as H1 | Boxplot; diabetes share per age quartile |

- **Bonferroni:** m = 3, so α_corr = 0.05/3 ≈ 0.0167 per test (or p·3 against 0.05). Formal additional tests from "found along the way" count, even failed ones. Otherwise label them as exploratory without a verdict. Shapiro and Levene are assumption checks and do not count.
- **Data basis of the tests:** `df_train`, hidden zeros as NaN, `dropna` per column and no imputation (it would blur group differences). State n per group.
- **Verdicts:**
  - *supported*: p ≤ 0.0167, direction as predicted, effect size named. Wording: "The training data support H1 (p = …, eta = …)."
  - *contradicted*: Direction clearly opposite (sign of the difference, plot). The one-tailed test does not establish this by itself, so only descriptively: "The data speak against H1."
  - *cannot tell yet*: p > 0.0167 with the right direction, or unclear assumptions. Wording: "insufficient evidence (p = …)". Fail to reject is not "H0 accepted". Never "proven" or "disproved".
- **Rationale and uncertainty** (expectation from domain knowledge, not from data):
  - H1: The 2-h glucose value in the OGTT is itself a diagnostic criterion (WHO). The direction is medically compelling, the uncertainty there is low. The real risk is circularity: "supported" then says little that is new (see 9).

> ⚠ **Correction K1:** According to the paper, `Outcome` is the onset of diabetes 1–5 years after an examination with a normal glucose test. Glucose is a legitimate predictor, not target leakage and not a diagnostic criterion for `Outcome`.

  - H2: High BMI promotes insulin resistance, an established type 2 risk factor. An effect is expected, but smaller and more overlapping than for glucose. Uncertainty medium.
  - H3: Risk grows with age, but here the uncertainty is greatest. The sample consists only of women aged 21 and over, Age is related to Pregnancies, and Bonferroni is strict. "cannot tell yet" would be a legitimate result, not a failure.

## 5. EDA & cleaning plan
- **Order:** Research question and hypotheses into the table, then load (size and columns only), then `train_test_split(stratify=Outcome, random_state=…)`. Everything else only on `df_train`.
- **Hidden zeros (from domain knowledge and the data description only):** Impossible as a real measurement are `Glucose` 0 (value from the tolerance test), `BloodPressure` 0 (diastolic, living people), `BMI` 0 (weight > 0), `SkinThickness` 0 mm (a skin fold has thickness) and `Insulin` 0 µU/ml (2-h value after sugar intake). Legitimate are `Pregnancies` 0 (never pregnant) and `Outcome` 0 (class). `Age` and `DiabetesPedigreeFunction` only need a validity check (range, type). Expectation: Insulin and SkinThickness are more affected than Glucose and BMI, because they are more laborious to measure. The extent is only revealed by `(df_train[cols] == 0).sum()`.
- **Missing type and handling:** Probably not MCAR (a measurement is omitted for clinical reasons), so MAR or MNAR. Deleting rows is not an option, it changes the sample and costs a lot at small n. Instead 0 → `np.nan` and **train median** per column (robust to skew, the mean would narrow the spread). Sensitivity run: model without Insulin/SkinThickness ("try the alternative"). Beyond the course so far, optional: indicator column "was missing", if the gap itself is informative (MNAR).
- **Further checks:** Duplicates, data types, `describe()`, histograms with skewness, boxplot/IQR as candidates. Keep real extreme values, only treat the impossible ones.
- **Change log:** `cleaning_log` with step · column · rule · learned value (`train_medians`). One function `apply_cleaning(df, medians)` runs for train and test with the same values. `StandardScaler` is fitted on train only, no row is deleted in the test set.
- **Leakage traps:**
  - Median over all data (the test set flows in).
  - **Class-specific median:** On the test set it is impossible, because `Outcome` of new patients is unknown. In training it would additionally write the target information into the features, the model learns "imputed value = answer", and the test score collapses.
  - Scaler or outlier limits on all data.
  - Threshold or cutoff after a look at the test set.
  - Rewording hypotheses after looking at the data (p-hacking).

## 6. Baselines
- **Random:** `DummyClassifier(strategy="stratified", random_state=42)` on `X_train, y_train`. By construction, recall and precision are approximately equal to the positive share `p`. One seed is chance; optionally average over several seeds. Additional references: `strategy="most_frequent"` (accuracy warning example) and `strategy="constant", constant=1` ("everyone sick"). Its F2 is 5p/(4p+1) and thus the bar a model has to clear. It is high as soon as p is not tiny (about 0.24 at p = 0.06).
- **Rule-based:** The marker is the column with the largest eta from step 4 (expected: Glucose, do not hard-code it). Rule: "value ≥ c means diabetes". c is chosen on the training set only. Candidates are the quantiles (or all values); for each c, `fbeta_score(beta=2)` and precision are computed. The highest F2 under precision ≥ `P_min` is chosen, i.e. the same metric and guardrail as for the model. Then freeze c.

## 7. Models
- `logreg`: all cleaned features (8), threshold 0.5. `logreg_balanced`: `class_weight="balanced"`, threshold 0.5 (shows the "weights" lever alone).
- **Scaling yes** (`StandardScaler`, fit on train only, first imputation, then scaling): sklearn regularizes (L2, `C=1`), the penalty depends on the units (Insulin in the hundreds, DPF below 2). In addition, lbfgs converges better, and the coefficients become comparable.
- **`logreg_balanced_thr` (candidate A):** balanced plus `thr_A`. Optional: `logreg_thr` (without balanced, threshold only; separates the two levers) and `logreg_cheap` without Glucose/Insulin (true initial screening with cheap features).
- **Threshold selection on the training set:**
  1. `proba_tr = predict_proba(X_train)[:, 1]`.
  2. Grid from 0.05 to 0.95 in steps of 0.01 (or `precision_recall_curve`).
  3. For each value compute precision, recall and F2 on train.
  4. Admissible is precision ≥ `P_min`. Among the admissible values take the one with the highest F2.
  5. Do not land on the edge: take the middle of the plateau, round to 0.05, freeze as `thr_A`.
  - `thr_A` can also be ≥ 0.5, because `balanced` already pushes the probabilities up. "Lowered" is a result of the procedure, not a requirement.

## 8. Results table and confusion matrix
| Model | Thr | Train: Recall · Precision · **F2** · AP | Test: Recall · Precision · **F2** · AP | F1 Train/Test | Accuracy (warning light only) | Δ F2 Train−Test |
|---|---|---|---|---|---|---|
| `dummy_stratified`, `dummy_alle_krank`, `rule_<marker>`, `logreg`, `logreg_balanced`, `logreg_balanced_thr`, if applicable `logreg_thr`, `logreg_cheap` | … | … | … | … | … | … |

- **Reading the table:** Who beats both baselines in F2 and at the same time in precision (above `p`)? How large is Δ (overfitting signal)? Does accuracy drop while recall rises? That would be intended, not an error.
- **Confusion matrix of the best model** (rows = truth, columns = prediction, 0 before 1):
  - FN cell (true 1, predicted 0) = missed cases, state in absolute numbers.
  - FP/TP = false alarms per case found, translated into confirmation tests per true finding.
  - Compare with `logreg_balanced` at 0.5: how many FN were bought with how many additional FP?
  - Look at the remaining FN individually (Glucose/BMI values, were they imputed?). Only describe, do not readjust.

## 9. Considerations & weaknesses
- **Precision collapse through double shift:** `balanced` and a lowered threshold both push toward recall. F2 is generous, since "everyone sick" already reaches 5p/(4p+1), and without `P_min` the "best" threshold can flag almost everyone. Countermeasures: guardrail, variant `logreg_thr`, read the confusion matrix.
- **Circularity and break with the screening concept:** Glucose from the 2-h OGTT is practically the diagnostic criterion. An "initial screening" with this measurement predicts what the diagnosis already provides, and H1 "supported" is almost guaranteed. `logreg_cheap` shows what a real screening with cheap features can do.

> ⚠ **Correction K1:** According to the paper, `Outcome` is the onset of diabetes 1–5 years after an examination with a normal glucose test. Glucose is a legitimate predictor, not target leakage and not a diagnostic criterion for `Outcome`.

- **Threshold overfitting:** Training probabilities are optimistic (the model knows the rows), and with a few hundred rows the edge wobbles. Countermeasures: plateau and rounding. Beyond the course so far, optional (week 3): `cross_val_predict` for out-of-fold probabilities.
- **Small sample:** The test set contains only a few dozen positive cases, one case more or less moves recall by several percentage points. Do not over-interpret model differences, read Δ Train−Test as a signal and not as a verdict.
- **Normality assumption:** With hundreds of rows, Shapiro rejects almost every trivial deviation, so the choice of test then hinges on a threshold value. Countermeasures: fix the rule in advance, add histogram and skewness, report both tests. Mann-Whitney compares distributions, not means, so report medians as well.
- **One-tailed and Bonferroni:** One-tailed tests cannot demonstrate the opposite direction. Bonferroni at m = 3 is strict and can push H3 down to "cannot tell yet". Two-tailed would be more conservative; that is a team decision to be made before looking at the data.
- **Assumptions instead of data:** β=2 and `P_min` are stipulations. Cross-check the ranking with F1, F2 and F0.5. `balanced` reflects frequency, not cost. Do not read coefficients causally (Age and Pregnancies are related).
- **Narrow hypotheses and imputation:** Three classics bring little surprise, so deliberately fill "found along the way" (e.g. whether the share of missing values differs by Outcome). Median imputation with a large share of zeros creates artificial mass at the median, hence the sensitivity run without these columns.

## 10. Control questions from a coach
1. You combine `balanced` with your own threshold: Which of the two levers brings how much recall, and what does each cost in false alarms? How do you know that without using the test set?
2. Glucose from the 2-h tolerance test is almost the diagnostic criterion: What can your screening still do then that a doctor with this value does not already know, and where would this value already be available in practice?

> ⚠ **Correction K1:** According to the paper, `Outcome` is the onset of diabetes 1–5 years after an examination with a normal glucose test. Glucose is a legitimate predictor, not target leakage and not a diagnostic criterion for `Outcome`.

3. Why β=2 and not 1 or 3, and where does your `P_min` come from: from the data, from the use case or from gut feeling? What would the model ranking look like with a different choice?

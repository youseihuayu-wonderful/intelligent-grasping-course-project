# Team Experimental Protocol

This is the common protocol for the current course foundation (steps 1–3), not a claim that every proposed method has already been implemented. Read the [dataset definition](DATASET.md) and reproduce the [reference baseline](../README.md#install-validate-and-run) before starting model work.

## Proposed responsibilities

| Member | Responsibility | Expected contribution |
| --- | --- | --- |
| A | Data and shared evaluation | Maintain the frozen data contract, leakage checks, shared probability/metric interface, and comparison records. |
| B | Logistic Regression and LASSO | Reproduce the reference logistic model; investigate fold-local LASSO feature selection with honest method naming. |
| C | Random Forest | Develop and document a Random Forest route under the same inputs, folds, and metrics. |
| D | XGBoost | Develop and document an XGBoost route under the same inputs, folds, and metrics. |

Only the Logistic Regression reference baseline is supplied in this repository. These assignments do not imply that LASSO, Random Forest, or XGBoost have been implemented or executed here. Existing upstream implementations and results are learning references, not final student submissions.

## Freeze the common experiment

All teammates must use the same:

- Byte-identical **v1 CSV**, manifest, and artifact hashes; no row edits, relabelling, resampling, silent filtering, or mixing in older batches.
- Exact **11-feature allowlist** in manifest order and the same label definition (`success`: 1 = success, 0 = failure).
- Existing condition groups, upstream training/validation/test assignments, and all three supplied grouped training CV folds. Never regenerate a row-level split or let related trials cross a fold boundary.
- Probability and prediction schema, fixed decision threshold, metrics, and class conventions described below.
- Declared tuning budget and comparison procedure. Agree on search spaces, candidate counts, randomness, compute limits, and selection/tie-break rules before comparing a new set of methods. Report actual differences rather than calling unequal searches a controlled comparison.

The current reference baseline searches three `C` values (`0.1`, `1.0`, `10.0`) with negative log loss on the frozen CV folds and refits on training rows only. The pipeline is `DictVectorizer(sparse=False) → StandardScaler → LogisticRegression`, using `lbfgs`, `max_iter=5000`, `tol=1e-8`, and random state 5644. These settings describe this reference, not a requirement to apply inappropriate logistic hyperparameters to other model families.

CV manifest row values are global zero-based CSV indices. Consumers working on training-only arrays must remap them, not treat them as local positions. Nuisance draws are paired across controls within a condition; they are not independent globally seeded groups. Preserve the supplied group identities even when settled predictor poses differ slightly.

## Prevent leakage and test peeking

1. **Fit preprocessing inside each training fold.** Learn category vocabulary, scaling, imputation if a future justified method needs it, transformations, feature screening, and any calibration only from the relevant training data. Do not fit them once on the whole CSV or on training plus validation before CV.
2. **Fit any LASSO selector inside each training fold.** Select its strength and the downstream classifier's settings without seeing that fold's held-out labels. A feature list obtained from all rows before CV is leakage.
3. Use training CV for hyperparameter search. Use only training/validation evidence for development and model comparisons; log validation-informed decisions, since repeated validation inspection makes it development evidence rather than an untouched final assessment.
4. Refit the chosen reference pipeline on the training partition only, then produce validation probabilities. The current baseline neither fits on validation nor exports or scores test predictions.
5. Do not examine test errors, tune thresholds against test labels, select features using test data, choose a method using historical test rankings, or repeatedly score the test set. Keep it out of training, preprocessing, model selection, calibration, and debugging decisions.

**The v1 test partition is already published and inspected historical evidence, not a new blind or untouched final test.** Preserving it does not restore blindness. Current reports must state this limitation. A genuinely fresh final claim would need newly held-out conditions collected or reserved under a later predeclared protocol; that work is outside current steps 1–3. Do not promise fresh-test performance from the existing partition.

The subsequent [independent-test preregistration](INDEPENDENT_TEST_PROTOCOL.md) defines a separate development pilot and a fresh final batch. It does not collect them or reclassify the historical split as blind. A holds concrete final inputs/seeds outside Git; model artifacts/settings must be frozen before final collection, and predictions locked before labels are joined for scoring. Follow that protocol's budgets, exclusion rules and grouped uncertainty procedure for the new final comparison.

No identity, metadata, execution duration, diagnostic, trajectory, or outcome-derived column may be a predictor. See the [explicit exclusion list](DATASET.md#forbidden-predictors). `group_id` and `split` organize evaluation only.

## Shared evaluation contract

For each evaluated row, save:

```text
row_index,trial_id,group_id,split,success,p_success,predicted_success
```

- `p_success` is a finite probability of **positive class 1**, in `[0, 1]`. Select the classifier's probability column by `classes_ == 1`, not an assumed arbitrary column order.
- Predict class **1** when `p_success >= 0.5`; predict **0** otherwise. An exact `0.5` tie is success. Use this rule rather than an estimator's potentially different tie convention.
- Keep the threshold fixed at **0.5** for shared comparisons. Do not choose a better-looking threshold after inspecting validation or test results and silently replace the shared rule.
- Join predictions back by preserved row identity and retain group/partition metadata for audit, never as model input.

Report these metrics on the same validation rows:

| Metric | Convention and interpretation |
| --- | --- |
| Log loss | Binary probabilistic loss, lower is better; training-CV negative log loss is the reference search score. |
| Brier score | Mean squared error of `p_success` against label 0/1, lower is better. |
| ROC-AUC | Rank discrimination with success class **1** positive; higher is better. Not a calibration metric. |
| Balanced accuracy | Mean of recall for classes 0 and 1; higher is better. |
| Failure recall, class **0** | Actual failures predicted as failures / all actual failures. With class 1 positive, this is `TN / (TN + FP)`; higher is better. |
| Confusion matrix | Rows = true classes `[0, 1]`; columns = predicted classes `[0, 1]`, giving `[[TN, FP], [FN, TP]]`. Include counts and labels. |

Report both class supports so success and failure performance are explicit. If adding success recall, precision, or F1, name the class and definition; do not replace the shared metrics with an unlabelled “recall.” Undefined metrics must be represented explicitly with their reason, not fabricated as zero or silently omitted. Accuracy alone is insufficient for this imbalanced dataset. Predicted probabilities are model estimates, not guarantees of physical success or proof of calibration.

Always include the constant-probability reference based **only on training success prevalence**. Do not estimate its probability from validation or test labels. Read actual values from [the reference report](../results/logistic_baseline_v1/report.json); do not copy upstream historical numbers into a new-run report.

### LASSO is not L1 Logistic Regression

The original [`grasp_classifier_linear.py`](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/grasp_classifier_linear.py#L106-L138) uses **squared-loss `Lasso` as a feature selector**, retaining nonzero coefficients, followed by **L2 Logistic Regression**. Encoding, scaling, selector fitting, and classifier fitting belong inside each CV pipeline. Report selection strength, retained encoded features, classifier settings, and any zero-feature/intercept-only case honestly.

**L1-penalized Logistic Regression** instead optimizes logistic loss with an L1 penalty. It is a different method, not an interchangeable implementation of that squared-loss selector. Name whichever method is actually run. A raw LASSO regression prediction is neither constrained to `[0, 1]` nor a class-probability estimate; do not present it as `p_success`, claim it is calibrated, or disguise it by clipping. In the original selector route, the downstream logistic classifier supplies class probabilities.

## Reproducible handoff

1. Create a focused branch; record the intended method and agreed search budget before running comparisons.
2. Run the shared validation command and reproduce the reference workflow. Keep all outputs in a new, method-specific directory; never overwrite another member's run or the committed reference.
3. Keep a run record with the command, code revision, data/fold hashes, feature list, dependency versions, random states, search space, chosen parameters, threshold, and actual prediction/metric artifacts. Preserve warnings and failures; do not suppress numerical problems or report an unfinished run as successful.
4. Submit a reviewed change explaining what was reused, what changed, and what was actually executed. A comparison must use the same rows and metric implementation. Data/protocol changes require a coordinated new version, not one member's silent adjustment.
5. Consolidate validation evidence with its limitations. Better validation scores alone do not show improved executed grasping, and no method is guaranteed to outperform the baseline.

## Numerical reproducibility

The first local run with NumPy 2.2.6 completed with matrix-operation warnings from the Apple Accelerate backend. Its original code, dependency lock, outputs and warnings are retained under `records/numerical_warning_attempt_01/`; it is not the release baseline. The course environment pins NumPy 2.5.0, after the upstream Accelerate fixes described in [NumPy issue 29820](https://github.com/numpy/numpy/issues/29820). The training and prediction block treats convergence and runtime numerical warnings as errors rather than hiding them. The original simulation environment and the frozen source dataset are unchanged.

## Attribution, collaboration, and claim boundaries

Cite the original repository, immutable source commit, dataset version, relevant implementations, and reused results. Each member must explain their own contribution and understanding. Do not rebrand pre-existing code, AI-generated work, or another teammate's results as an independently invented algorithm. Follow the instructor's actual course rules on collaboration, source reuse, and AI assistance; disclose tools and assistance in the form the course requires, and do not assume public availability makes reuse automatically permissible.

The [upstream simulation guide](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/main/docs/SIMULATIONS.md) keeps Experiment 1's independent five-object grasp-and-hold baseline, Experiment 2's continuous storage release, and older classifier batches distinct. Experiment 2 is fixed-controller engineering evidence, not new training data, not a classifier-driven demonstration, and not 100% overall classifier success. Do not infer labels from a video, teleport objects, or substitute edited animation for physical execution. Sharing MuJoCo, Panda, and assets does not establish generalization when clutter, poses, or controllers change. No vision, real-robot, or sim-to-real claim is supported by this foundation.

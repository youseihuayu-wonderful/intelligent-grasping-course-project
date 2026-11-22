# Model training and delivery specification — v1

## 1. Start here: ownership and scope

Predict the probability of **single-object grasp-and-hold success** from the same 11 pre-grasp context/command inputs. This stage compares classifiers; it does not optimize physical controls, add simulator trials, or demonstrate faster executed grasping.

| Member | Work to complete | Handoff |
| --- | --- | --- |
| A | Maintain frozen data, review leakage/reproducibility evidence, run the common evaluator | Reviewed manifests, one aligned comparison, uncertainty and all error cases |
| B | Reproduce the old Logistic reference; develop the new Logistic route and LASSO-selector-plus-Logistic route | Two independently fitted pipelines and their actual run evidence |
| C | Develop Random Forest | One fitted pipeline and its actual run evidence |
| D | Develop XGBoost; pin its additional dependency before fitting | One fitted pipeline, environment lock and actual run evidence |

B/C/D can work **in parallel now** on the CSV. They do not need a running simulator or the private final plan. A reviews run plans before searches and reviews the resulting evidence before accepting comparisons. Each member must understand and disclose reused code/AI assistance according to course rules.

**Current implementation boundary:** only the existing Logistic Regression reference and the shared evaluator have been executed here. This specification supplies no trained LASSO, Random Forest or XGBoost, no fabricated predictions, and no new training CLI.

## 2. Freeze inputs before training

1. Start from the course repository, run `uv sync --locked`, then `uv run --locked python baseline.py --validate-only`.
2. Use only [frozen v1](DATASET.md): 2,560 training rows / 160 groups for fitting; 640 validation rows / 40 groups for development scoring. Preserve the CSV, split assignments, group IDs and all three training-CV folds.
3. Project every input through the exact **11-feature allowlist in manifest order**. The [feature definitions](DATASET.md#the-11-allowed-predictors) distinguish context from commands. Do not add IDs, seeds, diagnostics, durations, outcomes, invented volume, or trajectory features.
4. `baseline.load_dataset(data_dir)` returns `rows`, checked `audit` hashes, and `cv` already remapped to TRAIN-local positions. Build training arrays in filtered CSV order. Do not feed global CV row IDs into a training-only array or regenerate a row-wise split.
5. Keep the 320-trial development pilot separate. Never tune on or score the already-public historical test. Do not request/open concrete private final inputs or seeds.

The [machine-readable search plan](../protocols/model_training_v1.json) is the source of truth for candidate lists, fixed settings, ordering and edge-case rules. Record its **file SHA-256** and the actual source revision in every new run. This plan does not alter data, the existing baseline or the independent-test preregistration.

## 3. First comparison: fixed search budget

| Route / model ID | Candidate construction | Planned candidates | Maximum fold pipeline fits + selected TRAIN refit |
| --- | --- | ---: | ---: |
| Logistic / `logistic` | Search regularization strength `C` | 12 | 36 + 1 |
| LASSO + Logistic / `lasso_logistic` | Cross selector `alpha` with downstream `C` | 12 | 36 + 1 |
| Random Forest / `random_forest` | Cross depth, features per split and minimum leaf size | 12 | 36 + 1 |
| XGBoost / `xgboost` | Cross learning rate, depth and minimum child weight | 12 | 36 + 1 |

These are **planned budgets, not completed fits**. Each pipeline fit can contain several estimator fits. Twelve attempted candidates is not equal compute, expressiveness or effective search coverage. Report attempted/eligible/completed counts, failures, CV and refit elapsed time, CPU/hardware, thread settings and versions. B owns two routes, not the same workload as a member owning one.

The committed reference searches only three `C` values. Keep it as a separately named reference; it is **not** the new twelve-candidate Logistic run. Do not combine both in one evaluator call under the same ID: run the old-reference comparison separately or package the real old files under the explicit ID `logistic_reference`.

No adaptive extra candidates, best-looking-seed selection, dataset-level class balancing/resampling, class reweighting, fitted probability calibration or early stopping is included in this first plan. Random Forest's declared internal bootstrap is part of that algorithm, not additional dataset balancing. Use seed 5644, CPU, one search job and one estimator job where supported. Record BLAS/OpenMP thread settings; this does not promise identical CPU time across machines.

### Pipelines and method naming

- All routes: fit dense `DictVectorizer` only on the current training fold's allowed feature dictionaries. It encodes the categorical family without ordinal integers. Do not build a vocabulary from validation/test rows.
- Linear routes: fit `StandardScaler` on the encoded columns inside that fold, then the classifier. The plain Logistic route uses L2 regularization. Raw coefficient magnitudes across differently scaled methods are not directly comparable.
- LASSO route: fit squared-loss `Lasso` on that fold's standardized features and binary labels; retain encoded columns by the plan's fixed absolute-coefficient threshold, excluding its intercept; fit L2 Logistic Regression on the selected columns. Both `alpha` and downstream `C` are tuned in the same frozen CV. Save selected names/counts and coefficients for each fold and the final TRAIN refit. This is **not L1 Logistic Regression**, and LASSO regression outputs are not `p_success`.
- Tree routes: fold-local encoding, no scaler. Random Forest supplies class-1 probabilities. XGBoost uses the fixed binary-logistic objective and CPU histogram trees. Its fixed tree count avoids an undeclared early-stopping holdout.
- If LASSO selects zero columns on any fold, mark that candidate explicitly ineligible; record every fold's status, not a partial-fold mean. Do not secretly restore all columns or insert a constant model. No replacement candidates are allowed. A zero-column selected final refit or no eligible candidate means no completed route; preserve evidence and review a new plan rather than invent a result.

`ConvergenceWarning`, numerical `RuntimeWarning`, nonfinite scores and unexpected fit errors fail the run. Preserve logs and diagnose them; do not quietly use `error_score=0`, suppress warnings or drop inconvenient folds. Any required change to solver limits, grids or preprocessing must be versioned and reviewed before another comparison; disclose prior attempts.

### Selection and refit order

1. Enumerate the plan's Cartesian grid by lexicographically sorted parameter names, preserving value-list order; the rightmost parameter changes fastest. Save zero-based `candidate_index`, the full parameter tuple and each frozen fold's score/status.
2. For each eligible candidate, minimize the **unweighted arithmetic mean of the three held-out-fold log losses**. Maximizing their mean negative log loss is equivalent. An exact mean-score tie selects the lowest candidate index. Do not switch to validation accuracy to break a tie.
3. Refit the selected **whole pipeline** on the 2,560 TRAIN rows only. Never refit on TRAIN + validation for this comparison.
4. Emit probabilities on all 640 validation trials. Select the `predict_proba` column by `classes_ == 1`; use finite, unrounded values in `[0,1]`. Predict success for `p_success >= 0.5`, including ties. Preserve original identity fields.
5. Record every validation inspection and any decision it influenced. Repeated development comparisons remain development evidence, not an untouched estimate of generalization. No method is promised to beat the reference.

## 4. Reproducible environment and source

Use the shared Python/core dependency lock. Random Forest, LASSO and Logistic use the existing scikit-learn installation. **XGBoost is not in the current root `uv.lock`.** Before training, D must provide an exact additional version and a reproducible method-specific environment/lock compatible with the common core; A reviews it. Do not silently replace the shared lock or use an unrecorded install. This delivery does not install or runtime-validate XGBoost.

Commit/review the actual training source before fitting; record that source revision, all imported project helper-file hashes, dependency-lock hash and search-plan hash. `code_sha256` must hash the exact training entry file supplied to the packager; it does not cover hidden helper changes. Preserve source snapshots and local-change disclosures where necessary. The artifact must include fitted encoding/scaling/selection and the classifier so a reviewer can reproduce the same probabilities from only allowed inputs. Never load untrusted pickle/joblib files merely because a SHA-256 is present.

## 5. Required delivery and run record

Use a new method/run directory; never overwrite the reference or another member's output. Keep all real files accessible in the shared layout (a reviewed branch/PR is sufficient while work is in progress).

| Deliverable | Required evidence | Checked by |
| --- | --- | --- |
| Training source and complete environment | Actual entry file, imported project helpers, revision/hashes, exact run command, dependency lock and versions | A/source review; entry-file hash by packager |
| Run report | Compatible JSON declarations below, actual selected parameters, plan/data/fold/source identity | Packager plus A's training-history review |
| CV results | Every planned candidate and all three fold losses/statuses, eligibility, selection order, warning/failure history, timings | A/source and run-record review |
| Fitted model artifact | Actual complete fitted pipeline; record selected encoded features for LASSO | Bytes bound by packager; trusted replay requires separate review |
| Validation predictions | All 640 authoritative trials, exact seven-column schema | Packager/evaluator |
| Training log and short method note | Raw stdout/stderr, actual attempts, reuse/contribution, limitations, validation-informed changes | A/reviewer |
| Submission manifest | Four actual files bound to hashes and portable relative paths | `evaluation_submission.py create` |

The **automatic minimum** is four files: training entry source, report, model artifact, predictions. Extra CV/environment/history evidence above is a **team-review requirement**, not something the current CSV loader can prove automatically.

The producer report follows the existing [reference JSON](../results/logistic_baseline_v1/report.json):

- `protocol`: `fit_partition="train"`, `score_partition="validation"`, the full ordered raw `features`, `decision_threshold=0.5`, and a nonempty `selection` description.
- `audit.sha256`: authoritative `dataset.csv`, `split_assignments.csv`, `cv_manifest.json` hashes copied from the checked package audit; a supplied `manifest.json` hash is checked too.
- `code_sha256`: SHA-256 of the actual supplied entry source; `versions`: nonempty string-valued package versions; `selection`: nonempty actual selected hyperparameters.
- Add the plan hash, source/helper/lock identities, search space/order, per-fold results, seed, timing, selected-feature records and development history to the report or linked tracked records. Do not substitute planned values for measured execution facts.
- Omit `validation` metrics if not calculated with the shared implementation. If supplied, the entire common schema must match the evaluator within tolerance. Keep full precision in JSON/CSV; round presentation tables only.

Prediction columns must be exactly, in this order:

```text
row_index,trial_id,group_id,split,success,p_success,predicted_success
```

Use the original CSV's **global** `row_index`, not a new 0–639 validation-local number. CSV row order may differ, but every authoritative validation trial must occur exactly once. The supplied success label is checked against frozen truth; it is never a replacement for that truth.

## 6. Package and evaluate with existing tools

This runnable example uses the **real old reference**, not an unimplemented model. Both outputs must be new; choose fresh names if already present:

```sh
uv run --locked python evaluation_submission.py create \
  --model-id logistic_reference \
  --predictions results/logistic_baseline_v1/validation_predictions.csv \
  --run-report results/logistic_baseline_v1/report.json \
  --training-code baseline.py \
  --model-artifact results/logistic_baseline_v1/model.joblib \
  --output submissions/logistic_training_guide_local.json

uv run --locked python evaluate_models.py \
  --submission submissions/logistic_training_guide_local.json \
  --reference logistic_reference \
  --output results/training_guide_comparison_local
```

For a newly trained route, use its plan model ID and its own real four files. Repeat `--submission` once for every actual model when running the comparison; choose an explicit reference, normally the **new** `logistic` route. Do not submit old Logistic probabilities as Random Forest/XGBoost/LASSO predictions. All referenced files must accompany a shared manifest; hashes alone cannot retrieve absent files.

See [UNIFIED_EVALUATION.md](UNIFIED_EVALUATION.md) for the authoritative schema, command options, metrics and outputs. The scorer recomputes log loss (primary), Brier score, ROC-AUC, balanced accuracy, failure-class-0 recall and labelled confusion matrices; it automatically includes the **TRAIN-only** constant prior. It saves all errors and 2,000 shared family-stratified whole-group bootstrap draws with paired contrasts. It does not load arbitrary model artifacts or audit actual training execution.

## 7. A's acceptance gate and final-test boundary

- [ ] Frozen data/folds and the reviewed plan/source identities match; no pilot append, test-guided change or extra input.
- [ ] Source and complete records show fold-local preprocessing/selection, declared tuning budget, all candidate/fold outcomes and TRAIN-only final refit.
- [ ] Actual model, environment, warnings, unsuccessful attempts and selected settings/features are preserved; no placeholders or renamed learner outputs.
- [ ] Four-file manifest passes the strict loader; all 640 rows align; class-1/tie conventions and common metrics agree.
- [ ] A reviewer replays the probability mapping from the reviewed trusted pipeline/environment and compares every validation probability. Do not extend the existing hash-restricted reference verifier to arbitrary files without a trust review; it accepts only the original known artifact.
- [ ] Comparison uses actual submissions with declared budgets/hardware and descriptive development uncertainty, not validation-based guarantees or causal feature claims.
- [ ] All four intended pipelines, exact artifact/source/environment hashes, selected parameters and thresholds are frozen **before final collection**. Missing routes are incomplete work, not replaced by the constant prior.

Only after that model-freeze gate should A consider the [independent final-test protocol](INDEPENDENT_TEST_PROTOCOL.md#6-team-handoff-and-freeze-order): collect the fixed batch, check integrity, lock predictions before joining labels, then perform the separately planned final scoring. The current validation-only evaluator has **no final-test mode**. This specification neither launches final collection nor claims that final scoring is already implemented or executed. Public v1 historical results and validation comparisons do not become a blind test by renaming them.

## 8. Specification verification record

The [verification log](../records/model_training_spec_check.log) records frozen-data/CV consistency, deterministic grid cardinality/order, the available scikit-learn estimator parameter checks, local documentation links, and a real execution of the packaging/evaluation example with temporary outputs. These checks do not fit any new model or runtime-validate XGBoost. The latter remains D's explicitly assigned environment/training task, not a completed result.

# Unified validation evaluation

## Purpose and scope

Use one entry point to compare submitted success probabilities against the **same frozen v1 validation truth**, with one metric implementation and reproducible condition-group uncertainty. This is Member A's shared evaluation infrastructure, not training of the other members' classifiers.

The evaluator uses 640 validation rows in 40 condition groups. It does not score historical test rows, the separate 320-trial development pilot, or the sealed independent final test. All data/fold hashes and the 11-feature allowlist remain unchanged. The classification decision is success when **`p_success >= 0.5`**, including exact ties.

## What each member submits

Provide four real files, bound by a portable JSON submission manifest:

| File | Required contents / purpose |
| --- | --- |
| Predictions CSV | Exactly one row for every frozen validation trial, with the agreed seven-column schema |
| Training run report | Actual feature/partition declarations, data/fold hashes, selected hyperparameters, software versions and training-code hash |
| Training source | The exact source snapshot matching the run report's SHA-256 |
| Model artifact | The actual fitted artifact, bound by SHA-256; the generic evaluator never deserializes it |

Prediction columns, in order:

```text
row_index,trial_id,group_id,split,success,p_success,predicted_success
```

`p_success` means **positive class 1**, not the failure probability. Row order can differ: the loader joins by authoritative `trial_id`, verifies the global row index, group, split and supplied label, then realigns probabilities to the frozen order. Supplied labels never replace ground truth. Missing/extra/duplicate IDs, train/test/pilot rows, malformed headers/cells, nonfinite or out-of-range probabilities and inconsistent threshold decisions are rejected.

The training report uses the existing [reference report](../results/logistic_baseline_v1/report.json) layout. Required fields are:

- `protocol`: `fit_partition="train"`, `score_partition="validation"`, `features` equal to the full raw allowlist in order, `decision_threshold=0.5`, and a nonempty `selection` description.
- `audit.sha256`: the exact `dataset.csv`, `split_assignments.csv` and `cv_manifest.json` hashes; a supplied `manifest.json` hash is checked too.
- `code_sha256`: the actual supplied training-source hash.
- `versions`: nonempty software/version declarations.
- `selection`: actual selected hyperparameters, not an empty stub.

The report may omit `validation` metrics because the shared evaluator computes them. If that field is supplied, it must contain the exact common metric schema and agree with recomputed values within numerical tolerance; stale or incorrectly reported scores are rejected. Do not round the stored machine-readable values just to format a table.

A LASSO-selector route still declares the common **11 raw predictors**. Encoded-feature selection belongs inside training folds and is reported separately; it does not justify reading outcome/identity columns or changing the shared input contract. Squared-loss LASSO selection followed by Logistic Regression is not L1-penalized Logistic Regression.

## Package the existing real baseline

From the course checkout after `uv sync --locked`:

```sh
uv run --locked python evaluation_submission.py create \
  --model-id logistic \
  --predictions results/logistic_baseline_v1/validation_predictions.csv \
  --run-report results/logistic_baseline_v1/report.json \
  --training-code baseline.py \
  --model-artifact results/logistic_baseline_v1/model.joblib \
  --output submissions/logistic_baseline_v1_local.json
```

This validates all evidence before creating the manifest. Paths are stored relative to the manifest, and every file is hashed. Keep the manifest and referenced files together in the repository layout when sharing them. Existing/symlink outputs and protected data/protocol/reference-run destinations are refused.

For the other routes, use the member's actual four files and a distinct model ID such as `lasso_logistic`, `random_forest` or `xgboost`. Do not rename baseline predictions and describe them as results of another learner. No placeholder submissions or invented scores are supplied here.

## Run the unified comparison

The committed reference handoff is [submissions/logistic_baseline_v1.json](../submissions/logistic_baseline_v1.json). Run:

```sh
uv run --locked python evaluate_models.py \
  --submission submissions/logistic_baseline_v1.json \
  --reference logistic \
  --output results/unified_validation_v1_local
```

Repeat `--submission` for each additional **real** submitted model. IDs must be unique. `--reference` must name a supplied model or the automatic `constant_train_prior`; it is chosen explicitly, not by selecting the best observed score. If omitted, the reference is the always-present training-prior constant.

The constant success probability is computed from **training labels only**, never validation prevalence. It is an intentional baseline, not a trained additional classifier or a replacement for missing methods.

There is no test/final mode or threshold-tuning option. A complete run refuses invalid submissions rather than silently comparing a convenient subset. The output directory must be new and outside frozen data, protocols, submissions and the original reference run. A custom data directory must also pass the pinned v1 loader; this is not a way to score an arbitrary CSV or mix pilot rows.

## Shared metrics and uncertainty

All point, family and resampled metrics call **`baseline.validation_metrics`**; there is no second competing implementation:

| Output key | Meaning / direction |
| --- | --- |
| `log_loss` | Primary probabilistic loss; lower is better; valid exact 0/1 probabilities use the common float64 epsilon clipping convention |
| `brier` | Brier score, the mean squared probability error; lower is better |
| `roc_auc` | Class-1 ranking discrimination; higher is better, not a calibration measure |
| `balanced_accuracy` | Mean class-0/class-1 recall when both classes are present |
| `failure_recall` | Recall of actual class-0 failures: `TN / (TN + FP)` |
| Confusion matrix | True row labels `[0,1]`, predicted column labels `[0,1]` |

The five scalar metrics use **2,000** paired bootstrap replicates with seed **5644**. Resample whole condition groups within each target family, preserving the original number of groups per family and carrying every selected group's rows together. The **same draws** are applied to all models and candidate-minus-reference contrasts. This preserves within-condition control/noise pairing instead of treating 640 rows as independent conditions.

The report includes 95% percentile intervals using NumPy's linear quantile convention, metric directions and defined/undefined replicate counts. If a resample lacks class support, the affected metric remains null. Quantiles based on the remaining defined resamples are explicitly conditional; an entirely undefined metric has null interval bounds. Undefined values never become convenient zeroes.

Negative candidate-minus-reference differences favour the candidate for log loss/Brier; positive differences favour it for the other scalar metrics. The exact whole-group multiplicities are saved in `bootstrap_group_weights.csv`, with source group order, generator/version and hashes for reproduction.

These are approximate, descriptive **development-validation** intervals with only a finite number of condition groups. They are not an untouched final assessment, a causal claim, a multiple-comparison-adjusted significance test or guaranteed population coverage. Unequal tuning budgets/hardware must be disclosed and cannot be cured merely by using a shared scorer.

## Error and calibration diagnostics

Every model gets all misclassified validation cases, not just selected favourable examples:

- `missed_failure`: actual failed grasp predicted successful.
- `missed_success`: actual successful grasp predicted failed.

Cases include row identity, the allowed context/command values, model probability, the common threshold decision, per-case clipped log loss, and an immutable link to the original physical trial record. The error summary additionally lists the ten highest-loss errors with deterministic tie ordering. A predictive error is not a newly diagnosed physical root cause.

Calibration uses ten predefined equal-width probability bins: left-closed/right-open, except the final bin includes 1. Empty bins have null mean probability/outcome frequency, not fabricated zero performance. These are reliability diagnostics, not a fitted calibrator or proof of calibrated probabilities.

## Authoritative artifacts

The actual existing-reference evaluation is stored under [results/unified_validation_v1](../results/unified_validation_v1):

| Artifact | Purpose |
| --- | --- |
| [summary.md](../results/unified_validation_v1/summary.md) | Readable point-metric comparison with scope limits |
| [report.json](../results/unified_validation_v1/report.json) | Inputs/hashes, canonical support, common metrics, declared training provenance, evaluator code/software identities and limits |
| [comparison.csv](../results/unified_validation_v1/comparison.csv) | Uniform machine-readable model comparison |
| [uncertainty.json](../results/unified_validation_v1/uncertainty.json) | Whole-group metric intervals and paired contrasts |
| [bootstrap_group_weights.csv](../results/unified_validation_v1/bootstrap_group_weights.csv) | Every resampling draw, not just a seed claim |
| [confusion_matrices.json](../results/unified_validation_v1/confusion_matrices.json) | Labelled matrices for all evaluated methods |
| [family_metrics.json](../results/unified_validation_v1/family_metrics.json) | Per-family metrics with supports and undefined reasons |
| [calibration_bins.json](../results/unified_validation_v1/calibration_bins.json) | Fixed-bin reliability diagnostics |
| [error_summary.json](../results/unified_validation_v1/error_summary.json) and [errors/](../results/unified_validation_v1/errors) | All model error cases and highest-loss diagnostics |
| [predictions/](../results/unified_validation_v1/predictions) | Canonical, aligned probabilities and threshold decisions for reproduction |

Execution evidence: [focused regression tests](../records/unified_evaluation_tests.log), [submission creation](../records/unified_submission_create.log), and [actual unified run](../records/unified_evaluation_run.log).

### Observed verification

- All **32 focused regression tests passed**, covering malformed/misaligned predictions, wrong partitions/labels/probabilities, threshold ties, provenance drift, whole-group pairing, missing class support and output protection.
- The real existing Logistic Regression and training-prior reference were scored on all authoritative validation rows, with **2,000** shared family-stratified group draws saved. Recomputed Logistic Regression metrics exactly reproduce its original stored report.
- The separate trusted-artifact replay checked every saved reference probability and the actual encoded feature/class mapping; the maximum absolute probability difference was **zero**. This replay is restricted to the known reference hash, not arbitrary pickle/joblib inputs.
- The [initial regression attempt](../records/unified_evaluation_tests_attempt_01.log) exposed portable-path resolution when the new submission directory did not exist. The path handling was corrected before any real submission/evaluation output was created; the regression was retained, not weakened or removed.

These observations verify the implemented checks and this real reference run. They are not a promise that future members' unreviewed training histories or all possible software inputs are 100% correct.

## What is verified, and what is not

Verified checks include frozen data/fold identity, validation-only complete row coverage, authoritative labels, numeric domains, threshold conventions, matching **declared** feature/fit contracts, exact supplied artifact bytes, recomputed scores and deterministic shared resampling. The producer's stated protocol, software and selected hyperparameters are retained as declarations, not silently promoted to an execution audit.

**A CSV and a hash-bound report cannot prove arbitrary training code actually fitted only training rows, performed fold-local preprocessing, or emitted the claimed probability column.** Hashing proves byte identity, not honesty or correctness of the entire learning process. Review training code and run records before making stronger claims about leakage-free or fully controlled model development. The generic evaluator deliberately never unpickles arbitrary model artifacts.

For the one known reference, a separate verifier permits deserialization **only after matching the exact trusted artifact SHA-256**. It replays the pipeline on the allowed validation features, checks encoded inputs/class mapping and compares every emitted probability with the saved CSV:

```sh
uv run --locked python verify_reference_predictions.py \
  --submission submissions/logistic_baseline_v1.json \
  --output records/reference_prediction_replay_local.json
```

The [actual trusted-reference replay record](../records/reference_prediction_replay.json) and [command output](../records/reference_prediction_replay.log) are separate from the generic scorer's no-deserialization guarantee. This is not a general instruction to load untrusted joblib/pickle files, and it does not certify the future members' training runs.

## Team next action

Members B/C/D should develop and tune their assigned methods under the same frozen data/folds and shared training rules, then submit actual artifacts/reports/predictions. A can run one comparison over all submitted manifests without changing metric code. The current smoke contains only the real Logistic Regression and training-prior reference; Random Forest, XGBoost and LASSO results are not implied to exist.

Start with the [training and delivery specification](MODEL_TRAINING_AND_DELIVERY.md), including its predeclared route budgets and A's acceptance gate. It distinguishes automatically checked submission fields from source/CV/environment evidence that still requires review.

Keep the 320-trial pilot separate unless the whole team agrees on a new coordinated development-data version. Do not collect/open the sealed independent final test until model freeze. This delivery does not alter v1, the pilot, the physical simulator or the final-test plan.

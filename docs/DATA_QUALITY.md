# Dataset v1: recorded-evidence quality audit

## Purpose and scope

This audit answers: **Does each frozen CSV row agree with its original planned input, executed-trial record, saved trajectory and fixed partition?** It does not ask whether a video looks successful, train a classifier, collect new data or change any label.

The source is the five-family, 3,840-row **single-object grasp-and-hold** collection, not the continuous drawer-storage demonstration. The course CSV, original train/validation/historical-test assignments and three grouped training folds remain frozen. See [the feature and label contract](DATASET.md).

## Actual findings

The full recorded-evidence audit passed. These are observed results of this run, not promised physical guarantees:

| Item | Observed result |
| --- | ---: |
| CSV rows audited | 3,840 |
| Recorded successes / failures | 1,569 / 2,271 |
| Source artifacts checked against the immutable snapshot | 7,693 |
| Saved trajectory samples inspected, including intentional terminal duplicates | 2,632,650 |
| Maximum predictor difference from independently derived pre-grasp inputs | 0 |
| Exact complete-physical-input duplicates | 0 |
| Observable duplicate vectors crossing splits | 0 |
| Observable feature vectors represented twice | 1,920 |
| Repeated observable vectors with conflicting labels | 391 |
| Targeted corruption/regression tests | 39 passed |
| Training examples saved as case studies | 14 |

All existing condition groups and frozen CV boundaries passed their checks. Every row has an original trial and a readable, snapshot-matched trajectory; no label was changed. The source reported no numerical-warning trials, and every exported raw row was checked for an empty warning list.

**The conflicting observable labels are an important modeling finding, not automatically a data error.** The two trials share the 11 exposed predictors but have distinct planned latent execution-error draws. Full physical inputs are not duplicates. A deterministic feature-only classifier cannot assign both observed labels correctly to one identical feature vector; the intended target is a conditional success probability, not a promise of perfect classification. These finite-sample conflicts do not estimate an exact population Bayes error.

Recorded controller diagnostics for the 1,451 failed training trials:

| Controller diagnostic | Training failures |
| --- | ---: |
| Lost bilateral contact during lift | 611 |
| Stable bilateral contact not established | 344 |
| Target-pose timeout | 237 |
| Hold postconditions not met | 208 |
| Physical predicates lost after controller completion | 51 |

These are recorded failure messages, not proven physical root causes. In addition, 64 successful training trials contain earlier counted collision episodes, so grasp success must not be described as a collision-free certificate.

**Decision:** v1 passes the implemented recorded-evidence checks and remains suitable for developing the course classifiers under its documented simulation assumptions. This audit does not establish sample sufficiency, classifier superiority, continuous unsampled contact or real-robot validity.


The authoritative outputs are:

| Artifact | What it provides |
| --- | --- |
| [Audit report](../records/data_quality_v1/report.json) | Actual checks, counts, input coverage, training-only diagnostics, software versions and limitations |
| [Per-trial audit CSV](../records/data_quality_v1/trial_audit.csv) | A result for every dataset row, with raw-record/trace paths and SHA-256 hashes |
| [Evidence inventory](../records/data_quality_v1/evidence_manifest.json) | Exact source-file sizes, SHA-256 and immutable upstream Git blob identities |
| [Training case studies](../records/data_quality_v1/training_cases.json) | Deterministically selected successful/failed training examples with evidence and final predicates |
| [Actual audit command output](../records/data_quality_run.log) | Progress and completion of the full audit |
| [Corruption regression checks](../records/data_quality_tests.log) | Results of targeted tests for wrong labels, inputs, timing, partitioning and evidence |

## What is checked

### 1. Integrity, completeness and traceability

- Validate the course dataset, split assignments and frozen CV against their pinned hashes and structural contracts.
- Require a one-to-one correspondence between planned trial IDs, aggregate raw records and CSV rows. Missing, duplicated or extra trials fail the audit.
- For every row, read both its individual raw JSON and compressed trajectory, and compare the individual record to the aggregate record.
- Match source artifact bytes against upstream commit [`7a2c5642113d0d3b9c5b5d5205feda843b2c043a`][source]. Changed/missing evidence fails; files are not silently skipped or repaired.
- Independently reproduce original physical-input fingerprints and reconcile the old inclusion decisions, split manifest and collection summary.
- Match frozen source implementations against recorded collection-environment hashes where those hashes exist. The original environment did not record `grasp_classifier_data.py`; its frozen Git identity is preserved instead. A missing historical runtime hash is not reported as a matched one.

An unchanged hash proves byte identity, not that the underlying physical model or label is correct.

### 2. Predictor provenance and configured physics

- Independently recompute all 11 allowed predictors. In particular, x/y and quaternion-derived yaw must come from the **settled pre-execution pose**, not the nominal reset pose or final pose.
- Require exact agreement with the planned raw settings/metadata, and reconcile exported labels and diagnostic durations with their raw records.
- Compare prescribed mass to recorded compiled mass; check positive recorded inertia and geometry extents.
- Check recorded object/finger friction settings, full friction vectors, contact dimensions, MULTICCD flag, and observed finger-target friction/contact dimensions.
- Check that recorded grasp offsets equal the scaled family profile plus the explicitly excluded latent execution error.

These are checks of stored configuration/compiled diagnostics. Collision geometry is not regenerated, and commands are not relabelled as measured TCP speed, grip force or achieved opening.

### 3. Recorded label and sampled-trajectory consistency

For every trace, check finite vectors/unit quaternions, timestamps, sampling gaps, legal phase/status transitions, initial/final pose agreement, execution duration and recorded phase metadata. The recorder intentionally appends an identical terminal sample twice; that duplicate is allowed, but contradictory same-time records are rejected.

A positive label must agree with terminal controller success, an independent observer timer of at least two simulation seconds and the recorded physical predicates. The observer timer must grow no faster than elapsed time, remain zero when ineligible, and agree with saved samples in its final hold window. The controller's own `verified_hold` predicate is distinct from the independent two-second observer timer.

**Important reference detail:** the controller captures its relative-position stability reference before a physics step, while the first forced hold sample is saved afterward. The exact reference is not stored. Consequently, displacement from the first saved stable sample is descriptive, not a remeasurement of the controller's one-centimetre test. The audit only applies the necessary bound that two positions each within one centimetre of one fixed reference cannot be more than two centimetres apart.

### 4. Sampling coverage, duplicates and leakage

- Reconstruct label-independent condition group IDs from nominal context and condition IDs; require condition/group bijection and the original split.
- Verify all 48 scale × family-specific mass × friction combinations per family, the declared family/split group counts, and nominal x/y/yaw Latin-hypercube strata.
- Verify eight distinct command configurations and two paired nuisance draws per condition, shared biases across controls within each draw, and coverage of the 54-configuration command pool across each family.
- Reject identical complete physical settings plus initial-state fingerprints repeated across trials. Also check the frozen split/CV group boundaries.
- Report repeated **observable** feature vectors separately. They can legitimately have different labels because latent execution error is intentionally not a predictor; such differences are not automatically corrupted labels.

Coverage of 54 configurations across a family is not evaluation of all 54 at every object condition. Sixteen trials in one condition are correlated observations, not sixteen independent conditions.

### 5. Failure diagnostics and examples

Exploratory outcome/feature summaries and case selection use **training rows only**. Validation/test labels are read only for integrity and duplicate consistency, not to select models, tune thresholds or mine predictive performance.

The original `failure_kind` gives any recorded collision episode precedence over its phase-based category. A recorded collision may co-occur with a contact-loss or motion-timeout diagnostic; it is not automatically the proven cause, a robot-obstacle impact or a verified object drop. The report therefore separates normalized English **controller diagnostics** from recorded heuristic failure kinds. Successful rows can also have earlier collision episodes; a positive grasp label is not a collision-free certificate.

Case selection is reproducible: first training success/failure per family, supplemented by the first training occurrence of each controller failure diagnostic. This is diagnostic coverage, not random sampling or a failure-prevalence estimate.

## Reproduce the audit

The audit uses the course lockfile and Python code, Git, and the original evidence checkout. It does **not** import MuJoCo, run the robot, fit classifiers, or require an API key. The original evidence remains upstream rather than duplicating hundreds of megabytes in this lightweight repository.

If the original simulation repository is not already available:

```sh
git clone https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms.git ../grasp-simulation-evidence
```

From this course repository:

```sh
uv sync --locked
PYTHONPATH=tests uv run --locked python -m unittest test_audit_trace test_audit_coverage test_data_quality
uv run --locked python audit_data_quality.py \
  --source-repo ../grasp-simulation-evidence \
  --output records/data_quality_v1_local
```

Supply your actual existing simulation-checkout path instead if appropriate. The frozen source commit must exist in its Git object database, and its recorded evidence must be present with the original bytes. A shallow checkout without that commit is insufficient. Core source text is read from the frozen Git blobs, not assumed to match a later working-tree revision.

The output path must be new. Existing files/directories, symlinks and output paths inside the frozen data directory are refused. No source data, split, label, historical output or ZIP is overwritten. Per-trial nested summaries in the audit CSV use JSON, not executable Python expressions.

## What this audit cannot certify

The physical observer was evaluated every **0.002 simulation seconds**, but stored trajectories retain approximately **0.020-second samples**, with additional phase/terminal samples. A short unlogged event cannot be independently excluded from these sampled trajectories. Contact/stability booleans also share the controller's physical predicates; full simulation state and contact-force history are not retained here.

Accordingly, a passing audit means **all implemented checks found the saved evidence internally consistent**. It is not a 100% guarantee of uninterrupted contact, simulator correctness, real-world grasp success or classifier performance. A separate physical rerun or newly recorded per-step evidence is needed for stronger verification.

The five families are rigid simulator proxies with associated grasp/contact profiles; their differences are not isolated causal shape effects. Mass/friction are simulator inputs, not deployment estimates. Reported training associations do not establish causation.

The historical test set and earlier results were already public and inspected. This audit does not make it blind again. A future final generalization claim requires a separately predeclared, newly held-out evaluation set. Classifier tuning and the other team members' model development can continue on the existing training/validation data without waiting for that collection.

[source]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/grasp_classifier_five_objects_20261121

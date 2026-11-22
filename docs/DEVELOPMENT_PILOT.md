# Development pilot v1: physical execution and evidence

## Purpose

This is execution of the frozen **320-trial development pilot**, not the private final test and not a new classifier comparison. Its 20 nominal condition groups cover five known families, four property corners per family, eight command configurations and two paired execution-error draws per group. The [preregistration](INDEPENDENT_TEST_PROTOCOL.md) is unchanged.

The question is whether the original simulator can execute those conditions and produce complete, auditable success/failure evidence. A successful **data-quality check** does not mean every physical grasp succeeds. Ordinary failed grasps must remain valid negative examples.

## Observed results

The full pilot and independent evidence-quality gates passed. All planned attempts produced verified labels; there were no unusable attempts, audit issues, MuJoCo numerical-warning trials or Python numerical-warning trials.

| Family | Trials | Successful grasps | Failed grasps | Pilot success fraction |
| --- | ---: | ---: | ---: | ---: |
| Bottle | 64 | 23 | 41 | 35.94% |
| Bowl | 64 | 24 | 40 | 37.50% |
| Glass tumbler | 64 | 26 | 38 | 40.63% |
| Pen | 64 | 45 | 19 | 70.31% |
| Spoon | 64 | 22 | 42 | 34.38% |
| **Total** | **320** | **140** | **180** | **43.75%** |

The independent auditor checked **2,205,159 contiguous 2 ms physics-step records** and **221,821 original sparse samples**, including the source recorder's intentional terminal duplicates. All original labels agreed with the reconstructed predicates and two-second acceptance clock. All 20 focused regression tests passed.

The separate CSV cross-check matched the original frozen feature extractor with maximum predictor difference **zero**. No actual observable feature vector, complete physical input, trial ID or condition group overlapped historical v1 evidence. All **140 unique runtime input files** matched their source fingerprints, and protected v1/design files remained unchanged.

Physical collection with three workers and dense logging took **756.91 seconds** (12.62 minutes). This is measured collection wall time, not robot task duration, an optimized sampling benchmark or total publication time; independent audit/archive processing took an additional 46.05 seconds.

The verified archive contains **1,290 logical files**, with every member checked byte-for-byte against the inventory. Its exact compressed size and SHA-256 are in `archive.json`. A start-time snapshot deliberately remains labelled `started` in the raw `run_manifest.json`; completion is recorded separately in the collection summary and final audit report, not rewritten over the original manifest.

**Conclusion:** the pilot demonstrates that the collection pipeline can produce genuine positive and negative labels with complete, independently auditable discrete-time evidence under these simulator assumptions. It does not establish classifier quality, statistical superiority or real-robot performance. The private final test was neither opened nor executed.


## Evidence and downloads

| Artifact | Purpose |
| --- | --- |
| [Pilot report](../results/development_pilot_v1/report.json) | Actual counts, quality status, class/family supports, source fingerprints and limitations |
| [Development-only dataset CSV](../results/development_pilot_v1/dataset.csv) | Verified labelled trials, with the original 11 predictors and traceable IDs |
| [Per-trial audit CSV](../results/development_pilot_v1/trial_audit.csv) | Individual audit verdicts, native-record and trace hashes, dense acceptance measurements |
| [All attempt dispositions](../results/development_pilot_v1/attempts.json) | Every planned attempt, including any explicit unusable/audit-error records |
| [Audit issues](../results/development_pilot_v1/audit_issues.json) | Explicit problems rather than silent omissions |
| [Independent export/overlap verification](../results/development_pilot_v1/export_verification.json) | CSV predictors versus the original extractor, labels versus raw outcomes, and actual historical-state/identity separation |
| [Case studies](../results/development_pilot_v1/case_studies.json) | First verified success/failure per family, when present; not random prevalence samples |
| [Runtime before](../results/development_pilot_v1/runtime_before.json) / [after](../results/development_pilot_v1/runtime_after.json) | Physical source/asset hashes and actual dependency versions |
| [Archive identity and download](../results/development_pilot_v1/archive.json) | Lossless raw-evidence archive URL, size and SHA-256 |
| [Logical evidence inventory](../results/development_pilot_v1/evidence_manifest.json) | Every original file path, exact byte count and SHA-256 |
| [Physical execution log](../records/development_pilot_run.log) | Actual fixed-budget execution output |
| [Independent audit log](../records/development_pilot_audit.log) | Actual post-execution audit and archive verification |
| [Focused regression checks](../records/development_pilot_tests.log) | Corrupt contacts/timing/labels, final-plan refusal and evidence-protection tests |

The full raw evidence is distributed as an attachment to the repository's [development-pilot-v1 Release](https://github.com/youseihuayu-wonderful/intelligent-grasping-course-project/releases/tag/development-pilot-v1), rather than inflating every Git clone. The archive contains **all** original raw trial records, sparse traces, dense contact/reference streams, instrumentation/warning records, aggregate records and collection provenance. No private final-test plan, final seed, API credential or unrelated source checkout is included.

## Unchanged physical experiment

The physical work uses the original simulation lockfile and `grasp_study_physics.run_one`, with the same Panda, object assets, solver/contact setup, control parameters and physical acceptance criteria as the recorded source environment. The runner checks all recorded runtime file hashes and package versions before execution and checks source hashes again afterward.

Every trial gets its own original reset/settling/grasp lifecycle; there is no object attachment, pose correction during execution, outcome-based stopping, favourable-case replacement or selective retry. No classifier, GPT or vision system controls this pilot. Scene resets **between independent trials** are expected; this is not the continuous drawer-storage mission.

The allowed model inputs remain:

```text
target, scale, mass_kg, sliding_friction, x_m, y_m, yaw_rad,
opening_m, closing_force_n, close_trigger_m, descent_speed_m_s
```

Pose columns come from the settled pre-execution object pose and normalized MuJoCo `wxyz` quaternion. Mass/friction are prescribed simulator truth; speed, force and opening are commands, not measurements of constant TCP velocity, actual grip force or achieved gap. IDs, role/split, outcomes, timing, raw evidence and latent error are not predictors.

## Read-only, every-physics-step telemetry

The original source retains roughly 20 ms sampled traces. For this pilot, an adapter additionally records a **dense stream after each 2 ms physics tick**, without editing the physical source or its raw labels.

For every tick it records:

- Absolute simulation time, sequential step index, controller phase/status.
- Target position/quaternion, TCP position, exact saved lift-height and relative-position references.
- Every target-body contact's geometry/body IDs and constraint address, with native normal forces for active contacts.
- The original controller's physical predicates and its distinct internal hold predicate.
- A guard proving that observing/serializing that tick left `qpos`, `qvel`, actuator controls and simulation time unchanged.

The adapter uses a separate native force-output buffer. Querying the original predicate getter may refresh the controller's contact-force scratch buffer, but does not modify the guarded physical state or issue an action. Its temporary simulation subclass is restored after each trial. The adapter and runner hashes are retained in the collection manifest.

The stream format is gzip-compressed JSON Lines. Its header declares the schema, timestep, target/finger/table body IDs, contact-column order and thresholds. Step records use these contact columns:

```text
target_geom_id, other_geom_id, other_body_id, efc_address, normal_force_n
```

Inactive contacts have a null normal force. A table contact counts as support under the original rule even when its constraint is inactive or its force is zero. A finger counts as touching only with an active contact and normal force **strictly greater than 0.01 N**.

## Independent post-execution checks

The auditor streams every dense record and independently reconstructs:

1. Bilateral finger contact from the native contact/force records.
2. Absence of table support from the full target-contact list.
3. Lift of at least **0.10 m** from the saved reference height.
4. Target-to-TCP relative-position displacement **less than 0.01 m** from the exact saved hold reference.
5. The continuous **two-second** acceptance clock during eligible `hold`/`done` states, resetting it on any predicate failure.

It does not accept a logged green flag or the controller's shorter internal hold as a substitute. Tick indices and times must cover every physics step without gaps, duplicates or reordering. Reference capture must agree with the actual preceding step, and final status, pose, duration, raw predicates and original label must agree with the reconstructed evidence. The existing sparse-trace audit is also applied.

For this new pilot, dense logging removes the old unsaved-2-ms-tick gap for the declared discrete-time predicates. It does **not** create an independent real-world sensor, validate the simulator's material/contact model, establish orientation stability, or replay the full dynamics from complete state. References still come from the controller, and forces from the same MuJoCo solver.

## Errors are not failed grasps

- A completed, reliable controller failure remains **label 0**.
- Invalid initialization, an exception, a MuJoCo numerical warning, a Python numerical warning, a mutation-guard failure or unverifiable evidence is an explicit **unusable/nonlabel** case, never a convenient negative.
- Original source records are preserved even when later instrumentation/archive checks fail.
- The aggregate attempt list preserves every planned trial. Any missing/unusable/unverifiable attempt prevents the pilot from passing its quality gate; there are no hidden replacements.
- `dataset.csv` contains only independently verified labelled rows. Its row count must be reconciled with the all-attempt list and issue report, not assumed to equal the planned budget.

The published success fraction, if available, describes this particular four-corner pilot among its verified trials. The pilot is intentionally not a representative sample of the full property grid, and its paired trials are not independent conditions. Do not call that fraction a classifier accuracy, final-test success rate or general robotic grasp success rate. Failure phases and recorded diagnostic categories are not independently proven root causes.

## Reproduction

Use a **new** evidence directory outside Git. From the original simulation checkout, execute the course runner with the original simulation environment:

```sh
uv run --locked python ../intelligent-grasping-course-project/run_development_pilot.py \
  --source-repo . \
  --output ../intelligent-grasping-pilot-evidence/development_pilot_v1_local \
  --workers 3
```

This executes only the pinned public pilot. It rejects changed plan/policy bytes and cannot use the withheld final plan as a substitute. Existing output paths, evidence destinations inside Git, and final-test custody destinations are refused.

From this course checkout, audit/export/archive that new run using the course environment:

```sh
PYTHONPATH=tests uv run --locked python -m unittest test_pilot_evidence_audit test_development_pilot
uv run --locked python audit_development_pilot.py \
  --source-repo ../robot-grasp-simulation \
  --evidence-dir ../intelligent-grasping-pilot-evidence/development_pilot_v1_local \
  --output results/development_pilot_v1_local \
  --archive ../intelligent-grasping-pilot-evidence/development_pilot_v1_local_evidence.tar.gz
uv run --locked python verify_pilot_export.py \
  --source-repo ../robot-grasp-simulation \
  --evidence-dir ../intelligent-grasping-pilot-evidence/development_pilot_v1_local \
  --export-dir results/development_pilot_v1_local
```

The audit hashes every raw file and verifies **every archived member's uncompressed bytes** against the inventory. Archive paths are relative under `development_pilot_v1/`. Inspect the saved `archive.json` for the actual published artifact identity; a new local run has different wall times/gzip metadata and is not expected to produce the same archive hash.

To recheck the published run without rerunning physics, download its Release archive, verify the archive SHA-256 first, extract it into a new evidence directory, and use the same audit command with a new export/archive destination. Preserve the archive's directory prefix when choosing `--evidence-dir`.

## How teammates should use this batch

This is a **separate development-only batch**, not a replacement for the frozen 3,840-row v1 package, not an untouched evaluation set, and not an automatic addition to one member's training data. It has no newly defined validation/test partitions or CV folds. Any later training-data expansion needs one coordinated version, grouped folds and equal access for all model developers.

The frozen independent final-test inputs/seeds remain outside Git and are not opened by the pilot runner or auditor. Final-test collection still waits for the agreed model-freeze prerequisites. This work does not train, rank or evaluate the four course classifiers.

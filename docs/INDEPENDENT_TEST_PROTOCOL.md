# Independent test design v1

Execution update: the separate development pilot has now been completed and audited; see [its actual results and evidence](DEVELOPMENT_PILOT.md). This page preserves the preregistered design and its design-only verification history. The independent final test remains unexecuted and sealed.

## 1. Research question and scope

How well do the frozen course classifiers predict **single-object grasp-and-hold success** under new initial poses and new latent execution errors in the existing five-family MuJoCo/Panda domain?

The predictors, controller, success rule and physical parameter levels remain those of [dataset v1](DATASET.md). This is not another split of the public dataset, a test on unseen geometry families, a drawer-storage evaluation, or a sim-to-real experiment. No new classifier is trained by this design step.

The old test labels/results were already public. The new final test therefore uses genuinely new planned conditions rather than renaming that historical partition.

## 2. Two separate batches

| Batch | Condition groups | Groups per family | Configurations per group | Paired error draws | Planned trials | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Development pilot | 20 | 4 | 8 | 2 | 320 | Check the collection pipeline; development evidence only |
| Independent final test | 240 | 48 | 8 | 2 | 3,840 | One locked comparison after model freeze |

These are **planned budgets, not collected samples or guarantees of statistical power**. The pipeline does not collect extra trials until a preferred class balance or model ranking appears.

The pilot contains four predefined property corners per family: increasing mass levels paired with decreasing friction levels, with scale sequence 0.8, 1.0, 1.2, 0.8. It covers all four family mass/friction levels and all three scales, but is not a representative prevalence sample. Its `split=train` metadata means development-only eligibility in the original collector; it does not append rows to frozen v1 or define new CV folds.

The final test covers the complete **3 scale × 4 family mass × 4 friction** grid once per family. Each group gets a new pose and two new error draws. The eight candidate commands follow the existing anchored/contrast design; all 54 joint command possibilities occur across each final-test family, not at every condition. Every classifier is evaluated on the same planned candidates, not allowed to choose an easier evaluation subset.

## 3. Factors held within the existing domain

| Factor | Settings |
| --- | --- |
| Families | Bowl, bottle, spoon, glass tumbler, pen |
| Relative scale | 0.8, 1.0, 1.2 |
| Bowl/bottle mass | 0.06, 0.18, 0.30, 0.45 kg |
| Spoon mass | 0.015, 0.025, 0.040, 0.060 kg |
| Glass mass | 0.080, 0.150, 0.250, 0.350 kg |
| Pen mass | 0.005, 0.010, 0.020, 0.030 kg |
| Sliding-friction setting | 0.20, 0.40, 0.80, 1.20 |
| Nominal x/y/yaw | x: 0.475–0.525 m; y: −0.100–−0.060 m; yaw: −0.15–0.15 rad |
| Opening command | 0.04, 0.08 m |
| Closing tendon-force command | 2, 4, 8 N |
| Closure trigger distance | 0, 0.01, 0.02 m |
| Descent-speed setting | 0.04, 0.10, 0.20 m/s |
| Hidden execution error | Independent uniform [−4,+4] mm per body-frame XYZ component, paired across all eight commands in a draw |

All 11 predictors retain their original meaning. Model inputs use the **settled** pre-grasp pose, which will only be available after actual scene settling; nominal pose is used for design/grouping. Latent biases, seeds, IDs, diagnostics, trajectories and outcomes are never predictors. Family-specific mass levels and contact profiles do not imply a causal shape-only effect. The rigid glass does not fracture and the pen does not bend.

## 4. New conditions, reproducibility and blinding

The planner reads the immutable original **unlabelled collection protocol**, not its trial outcomes. It checks source identities and unchanged v1 data/fold hashes without using labels to choose conditions.

- New condition namespaces separate pilot, final test and historical data.
- Each family/role receives Latin-hypercube nominal x/y/yaw poses.
- A new pose must be at least **5 mm away in XY** from historical poses with the same family, scale, mass and friction; final poses also avoid matching pilot poses. This is a design-space distance, not a guarantee about the eventual settled state.
- Whole candidate pose designs can be rejected only for this predeclared geometric constraint, never for simulated outcomes. A fixed maximum of 1,000 candidate designs per family prevents indefinite silent search.
- Nuisance seeds are fresh relative to historical data and pilot, and are not reused across final condition groups. Two draws remain paired across each group's eight commands.
- Group IDs are derived from nominal context and condition ID, independently of labels.

The 5 mm constraint makes this a **constrained same-domain test design**, not IID sampling from the original pose process and not an out-of-distribution robustness claim. Actual settled-state overlap and physical initialization still need checking during collection.

### Public versus private artifacts

The [public policy](../protocols/independent_test_v1/policy.json), [frozen pilot plan](../protocols/independent_test_v1/pilot_plan.json), [final-plan commitment](../protocols/independent_test_v1/final_commitment.json) and [design validation](../protocols/independent_test_v1/design_validation.json) are committed here.

The **actual final plan and its 256-bit random planning seed stay outside every Git checkout**, in a custodian-only local directory. The directory/file permissions exclude other OS users. Publishing a reproducible final seed now would let teammates run the deterministic simulator and obtain labels before model freeze, even if no label CSV were uploaded. Therefore the final seed and concrete final inputs are deliberately withheld, not merely the labels.

SHA-256 binds the private plan to the published commitment. It is not encryption or access control; privacy depends on local custody, permissions and team conduct. Keep a secure private backup without uploading it to this repository. After predictions and final evaluation are frozen, release the plan, seed and actual evidence together for independent reproduction.

Fields named `*_payload_sha256` hash the planner's sorted, indented JSON representation with its trailing newline; they are not presented as hashes of the original differently formatted source files. The immutable source commit and existing data manifest retain source-file identities.

## 5. Label and abnormal-attempt rules

Use the unchanged physical observer: terminal controller success plus all four predicates—bilateral contact, no table support, at least 0.10 m lift and relative-position stability within 0.01 m—for at least two continuous simulation seconds. Stability is not a full-orientation guarantee. The observer shares controller predicates, and the original sampled trace format cannot independently prove every unsaved physics step; see [the audit limitations](DATA_QUALITY.md#what-this-audit-cannot-certify).

- Preserve ordinary controller/grasp failures as **label 0**, including difficult heavy/slippery configurations.
- Preserve invalid initialization, internal errors and numerical-warning attempts as explicit **nonlabels**, with raw errors and trial IDs. Do not turn software faults into ordinary negatives.
- Do not replace unfavourable cases, retry selectively, drop rows to improve scores, or stop when the desired success ratio appears.
- A missing or unusable final attempt blocks a confirmatory comparison. Preserve the incomplete batch and diagnose it; do not silently score a convenient subset. If an environment/protocol change is necessary, version and document it before any replacement full-batch run.

Pilot feedback can identify implementation problems. It cannot silently alter this frozen final plan, relax the physical success rule or remove hard conditions. Material changes require a new preregistration and renewed model-freeze review.

## 6. Team handoff and freeze order

Member A is the data custodian. Members B/C/D can continue model development on the existing v1 training/validation data while the pilot is checked.

Before A launches the **final** collection, record each Logistic Regression, LASSO-selector-plus-Logistic, Random Forest and XGBoost model's:

- Exact fitted artifact hashes and training-code revision.
- v1/fold hashes and the common 11-feature allowlist.
- Selected hyperparameters, fold-local preprocessing/feature selection and any validation-only calibration.
- Dependency versions, model selection history, classification threshold and tie rule.

For this comparison, all final fitted pipelines use v1 **training rows only**, matching the shared baseline contract. Agree and document appropriate tuning budgets on development data before model freeze. No final outcomes, fitted test preprocessing, test-guided feature selection or test-guided calibration are allowed.

After collection and integrity checks, use frozen pipelines to produce and lock predictions on the same row identities **before joining test labels for scoring**. A checks the actual model-freeze record before launching collection; the planning CLI does not launch the collector or claim to enforce team behaviour. Once final labels are opened, changes are exploratory follow-up work, not repeated untouched-final-test evidence.

## 7. Predeclared evaluation

Primary metric: **log loss**. Secondary outputs: Brier score, ROC-AUC, balanced accuracy, failure-class-0 recall, labelled confusion matrix and class supports. Include the constant-probability reference estimated from **training labels only**.

Use class-1 probabilities, float64 scoring, the common log-loss clipping convention, and `p_success >= 0.5` for predicted success, including exact ties. Report undefined metrics as null with an explanation rather than zero. The policy specifies these conventions; computing final metrics is a later evaluation task, not a result of this design step.

Uncertainty will use **2,000 paired bootstrap resamples of whole condition groups**, stratified by target family, with seed 5644. Carry all 16 trials of a sampled group together and use identical sampled groups for all models. Do not bootstrap rows as if paired controls/noise draws were independent. These are approximate descriptive intervals conditional on this constrained/factorial design, not guaranteed population coverage or multiplicity-adjusted proof of superiority.

This comparison evaluates success-probability prediction under the predefined candidate distribution. It does not itself demonstrate faster executed grasping, identify globally optimal controls or guarantee a model will outperform Logistic Regression. Those claims require separate task-execution evidence.

## 8. What has been verified at this stage

The design checker validates trial budgets, complete final property grids, candidate coverage, paired noise, pose strata, historical/pilot exclusions, group IDs, unchanged v1 hashes and absence of outcome fields. Re-reading the saved bundle must reproduce its deterministic private plan and published hashes. Targeted tests exercise corruption, leakage, private-file permissions and overwrite protection.

**Observed proof:** all 12 boundary tests passed, the real design build completed, and verification of the saved public/private bundle passed. No historical condition/nuisance seed was reused; the five final families each cover all 54 candidate configurations. These are design checks, not executed-grasp results.

The first synthetic-test attempt exposed an undefined minimum distance when its reduced history had no matching property combination. The summary now uses JSON null rather than infinity; no checks were removed. The [original failed attempt](../records/independent_test_design_tests_attempt_01.log) is retained alongside the passing run.

Authoritative execution records:

- [Planning regression tests](../records/independent_test_design_tests.log)
- [Actual design build](../records/independent_test_design_build.log)
- [Saved-bundle verification](../records/independent_test_design_verify.log)

**No new physical trial, success/failure label, classifier training or final score is produced here.** Collector compatibility means the `[trial_id, settings, metadata]` shape and parameter domain match the original recorded interface. New physical initialization and runtime validity have not been tested; that belongs to the pilot, not this planning-only check.

## 9. Reproduce or inspect

From this course checkout, with the original simulation/evidence checkout available:

```sh
PYTHONPATH=tests uv run --locked python -m unittest test_independent_test_design
uv run --locked python design_independent_test.py verify \
  --source-repo ../robot-grasp-simulation \
  --public-dir protocols/independent_test_v1 \
  --private-dir ../intelligent-grasping-private-evaluation/final_test_v1
```

Verification of the sealed final design requires A's private file. A public checkout alone can read the complete policy/pilot, inspect the code and run synthetic boundary tests, but cannot regenerate the withheld final conditions yet. This is intentional.

For a **separate development-only planning experiment**, choose new public/private output directories:

```sh
uv run --locked python design_independent_test.py build \
  --source-repo ../robot-grasp-simulation \
  --public-dir protocols/local_design_experiment \
  --private-dir ../intelligent-grasping-private-evaluation/local_design_experiment
```

That produces a different random final commitment; it does not replace the committed preregistration or qualify as another untouched test after results are known. Existing output paths, symlinks, frozen-data destinations and private destinations inside Git are refused. The committed design is bound to its planning-code hash; check out its corresponding code revision when reproducing it after future code changes.

**Current handoff:** the development pilot has been executed with all-attempt preservation and independently audited; consult [the pilot report](DEVELOPMENT_PILOT.md) before considering any coordinated development-data expansion. Keep its labels separate from final-test evidence. Do not start the independent final batch until the model-freeze prerequisites above are satisfied.

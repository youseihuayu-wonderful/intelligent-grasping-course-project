# Dataset v1

## Download and provenance

- [Raw CSV download](https://raw.githubusercontent.com/youseihuayu-wonderful/intelligent-grasping-course-project/main/data/v1/dataset.csv)
- [Local CSV](../data/v1/dataset.csv) · [Package manifest](../data/v1/manifest.json)
- [Frozen row assignments](../data/v1/split_assignments.csv) · [Frozen three-fold CV manifest](../data/v1/cv_manifest.json)
- [Source environment](../data/v1/source_environment.json) · [Source collection summary](../data/v1/source_collection_summary.json)

The CSV is copied byte-for-byte from `results/grasp_classifier_five_objects_20261121/dataset.csv` in the [original simulation repository][upstream], frozen at commit [`7a2c5642113d0d3b9c5b5d5205feda843b2c043a`][snapshot]. Its SHA-256 is:

```text
d218aca702d21fcd0bfe55d60b8022939e732a33c5cb81e42cf07f319716064a
```

`data/v1/manifest.json` records the dataset version, exact feature allowlist, artifact hashes, fixed threshold, split counts, and source provenance. Paths in upstream evidence may identify the original collector's machine; they are provenance, not dependencies for running this CSV-only project. Validate the package with:

```sh
uv run --locked python baseline.py --validate-only
```

The original collection reported 3,840 attempted and labelled trials and zero warnings. That is a property of this recorded collection, not a general runtime or throughput guarantee. No physical evidence or historical labels are changed by this course package.

## Counts and preserved partitions

| Partition | Rows | Condition groups | Successes (1) | Failures (0) |
| --- | ---: | ---: | ---: | ---: |
| Training | 2,560 | 160 | 1,109 | 1,451 |
| Validation | 640 | 40 | 243 | 397 |
| Historical test | 640 | 40 | 217 | 423 |
| Total | 3,840 | 240 | 1,569 | 2,271 |

Each of the five families—`bowl`, `bottle`, `spoon`, `glass`, `pen`—has 768 rows. Each group contains 16 trials: eight control configurations evaluated with two paired nuisance draws. These are not 3,840 independent object conditions.

Keep CSV row order, `row_index`, `trial_id`, `group_id`, and `split` unchanged. Groups must never cross training, validation, test, or the training/validation sides of a CV fold. Do not replace the upstream partitions with a random row split.

The CV manifest preserves the original three grouped folds of the training partition. Its `train_rows` and `validation_rows` are **global zero-based CSV row indices**, not positions within a training-only array. The baseline validates membership and group isolation, then maps these indices to training-local positions. Preserve the supplied fold arrays and group lists instead of regenerating folds from a seed. The frozen CV seed is 5644; it is not the physical collection seed or a row's nuisance draw.

The upstream [`group_ids` implementation][data-code] builds label-independent connected components from **nominal** target/state settings and prospective condition IDs, additionally linking matched global seeds for historical evidence when applicable. Nominal pose defines grouping; the slightly different settled pose used as a predictor does not create a new group. In this prospective batch, `nuisance_seed` denotes a local error draw paired across all eight controls within its condition. It is not a globally shared independent simulation seed, an independent group identifier, or a predictor. Latent body-frame XYZ grasp errors are excluded from features.

**The existing test data and results have already been published and inspected.** Preserve this partition for historical traceability, but never describe it as a new untouched final test. It must not guide tuning. Current baseline evaluation uses training CV and validation only. Fresh final claims require newly held-out conditions later, outside current steps 1–3.

## The 11 allowed predictors

Use these fields in the exact manifest order. The first seven describe pre-grasp context; the last four are proposed controller commands.

| Predictor | Type / unit | Role and meaning |
| --- | --- | --- |
| `target` | Categorical: five named families | Context: semantic object family, also associated with the upstream geometry, grasp profile, and contact model; not an independently measured shape descriptor. |
| `scale` | Numeric, dimensionless multiplier | Context: relative scaling of the family's geometry and nominal grasp offset; not measured volume or a common absolute size across families. |
| `mass_kg` | Numeric, kg | Context: prescribed object mass in simulation; simulator ground truth, not a mass estimated from vision. |
| `sliding_friction` | Numeric, dimensionless coefficient | Context: prescribed sliding-friction setting for the object/finger setup; not a measured constant contact coefficient at every contact. |
| `x_m` | Numeric, m | Context: settled target position along the world-frame x axis, observed before grasp execution. |
| `y_m` | Numeric, m | Context: settled target position along the world-frame y axis, observed before grasp execution. |
| `yaw_rad` | Numeric, rad | Context: settled target yaw derived from its initial MuJoCo `wxyz` quaternion before grasp execution. |
| `opening_m` | Numeric, m | Control: requested initial gripper opening, not a measured final gap. |
| `closing_force_n` | Numeric, N | Control: requested closing tendon-force command; not measured per-finger normal force or guaranteed achieved grip force. |
| `close_trigger_m` | Numeric, m | Control: remaining vertical distance above the planned grasp point at which aligned descent can trigger early closure; zero uses closure at descent completion. |
| `descent_speed_m_s` | Numeric, m/s | Control: configured descent-speed parameter; not a measurement or guarantee of constant TCP velocity. |

The source [`features_from_row`][data-code] extracts only pre-execution inputs. The collector settles the scene for one simulation second, then records initial pose before starting the grasp. Thus pose columns are settled observations, not outcome leakage and not simply the nominal pose used to form groups. The source extractor supports a nominal-pose fallback when no initial pose is supplied; that does not change this batch's recorded settled-pose interpretation.

Mass, friction, and pose availability are simulation assumptions, not proof that a real robot could observe them without additional sensing. The CSV provides **no object dimensions, measured volume, image features, or independently measured geometric shape descriptors**. Do not invent these quantities from category labels or treat `scale` as volume. Mass and scale vary independently, so density need not remain constant.

### Observed v1 numeric coverage

The following values come from `feature_ranges` in the frozen manifest, not from an older two-object batch. These are observed dataset values, not universal simulator limits or guarantees of out-of-range validity.

| Predictor | Observed values or range |
| --- | --- |
| `scale` | 0.8, 1.0, 1.2 |
| `mass_kg` | 0.005–0.45 kg across 15 family-dependent levels |
| `sliding_friction` | 0.2, 0.4, 0.8, 1.2 |
| `x_m` | Approximately 0.475131–0.524736 m |
| `y_m` | Approximately −0.099721–−0.060159 m |
| `yaw_rad` | Approximately −0.149099–0.149332 rad |
| `opening_m` | 0.04, 0.08 m |
| `closing_force_n` | 2, 4, 8 N |
| `close_trigger_m` | 0, 0.01, 0.02 m |
| `descent_speed_m_s` | 0.04, 0.10, 0.20 m/s |

See the manifest for exact pose extrema and the full mass-level list. A global mass range does not mean every family was evaluated at every mass. No new physical trials were collected to create this course package; its source artifacts were matched against the upstream GitHub blobs at the frozen commit.

### Forbidden predictors

The complete CSV also contains the following fields, which must not enter any model:

| Field | Permitted purpose only |
| --- | --- |
| `row_index`, `trial_id` | Traceability and joining predictions to evidence |
| `group_id` | Leakage-safe grouping |
| `split` | Fixed partition assignment |
| `success` | Supervised label |
| `execution_sim_s` | Post-execution diagnostic duration |
| `evidence_source` | Provenance |

Do not parse IDs, file paths, or split names into features. Also exclude upstream metadata, seeds/nuisance seeds, control IDs, latent `bias_*` errors, trial status, failure reasons, final poses, contact diagnostics, hold times, trajectories, and any other outcome-derived information. The feature allowlist is exhaustive: extra columns are not optional predictors.

## What the label means

`success = 1` means a completed, physically verified **single-object grasp-and-hold**; `success = 0` means a usable failed grasp attempt. Invalid initial states and unreliable evidence are quarantined by the upstream audit, not relabelled as ordinary negatives. The exported batch contains 3,840 usable rows.

In [`grasp_study_physics.run_one`][physics-code], an observer with its own timer checks the following four physical predicates continuously at simulation steps during `hold`/`done`:

1. `bilateral_grip`: both fingers contact the target.
2. `clear_of_table`: the target has no table support.
3. `target_lifted`: target height is at least **0.10 m** above its reference height.
4. `relative_pose_stable`: target-to-TCP relative-position displacement from the hold reference has norm **less than 0.01 m**.

All four must remain true for at least **2.0 continuous simulation seconds**, with controller status `succeeded` required for the final positive label. A predicate failure resets the observer timer; a brief lift or the controller's own shorter hold check is insufficient. “Independent” refers to this separate timing/acceptance check: it shares the controller's physical predicates, not an independent sensor. The predicate implementations are in [`grasp.py`][grasp-code]. Stability here is the implemented relative-position test, not a separate full-orientation guarantee.

This is not a label for transfer, drawer placement, release, drawer closure, or all-five-object completion. A rendered video, green UI state, or successful scripted storage sequence must not be used to assign additional training labels.

## Dataset and simulation boundaries

The five-family models are rigid proxies, not scans or validated material models. Glass neither fractures nor contains fluid; a pen neither bends nor writes. Families have different geometry, grasp profiles, and some contact settings, so a family effect cannot be interpreted as a causal shape-only result. There is no vision pipeline or sim-to-real validation here.

The [upstream simulation guide][simulations] provides separate links and runtime instructions for the **first independent five-object grasp-and-hold baseline (Experiment 1)** and **recorded continuous five-object drawer storage (Experiment 2)**. Runnable code, assets and authoritative execution evidence stay upstream; an unchanged English MP4 and animated preview are also mirrored in the [homepage demo](../README.md#simulation-demo). Experiment 2 is frozen at the [`experiment-02-five-object-storage` release][storage-release], commit `7a2c5642113d0d3b9c5b5d5205feda843b2c043a`. It is fixed-controller engineering evidence, not a new training batch and not proof of 100% classifier success. Shared MuJoCo/Panda/assets do not establish generalization across changed poses, clutter, or controllers.

Do not mix v1 with the older **550-row two-object** [`grasp_classifier_comparison_20261115` dataset][older-data], which includes historical training supplementation. The older [`grasp_classifier_csv_baseline_20261121` run][older-baseline] is a CSV-only reference on that older dataset, not the 3,840-row data source. Earlier screening/development evidence, Experiment 1 demonstrations, this classifier collection, and Experiment 2 storage have distinct purposes; none supplies interchangeable labels. Existing source/results are references with provenance, not final student submissions.

[upstream]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms
[snapshot]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/7a2c5642113d0d3b9c5b5d5205feda843b2c043a
[data-code]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/grasp_classifier_data.py
[physics-code]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/grasp_study_physics.py#L187-L297
[grasp-code]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/grasp.py#L254-L264
[simulations]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/main/docs/SIMULATIONS.md
[storage-release]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/experiment-02-five-object-storage
[older-data]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/grasp_classifier_comparison_20261115
[older-baseline]: https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/grasp_classifier_csv_baseline_20261121

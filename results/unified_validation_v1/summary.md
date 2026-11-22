# Unified validation evaluation

Development validation only; no historical-test, pilot or private-final scoring.

| Model | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Balanced accuracy ↑ | Failure recall ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant_train_prior | 0.669812 | 0.238389 | 0.500000 | 0.500000 | 1.000000 |
| logistic | 0.494009 | 0.163581 | 0.838226 | 0.709431 | 0.871537 |

Rows: 640; condition groups: 40; paired-difference reference: `logistic`.

Read `uncertainty.json` for paired, family-stratified whole-group intervals, defined-replicate counts and limitations. `bootstrap_group_weights.csv` preserves every draw.

All submitted models use the same authoritative row identities and scoring rules. Training-feature/CV claims are checked declarations tied to hashed files, not an independent audit of arbitrary training execution. The evaluator never unpickles model artifacts.

Only submitted real models are listed. The constant reference uses training prevalence only. No model ranking here is a fresh final-test or physical-execution claim.

# Revised proposal and supplementary study plan

**EECE 5644: Machine Learning — Fall 2026**  
Matthew Mikalsen · Benjamin Heep · Francesco Di Mise · Shihua YU

This directory publishes the reviewed English revision, preserving the I–VI research-question structure. The proposal and supplement describe the intended study; new learner results, ablations, policy execution and final-test scores are **not** implied to exist. Publication does not change the frozen dataset or run an experiment.

## Read or download

| Document | Read on GitHub | PDF | Editable Word | LaTeX source |
| --- | --- | --- | --- | --- |
| Research proposal — 3-page PDF | [Proposal](Proposal.md) | [View PDF](Proposal.pdf) · [Download](https://raw.githubusercontent.com/youseihuayu-wonderful/intelligent-grasping-course-project/main/docs/proposal/Proposal.pdf) | [Download Word](https://raw.githubusercontent.com/youseihuayu-wonderful/intelligent-grasping-course-project/main/docs/proposal/Proposal.docx) | [Proposal.tex](Proposal.tex) |
| Supplementary study plan — 2-page PDF | [Study plan](Study_Plan.md) | [View PDF](Study_Plan.pdf) · [Download](https://raw.githubusercontent.com/youseihuayu-wonderful/intelligent-grasping-course-project/main/docs/proposal/Study_Plan.pdf) | [Download Word](https://raw.githubusercontent.com/youseihuayu-wonderful/intelligent-grasping-course-project/main/docs/proposal/Study_Plan.docx) | [Study_Plan.tex](Study_Plan.tex) |

Share **this directory link** with teammates so both documents stay together. The Markdown is readable without downloading software; the PDFs preserve reviewed typesetting, and Word files support editing. Original local documents were not overwritten.

## How this connects to model work

1. Read the proposal for the research questions, actual 11-feature contract, classifier rationale and claim boundaries.
2. Use the existing [training and delivery specification](../MODEL_TRAINING_AND_DELIVERY.md) and [base search plan](../../protocols/model_training_v1.json) for the four primary full-feature routes: Logistic Regression, LASSO selection plus Logistic Regression, Random Forest, and XGBoost. The supplement cites the immutable base revision and file hash.
3. Use the [supplement](Study_Plan.md) for additional speed-only/object-only models, the physics-inspired proxy baseline, explicit interactions, and the separate matched policy-execution study. These are planned extensions, not silently changed completed results or replacements for the base protocol.
4. B/C/D develop the primary models in parallel. A manages data and shared evaluation. Review the supplemental implementation/assignment details together before executing additional routes or collecting policy-study labels.
5. Keep the [independent final-test protocol](../INDEPENDENT_TEST_PROTOCOL.md) separate. Final collection waits for model freeze; neither the old published test partition nor development validation is an untouched final assessment.

The existing [unified evaluator](../UNIFIED_EVALUATION.md) scores complete frozen validation submissions and checks declarations/hashes. Its checks alone do not prove fold-local training execution. New final scoring and model-guided policy execution are still required implementation work, not capabilities established by the demo videos.

## Document verification and rebuilding

The retained [document verification record](verification.json) identifies the eight reviewed document files by SHA-256 and records the original checks: Word text matches Markdown, Word has native equation objects, PDFs have no missing glyphs or overfull boxes, and the original Word/base training plan were unchanged. The prior visual review covered the rendered PDF pages, not Microsoft Word layout. This record is document evidence, not training or grasp-performance evidence.

Final compilation logs are retained for [the proposal](proposal_compile_final.log) and [the supplement](study_plan_compile_final.log). Font-request notices were present; they are not hidden. To rebuild the PDF from LaTeX, with Tectonic installed, run in a scratch copy of this directory:

```sh
tectonic --keep-logs Proposal.tex
tectonic --keep-logs Study_Plan.tex
```

Do not overwrite reviewed artifacts and leave stale hashes when revising. Preserve the source, regenerate the affected formats, review equations/content/layout, and publish new verification identities. The [publication check](../../records/proposal_publication_check.log) records byte-for-byte transfer integrity and local-link checks for this repository delivery.

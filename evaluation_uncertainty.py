"""Paired, target-family-stratified whole-condition bootstrap on VALIDATION."""
from __future__ import annotations

from collections import Counter

import numpy as np

from baseline import validation_metrics


METRICS = ("log_loss", "brier", "roc_auc", "balanced_accuracy", "failure_recall")


def _interval(estimate, estimate_reason, values, reasons):
    """Retain undefined support instead of replacing missing metrics with zero."""
    valid = [value for value in values if value is not None]
    lower, upper = (None, None)
    if valid:
        bounds = np.quantile(valid, [0.025, 0.975], method="linear")
        lower, upper = (float(value) for value in bounds)
    if not valid:
        condition = "undefined_no_valid_resamples"
    elif len(valid) != len(values):
        condition = "conditional_on_metric_defined"
    else:
        condition = "all_resamples_defined"
    return {
        "estimate": estimate,
        "estimate_undefined_reason": estimate_reason,
        "lower": lower,
        "upper": upper,
        "valid_resamples": len(valid),
        "undefined_resamples": len(values) - len(valid),
        "undefined_reasons": dict(sorted(Counter(reasons).items())),
        "interval_condition": condition,
    }


def _difference(candidate, reference, metric):
    if candidate[metric] is None or reference[metric] is None:
        reasons = []
        if candidate[metric] is None:
            reasons.append(f"candidate: {candidate['undefined'][metric]}")
        if reference[metric] is None:
            reasons.append(f"reference: {reference['undefined'][metric]}")
        return None, "; ".join(reasons)
    return float(candidate[metric] - reference[metric]), None


def grouped_bootstrap(
    rows: list[dict],
    probabilities_by_model: dict[str, list[float]],
    *,
    reference: str,
    resamples: int = 2000,
    seed: int = 5644,
) -> dict:
    """Return percentile intervals, paired contrasts, and reproducible group draws.

    Each family contributes its original number of groups to every draw. All
    rows of each selected group are carried with its sampled multiplicity, and
    that exact row draw is used for every model. Differences are always
    candidate minus reference, not differences between independent intervals.
    """
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer, not a boolean")
    if type(seed) is not int or seed <= 0:
        raise ValueError("seed must be a positive integer, not a boolean")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Only nonempty VALIDATION rows may be bootstrapped")
    if not isinstance(probabilities_by_model, dict) or not probabilities_by_model:
        raise ValueError("At least one model probability array is required")
    if any(not isinstance(name, str) or not name.strip() for name in probabilities_by_model):
        raise ValueError("Model identifiers must be nonempty strings")
    if not isinstance(reference, str) or reference not in probabilities_by_model:
        raise ValueError("reference must identify a supplied model")

    group_rows, group_families = {}, {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("split") != "validation":
            raise ValueError("Only nonempty VALIDATION rows may be bootstrapped")
        group, family = row.get("group_id"), row.get("target")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Each row requires a nonempty string group_id")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Each row requires a nonempty string target family")
        if group_families.setdefault(group, family) != family:
            raise ValueError("A condition group cannot cross target families")
        if str(row.get("success")) not in {"0", "1"}:
            raise ValueError("Invalid binary label")
        group_rows.setdefault(group, []).append(index)

    model_order = sorted(probabilities_by_model)
    probabilities, point = {}, {}
    for model in model_order:
        try:
            probabilities[model] = np.asarray(probabilities_by_model[model], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid success probabilities for {model}") from error
        # The baseline is the single source of metric math and probability checks.
        point[model] = validation_metrics(rows, probabilities[model])

    group_order = sorted(group_rows)
    family_order = sorted(set(group_families.values()))
    family_groups = {
        family: [index for index, group in enumerate(group_order)
                 if group_families[group] == family]
        for family in family_order
    }
    rng = np.random.default_rng(seed)
    weights = np.zeros((resamples, len(group_order)), dtype=np.int64)
    values = {model: {metric: [] for metric in METRICS} for model in model_order}
    reasons = {model: {metric: [] for metric in METRICS} for model in model_order}
    candidates = [model for model in model_order if model != reference]
    differences = {model: {metric: [] for metric in METRICS} for model in candidates}
    difference_reasons = {model: {metric: [] for metric in METRICS} for model in candidates}

    for replicate in range(resamples):
        for indices in family_groups.values():
            selected = rng.choice(indices, size=len(indices), replace=True)
            weights[replicate] += np.bincount(selected, minlength=len(group_order))
        selected_rows = np.asarray([
            index
            for group_index, group in enumerate(group_order)
            for _ in range(int(weights[replicate, group_index]))
            for index in group_rows[group]
        ], dtype=np.intp)
        sampled_rows = [rows[index] for index in selected_rows]
        scores = {
            model: validation_metrics(sampled_rows, probabilities[model][selected_rows])
            for model in model_order
        }
        for model in model_order:
            for metric in METRICS:
                value = scores[model][metric]
                values[model][metric].append(value)
                if value is None:
                    reasons[model][metric].append(scores[model]["undefined"][metric])
        for model in candidates:
            for metric in METRICS:
                value, reason = _difference(scores[model], scores[reference], metric)
                differences[model][metric].append(value)
                if reason is not None:
                    difference_reasons[model][metric].append(reason)

    model_intervals = {
        model: {
            metric: _interval(
                point[model][metric], point[model]["undefined"].get(metric),
                values[model][metric], reasons[model][metric],
            )
            for metric in METRICS
        }
        for model in model_order
    }
    difference_intervals = {}
    for model in candidates:
        difference_intervals[model] = {}
        for metric in METRICS:
            estimate, reason = _difference(point[model], point[reference], metric)
            difference_intervals[model][metric] = _interval(
                estimate, reason, differences[model][metric], difference_reasons[model][metric],
            )

    return {
        "summary": {
            "reference": reference,
            "metric_directions": {
                metric: "lower_is_better" if metric in {"log_loss", "brier"} else "higher_is_better"
                for metric in METRICS
            },
            "protocol": {
                "partition": "validation",
                "resampling_unit": "whole condition group_id",
                "stratification": "target family; original group count per family",
                "rows": len(rows),
                "groups": len(group_order),
                "family_group_counts": {family: len(indices) for family, indices in family_groups.items()},
                "family_order": family_order,
                "seed": seed,
                "resamples": resamples,
                "random_generator": "numpy.random.default_rng",
                "bit_generator": type(rng.bit_generator).__name__,
                "numpy_version": np.__version__,
                "metric_implementation": "baseline.validation_metrics",
                "confidence_level": 0.95,
                "interval": "95% percentile on valid resamples; numpy.quantile([0.025, 0.975], method='linear')",
                "undefined_handling": "Never impute zero; intervals condition on metric-defined resamples; null bounds if none are valid.",
                "difference": "candidate minus reference on identical group draws",
                "favourable_difference": {"log_loss": "negative", "brier": "negative",
                                         "roc_auc": "positive", "balanced_accuracy": "positive",
                                         "failure_recall": "positive"},
                "group_weights": "Rows are replicate order; columns are group_order; entries count whole-group multiplicities.",
                "limitations": [
                    "Validation is development data, not an independent final-test assessment.",
                    "Finite condition-group samples limit bootstrap precision and generalization; family group counts are fixed.",
                    "Paired draws preserve shared-row and within-group latent/control correlation; cross-group latent/control correlation is not modeled.",
                    "Intervals with undefined replicates are conditional on metric-defined support, not unconditional 95% coverage guarantees.",
                    "No winner or statistical-significance claim and no multiple-comparison adjustment is made.",
                ],
            },
            "models": model_intervals,
            "differences": difference_intervals,
        },
        "group_order": group_order,
        "group_weights": weights.tolist(),
    }

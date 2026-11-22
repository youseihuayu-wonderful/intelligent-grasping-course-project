"""Replay ONLY the known trusted reference artifact, never an arbitrary submission model."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import warnings

import joblib
import numpy as np

from baseline import FEATURES, feature_dict, load_dataset
from evaluation_submission import load_submission
from evaluate_models import new_output

ROOT = Path(__file__).resolve().parent
TRUSTED_REFERENCE_SHA256 = "487a03486b52b84613015c4d2ae9aa805b808557286821485f34db1582e7300d"


def verify(submission_path, output):
    output = new_output(output)
    package = load_dataset(ROOT/"data/v1")
    rows = [row for row in package["rows"] if row["split"] == "validation"]
    submission_path = Path(submission_path)
    submitted = load_submission(submission_path, rows, package["audit"])
    if submitted["model_id"] != "logistic":
        raise ValueError("This verification is restricted to the original reference")
    entry = submitted["provenance"]["files"]["model_artifact"]
    artifact = (submission_path.parent/entry["path"]).read_bytes()
    if hashlib.sha256(artifact).hexdigest() != TRUSTED_REFERENCE_SHA256:
        raise ValueError("Refusing to deserialize an unknown model artifact")
    # Bytes are pinned to the assistant-created reference, closing path-replacement races.
    model = joblib.load(io.BytesIO(artifact))
    classes = model.classes_.tolist()
    if classes != [0, 1]:
        raise ValueError("Unexpected reference class mapping")
    encoded = model.named_steps["vectorizer"].get_feature_names_out().tolist()
    allowed = set(FEATURES[1:]) | {"target=" + name for name in {row["target"] for row in rows}}
    if set(encoded) != allowed:
        raise ValueError("Reference encodes unexpected/missing observable features")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predicted = model.predict_proba([feature_dict(row) for row in rows])[:, classes.index(1)]
    observed = np.asarray(submitted["probabilities"], dtype=np.float64)
    if not np.isfinite(predicted).all() or not np.allclose(predicted, observed, rtol=0, atol=1e-12):
        raise ValueError("Saved reference predictions differ from trusted artifact replay")
    result = {"status": "passed_trusted_reference_prediction_replay", "rows": len(rows),
              "partition": "validation", "probability_class": 1, "classes": classes,
              "model_sha256": TRUSTED_REFERENCE_SHA256,
              "maximum_absolute_probability_difference": float(np.max(np.abs(predicted-observed))),
              "encoded_input_features": encoded, "only_allowed_observables_encoded": True,
              "dataset_sha256": package["audit"]["sha256"]["dataset.csv"],
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "models_trained": 0, "historical_test_rows_scored": 0, "private_final_opened": False,
              "scope": "Separate trusted-reference replay; the generic evaluator never deserializes model artifacts. This does not certify arbitrary submissions' training history."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.submission, args.output)


if __name__ == "__main__":
    main()

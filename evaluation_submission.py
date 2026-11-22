"""Hash-bound, validation-only prediction submissions; artifacts are never loaded."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re

from baseline import FEATURES, load_dataset, validation_metrics

ROOT = Path(__file__).resolve().parent
FIELDS = ["row_index", "trial_id", "group_id", "split", "success", "p_success",
          "predicted_success"]
FILE_KEYS = ("predictions", "run_report", "training_code", "model_artifact")
DATA_HASH_KEYS = ("dataset.csv", "split_assignments.csv", "cv_manifest.json")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_PROBABILITY = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _model_id(value) -> str:
    if (not isinstance(value, str) or not _MODEL_ID.fullmatch(value)
            or value == "constant_train_prior"):
        raise ValueError("Invalid or reserved model_id")
    return value


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(content: bytes, description: str) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    try:
        result = json.loads(content.decode("utf-8-sig"), object_pairs_hook=unique_object,
                            parse_constant=invalid_constant)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid {description}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{description} must be a JSON object")
    # The decoder can otherwise turn a large exponent such as 1e999 into infinity.
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f"Nonfinite value in {description}") from exc
    return result


def _file_path(parent: Path, entry) -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ValueError("File entries require exactly path and sha256")
    relative, digest = entry["path"], entry["sha256"]
    if (not isinstance(relative, str) or not relative or "\x00" in relative
            or "\\" in relative or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive):
        raise ValueError("Submission file paths must be portable relative paths")
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("Invalid file sha256")
    # A new manifest parent may not exist yet (for example new/../inputs).
    # Resolve that path without creating outputs, then require the input file.
    path = (parent / relative).resolve()
    if not path.is_file():
        raise ValueError(f"Submission input is not a regular file: {relative}")
    return path


def _prediction_probabilities(content: bytes, validation_rows: list[dict]) -> list[float]:
    expected = {}
    for row in validation_rows:
        trial = row.get("trial_id")
        if (not isinstance(trial, str) or not trial or trial in expected
                or row.get("split") != "validation"
                or str(row.get("success")) not in {"0", "1"}
                or not isinstance(row.get("group_id"), str) or not row["group_id"]
                or not re.fullmatch(r"0|[1-9][0-9]*", str(row.get("row_index")))):
            raise ValueError("Invalid authoritative validation rows")
        expected[trial] = row
    if not expected:
        raise ValueError("Authoritative validation rows must be nonempty")
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader, None) != FIELDS:
            raise ValueError("Predictions require the exact seven-column header")
        found = {}
        for cells in reader:
            if len(cells) != len(FIELDS) or any(not value for value in cells):
                raise ValueError("Missing or extra prediction cells")
            row = dict(zip(FIELDS, cells))
            trial = row["trial_id"]
            if trial not in expected:
                raise ValueError(f"Non-validation or unknown trial_id: {trial}")
            if trial in found:
                raise ValueError(f"Duplicate trial_id: {trial}")
            truth = expected[trial]
            for key in ("row_index", "group_id", "split", "success"):
                if row[key] != str(truth[key]):
                    raise ValueError(f"Authoritative {key} mismatch for {trial}")
            if row["success"] not in {"0", "1"} or row["predicted_success"] not in {"0", "1"}:
                raise ValueError("Prediction labels must be binary 0 or 1")
            try:
                probability = float(row["p_success"])
            except ValueError as exc:
                raise ValueError(f"Invalid p_success for {trial}") from exc
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid p_success for {trial}")
            if int(row["predicted_success"]) != int(probability >= 0.5):
                raise ValueError(f"predicted_success disagrees with p_success >= 0.5 for {trial}")
            if not _PROBABILITY.fullmatch(row["p_success"].strip()):
                raise ValueError(f"Invalid p_success for {trial}")
            found[trial] = probability
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Malformed predictions CSV: {exc}") from exc
    if found.keys() != expected.keys():
        raise ValueError("Predictions do not cover every validation row")
    return [found[row["trial_id"]] for row in validation_rows]


def _equal_metrics(actual, expected) -> bool:
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_equal_metrics(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_equal_metrics(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, float):
        return (type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12))
    return type(actual) is type(expected) and actual == expected


def _validate_report(report: dict, package_audit: dict, code_hash: str,
                     validation_rows: list[dict], probabilities: list[float]) -> None:
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Missing protocol metadata")
    for key, expected in (("fit_partition", "train"), ("score_partition", "validation"),
                          ("features", list(FEATURES)), ("decision_threshold", 0.5)):
        if protocol.get(key) != expected:
            raise ValueError(f"Protocol mismatch: {key}")
    if not isinstance(protocol.get("selection"), str) or not protocol["selection"].strip():
        raise ValueError("Missing protocol selection declaration")
    audit = report.get("audit")
    hashes = audit.get("sha256") if isinstance(audit, dict) else None
    trusted = package_audit.get("sha256") if isinstance(package_audit, dict) else None
    if not isinstance(hashes, dict) or not isinstance(trusted, dict):
        raise ValueError("Missing package hash metadata")
    for name in (*DATA_HASH_KEYS, *(("manifest.json",) if "manifest.json" in hashes else ())):
        value = hashes.get(name)
        if (not isinstance(value, str) or not _HASH.fullmatch(value)
                or value != trusted.get(name)):
            raise ValueError(f"Frozen package hash mismatch: {name}")
    if report.get("code_sha256") != code_hash:
        raise ValueError("Training code hash mismatch")
    versions = report.get("versions")
    if (not isinstance(versions, dict) or not versions
            or any(not isinstance(k, str) or not k.strip()
                   or not isinstance(v, str) or not v.strip() for k, v in versions.items())):
        raise ValueError("Missing or invalid software versions")
    selection = report.get("selection")
    if (not isinstance(selection, dict) or not selection
            or any(not key.strip() for key in selection)):
        raise ValueError("Missing selected hyperparameters")
    if "validation" in report:
        recomputed = validation_metrics(validation_rows, probabilities)
        if not _equal_metrics(report["validation"], recomputed):
            raise ValueError("Reported validation metrics disagree with recomputed metrics")


def _validate_submission(manifest: dict, parent: Path, validation_rows: list[dict],
                         package_audit: dict) -> dict:
    if set(manifest) != {"schema_version", "model_id", "probability_class", "files"}:
        raise ValueError("Invalid submission manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported submission schema_version")
    if type(manifest["probability_class"]) is not int or manifest["probability_class"] != 1:
        raise ValueError("Submission probability_class must be 1")
    model_id = _model_id(manifest["model_id"])
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(FILE_KEYS):
        raise ValueError("Submission requires exactly four input files")
    contents = {}
    for name in FILE_KEYS:
        path = _file_path(parent, files[name])
        if name in {"predictions", "run_report"}:
            contents[name] = path.read_bytes()
            digest = hashlib.sha256(contents[name]).hexdigest()
        else:
            digest = _digest(path)
        if digest != files[name]["sha256"]:
            raise ValueError(f"Submission file hash mismatch: {name}")
    probabilities = _prediction_probabilities(contents["predictions"], validation_rows)
    report = _json(contents["run_report"], "run report")
    _validate_report(report, package_audit, files["training_code"]["sha256"],
                     validation_rows, probabilities)
    return {"model_id": model_id, "probabilities": probabilities, "source_report": report,
            "provenance": {
                "schema_version": 1, "probability_class": 1, "files": files,
                "training_execution_independently_verified": False,
                "model_artifact_deserialized": False,
                "feature_declaration_checked": True,
                "limitations": [
                    "Feature names and training protocol are checked declarations, not model introspection.",
                    "Hash binding does not prove fit data, fold-local preprocessing, or probability origin.",
                    "Prediction validation cannot rule out malicious or misreported training leakage."]}}


def load_submission(manifest_path: Path, validation_rows: list[dict], package_audit: dict) -> dict:
    """Verify a portable manifest and align predictions to authoritative validation order."""
    path = Path(manifest_path)
    content = path.read_bytes()
    result = _validate_submission(_json(content, "submission manifest"), path.parent,
                                  validation_rows, package_audit)
    result["provenance"]["manifest_sha256"] = hashlib.sha256(content).hexdigest()
    return result


def _check_output(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing an existing or symlink submission output")
    resolved = output.resolve()
    protected = [ROOT / "data", ROOT / "protocols"]
    results = ROOT / "results"
    if results.is_dir():
        protected.extend(path for path in results.iterdir() if path.is_dir())
    for directory in protected:
        if resolved == directory.resolve() or directory.resolve() in resolved.parents:
            raise ValueError("Refusing output inside frozen data, protocols, or an existing reference run")


def create_submission(*, model_id: str, predictions: Path, run_report: Path,
                      training_code: Path, model_artifact: Path, output: Path,
                      validation_rows: list[dict], package_audit: dict) -> dict:
    """Create a new portable manifest only after checking all supplied evidence."""
    _model_id(model_id)
    output = Path(output)
    _check_output(output)
    inputs = dict(predictions=predictions, run_report=run_report,
                  training_code=training_code, model_artifact=model_artifact)
    files = {}
    for name, source in inputs.items():
        path = Path(source).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Submission input is not a regular file: {name}")
        if path == output.resolve():
            raise ValueError("Submission output cannot replace an input")
        files[name] = {"path": os.path.relpath(path, output.parent.resolve()),
                       "sha256": _digest(path)}
    manifest = {"schema_version": 1, "model_id": model_id, "probability_class": 1,
                "files": files}
    _validate_submission(manifest, output.parent, validation_rows, package_audit)
    serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    _check_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also refuses a file or symlink installed after the checks.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Validate evidence and create a portable submission")
    create.add_argument("--model-id", required=True)
    for name in (*FILE_KEYS, "output"):
        create.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    create.add_argument("--data-dir", type=Path, default=ROOT / "data" / "v1")
    args = parser.parse_args(argv)
    try:
        _check_output(args.output)
        data_root = args.data_dir.resolve()
        if args.output.resolve() == data_root or data_root in args.output.resolve().parents:
            raise ValueError("Refusing output inside the input data package")
        package = load_dataset(args.data_dir)
        create_submission(model_id=args.model_id, predictions=args.predictions,
                          run_report=args.run_report, training_code=args.training_code,
                          model_artifact=args.model_artifact, output=args.output,
                          validation_rows=[row for row in package["rows"] if row["split"] == "validation"],
                          package_audit=package["audit"])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"created": str(args.output), "model_id": args.model_id}))


if __name__ == "__main__":
    main()

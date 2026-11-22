"""Export all frozen v1 data and feature documentation as one verified ZIP.

Run: uv run --locked python package_data.py
The existing data/v1 files are authoritative and are never changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from baseline import load_dataset

ROOT = Path(__file__).resolve().parent
DATA_FILES = (
    'dataset.csv', 'split_assignments.csv', 'cv_manifest.json',
    'manifest.json', 'source_environment.json', 'source_collection_summary.json',
)
REPOSITORY = 'https://github.com/youseihuayu-wonderful/intelligent-grasping-course-project'
GUIDE = f'''# Grasp dataset v1 — complete data download

This archive contains all six frozen data/v1 files and docs/DATASET.md,
which explains the features, units, labels, grouping, splits and provenance.
The CSV contains 3,840 real simulated grasp-and-hold trials.

Keep the data/v1 directory intact. The CSV and fixed split/CV assignments
are unchanged from the course repository. No simulation videos, models,
older datasets or new labels are mixed into this data-only export.
PACKAGE_MANIFEST.json contains SHA256 hashes for every payload file.

Training code and the reproducible environment are available at:
{REPOSITORY}
From that checkout, validate this extracted dataset with:
uv run --locked python baseline.py --data-dir /path/to/extracted/data/v1 --validate-only

The historical test partition has already been published and inspected.
It is not a new blind final test and must not be used for tuning.

## Simulation demo

The separate recorded simulation can be viewed online at:
{REPOSITORY}#simulation-demo
'''


def digest(content):
    return hashlib.sha256(content).hexdigest()


def package(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite an existing export: {output}')
    data_dir = ROOT / 'data/v1'
    checked = load_dataset(data_dir)
    if {path.name for path in data_dir.iterdir()} != set(DATA_FILES):
        raise ValueError('The frozen v1 directory contains an unexpected or missing file')
    payload = {f'data/v1/{name}': (data_dir / name).read_bytes() for name in DATA_FILES}
    manifest = checked['manifest']
    for name, expected in manifest['files'].items():
        content = payload[f'data/v1/{name}']
        if digest(content) != expected['sha256'] or len(content) != expected['bytes']:
            raise ValueError(f'Frozen metadata file differs from its manifest: {name}')
    payload['docs/DATASET.md'] = (ROOT / 'docs/DATASET.md').read_bytes()
    payload['README.md'] = GUIDE.encode('utf-8')
    inventory = {
        'kind': 'complete_frozen_course_data_export',
        'dataset_version': manifest['dataset_version'],
        'dataset_sha256': manifest['dataset_sha256'],
        'source': manifest['source'],
        'scope': 'All v1 data files and feature documentation; no relabelling, resplitting, training, videos or older batches.',
        'files': {name: {'sha256': digest(content), 'bytes': len(content)}
                  for name, content in sorted(payload.items())},
    }
    payload['PACKAGE_MANIFEST.json'] = (json.dumps(inventory, indent=2, sort_keys=True) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='grasp-data-export-') as temporary:
        temporary = Path(temporary)
        pending = temporary / 'dataset.zip'
        with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content, compresslevel=9)
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(payload):
                raise ValueError('ZIP CRC or file inventory check failed')
            for name, content in payload.items():
                if archive.read(name) != content:
                    raise ValueError(f'Exported bytes differ: {name}')
            # Member paths are the fixed payload names above, never external input.
            extracted = temporary / 'extracted'
            archive.extractall(extracted)
        restored = load_dataset(extracted / 'data/v1')
        if restored['audit'] != checked['audit']:
            raise ValueError('Extracted data, splits or CV differ from the frozen package')
        with pending.open('rb') as source, output.open('xb') as target:
            shutil.copyfileobj(source, target)
    print(json.dumps({
        'file': str(output), 'bytes': output.stat().st_size,
        'sha256': digest(output.read_bytes()), 'members': sorted(payload),
        'rows': manifest['rows'], 'groups': manifest['groups'],
        'verified': 'All source bytes, ZIP CRC, extracted frozen data, partitions and grouped CV',
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'grasp_dataset_v1.zip')
    args = parser.parse_args()
    package(args.output.absolute())


if __name__ == '__main__':
    main()

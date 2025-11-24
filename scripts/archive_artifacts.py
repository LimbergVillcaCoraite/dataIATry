#!/usr/bin/env python3
"""Mueve artefactos intermedios (`*_hpo.joblib`, `*_hpo.metrics.json`) a `artifacts/`.

Uso:
    python3 scripts/archive_artifacts.py --dry-run
    python3 scripts/archive_artifacts.py
"""
from pathlib import Path
import shutil
import argparse


def find_artifacts(models_dir: Path):
    patterns = ['*_hpo.joblib', '*_hpo.metrics.json', '*_hpo*.joblib']
    files = []
    for p in patterns:
        files.extend(models_dir.glob(p))
    return sorted(set(files))


def main(dry_run=False):
    repo = Path(__file__).resolve().parents[1]
    models_dir = repo / 'models'
    out_dir = repo / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_artifacts(models_dir)
    if not files:
        print('No artifacts found to archive.')
        return
    for f in files:
        dest = out_dir / f.name
        print(f"{f} -> {dest}")
        if not dry_run:
            shutil.move(str(f), str(dest))

    if not dry_run:
        print('Move completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='List files without moving')
    args = parser.parse_args()
    main(dry_run=args.dry_run)

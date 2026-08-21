#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${example_dir}/../.." && pwd)"

cd "${repo_root}"
python -m pip install -e .

cd "${example_dir}"
python ../../train.py --config config.yaml

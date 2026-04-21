#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python main.py train --config configs/wsrvos_a2d.yaml "$@"

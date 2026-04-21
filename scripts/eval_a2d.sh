#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python main.py eval --config configs/wsrvos_a2d.yaml "$@"

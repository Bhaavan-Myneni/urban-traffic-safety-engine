#!/usr/bin/env bash
# Backward-compatible wrapper for AI City / WTS subset import.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/fetch_dataset_subset.sh" --dataset aicity "$@"

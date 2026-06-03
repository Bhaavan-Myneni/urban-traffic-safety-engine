#!/usr/bin/env bash
# Import videos from the Kaggle "Road Traffic Video Monitoring" notebook dataset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/fetch_dataset_subset.sh" --dataset kaggle_traffic "$@"

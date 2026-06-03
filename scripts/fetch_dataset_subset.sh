#!/usr/bin/env bash
#
# Copy video files from a local extracted folder into a registered dataset directory,
# rename with a stable prefix, and run register_dataset.py.
#
# Does NOT download data automatically.
#
# Usage:
#   bash scripts/fetch_dataset_subset.sh \
#     --dataset kaggle_traffic \
#     --source-dir /path/to/videos \
#     --limit 10
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGISTER_SCRIPT="${PROJECT_ROOT}/scripts/register_dataset.py"

DATASET=""
SOURCE_DIR=""
LIMIT=""

usage() {
  cat <<'EOF'
Usage: fetch_dataset_subset.sh --dataset NAME --source-dir PATH --limit N

Supported datasets:
  aicity            -> data/raw/aicity_videos/         (prefix: aicity_)
  kaggle_traffic    -> data/raw/kaggle_traffic_videos/ (prefix: kaggle_)
  bdd100k           -> data/raw/bdd100k_videos/         (prefix: bdd100k_)

Options:
  --dataset NAME      Dataset key (required)
  --source-dir PATH   Local folder with videos (searched recursively)
  --limit N           Maximum videos to copy (must be > 0)
  -h, --help          Show this help message

Example (Kaggle notebook source videos):
  bash scripts/fetch_dataset_subset.sh \
    --dataset kaggle_traffic \
    --source-dir ~/Downloads/road-traffic-video-monitoring \
    --limit 10
EOF
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

resolve_dataset_paths() {
  case "${DATASET}" in
    aicity)
      TARGET_DIR="${PROJECT_ROOT}/data/raw/aicity_videos"
      FILE_PREFIX="aicity_"
      DISPLAY_NAME="AI City"
      ;;
    kaggle_traffic)
      TARGET_DIR="${PROJECT_ROOT}/data/raw/kaggle_traffic_videos"
      FILE_PREFIX="kaggle_"
      DISPLAY_NAME="Kaggle Traffic (notebook)"
      ;;
    bdd100k)
      TARGET_DIR="${PROJECT_ROOT}/data/raw/bdd100k_videos"
      FILE_PREFIX="bdd100k_"
      DISPLAY_NAME="BDD100K"
      ;;
    *)
      die "Unknown dataset '${DATASET}'. Use: aicity, kaggle_traffic, bdd100k"
      ;;
  esac
}

parse_args() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        [[ $# -ge 2 ]] || die "Missing value for --dataset"
        DATASET="$2"
        shift 2
        ;;
      --source-dir)
        [[ $# -ge 2 ]] || die "Missing value for --source-dir"
        SOURCE_DIR="$2"
        shift 2
        ;;
      --limit)
        [[ $# -ge 2 ]] || die "Missing value for --limit"
        LIMIT="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1 (use --help)"
        ;;
    esac
  done

  [[ -n "${DATASET}" ]] || die "--dataset is required"
  [[ -n "${SOURCE_DIR}" ]] || die "--source-dir is required"
  [[ -n "${LIMIT}" ]] || die "--limit is required"

  if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
    die "--limit must be a positive integer (got: ${LIMIT})"
  fi
  if [[ "${LIMIT}" -lt 1 ]]; then
    die "--limit must be greater than 0 (got: ${LIMIT})"
  fi

  resolve_dataset_paths
}

main() {
  parse_args "$@"

  local resolved_source
  resolved_source="$(cd "${SOURCE_DIR}" && pwd)"

  if [[ ! -d "${resolved_source}" ]]; then
    die "Source directory does not exist: ${resolved_source}"
  fi

  local video_list
  video_list="$(mktemp)"
  find "${resolved_source}" -type f \( \
    -iname '*.mp4' -o \
    -iname '*.mov' -o \
    -iname '*.avi' -o \
    -iname '*.mkv' \
    \) 2>/dev/null | LC_ALL=C sort > "${video_list}"

  local found=0
  if [[ -s "${video_list}" ]]; then
    found="$(wc -l < "${video_list}" | tr -d ' ')"
  fi

  if [[ "${found}" -eq 0 ]]; then
    rm -f "${video_list}"
    die "No video files found under ${resolved_source} (.mp4, .mov, .avi, .mkv)"
  fi

  mkdir -p "${TARGET_DIR}"

  local copied=0
  local skipped=0
  local skipped_log
  skipped_log="$(mktemp)"
  local src dest ext dest_name

  while IFS= read -r src; do
    [[ -n "${src}" ]] || continue

    if [[ "${copied}" -ge "${LIMIT}" ]]; then
      skipped=$((skipped + 1))
      if [[ "${skipped}" -le 5 ]]; then
        printf '    - %s\n' "$(basename "${src}")" >> "${skipped_log}"
      fi
      continue
    fi

    ext="$(printf '%s' "${src##*.}" | tr '[:upper:]' '[:lower:]')"
    copied=$((copied + 1))
    printf -v dest_name '%s%03d.%s' "${FILE_PREFIX}" "${copied}" "${ext}"
    dest="${TARGET_DIR}/${dest_name}"

    if [[ -e "${dest}" ]]; then
      rm -f "${dest}"
    fi

    cp -f "${src}" "${dest}"
  done < "${video_list}"

  rm -f "${video_list}"

  log ""
  log "=== ${DISPLAY_NAME} Subset Copy ==="
  log "  Dataset          : ${DATASET}"
  log "  Source directory : ${resolved_source}"
  log "  Target directory : ${TARGET_DIR}"
  log "  Videos found     : ${found}"
  log "  Videos copied    : ${copied}"
  log "  Videos skipped   : ${skipped} (limit: ${LIMIT})"
  if [[ "${skipped}" -gt 0 ]]; then
    log "  Skipped files    :"
    cat "${skipped_log}"
    if [[ "${skipped}" -gt 5 ]]; then
      log "    - ... and $((skipped - 5)) more"
    fi
  fi
  rm -f "${skipped_log}"
  log "==============================="
  log ""

  if [[ ! -f "${REGISTER_SCRIPT}" ]]; then
    die "register_dataset.py not found: ${REGISTER_SCRIPT}"
  fi

  log "Registering dataset manifest..."
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    die "python3 or python not found in PATH (or activate .venv)"
  fi

  (
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
    "${PYTHON}" "${REGISTER_SCRIPT}" \
      --video-dir "${TARGET_DIR}" \
      --dataset "${DATASET}"
  )

  log ""
  log "Next step:"
  log "  python scripts/process_dataset.py --dataset ${DATASET}"
  log ""
  log "Then compare datasets:"
  log "  python scripts/compare_datasets.py"
  log ""
}

main "$@"

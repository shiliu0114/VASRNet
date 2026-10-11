#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 MODEL_FOLDER INPUT_FOLDER OUTPUT_FOLDER [extra prediction options]" >&2
  exit 2
fi
MODEL_FOLDER="$1"
INPUT_FOLDER="$2"
OUTPUT_FOLDER="$3"
shift 3

PYTHON="${PYTHON:-python3}"
read -r -a fold_args <<< "${FOLDS:-0}"
"$PYTHON" -m vasrnet_release.predict \
  --input "$INPUT_FOLDER" --output "$OUTPUT_FOLDER" --model "$MODEL_FOLDER" \
  --folds "${fold_args[@]}" \
  --checkpoint "${CHECKPOINT_NAME:-checkpoint_best.pth}" \
  --device "${DEVICE:-cuda}" "$@"

#!/usr/bin/env bash
set -e

# Setup cuDNN runtime path if present in local venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _cudnn in "$SCRIPT_DIR"/.venv/lib/python*/site-packages/nvidia/cudnn/lib; do
    if [ -d "$_cudnn" ]; then
        export LD_LIBRARY_PATH="$_cudnn:${LD_LIBRARY_PATH:-}"
    fi
done

FAMILY="${1:-rtmdet}"
if [ "$FAMILY" = "rtmdet" ] || [ "$FAMILY" = "yolox" ] || [ "$FAMILY" = "yolov1" ]; then
    shift || true
    exec uv run python "training_scripts/$FAMILY/train.py" "$@"
elif [ "$FAMILY" = "python" ]; then
    shift
    exec uv run python "$@"
elif [[ "$FAMILY" == *.py ]]; then
    SCRIPT="$FAMILY"
    shift
    exec uv run python "$SCRIPT" "$@"
else
    exec uv run python training_scripts/rtmdet/train.py "$@"
fi

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/container.env"

if [[ "$UNBRANDING_IMAGE" = /* ]]; then
  IMAGE="$UNBRANDING_IMAGE"
else
  IMAGE="$REPO_ROOT/$UNBRANDING_IMAGE"
fi

if [[ "$UNBRANDING_CACHE" = /* ]]; then
  CACHE="$UNBRANDING_CACHE"
else
  CACHE="$REPO_ROOT/$UNBRANDING_CACHE"
fi

CONTAINER_RUNTIME="$(command -v apptainer || command -v singularity)"

if [[ ! -f "$IMAGE" ]]; then
  echo "container image not found: $IMAGE" >&2
  exit 1
fi

mkdir -p "$CACHE/hf" "$CACHE/vllm/triton"

MODEL_DIR="/net/scratch/hscra/plgrid/plgvltkv/models"

exec "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$CACHE:/cache" \
  --bind "$MODEL_DIR:/models:ro" \
  --pwd /workspace \
  --env "HF_HOME=/cache/hf" \
  --env "HUGGINGFACE_HUB_CACHE=/cache/hf/hub" \
  --env "VLLM_CACHE_ROOT=/cache/vllm" \
  --env "TRITON_CACHE_DIR=/cache/vllm/triton" \
  --env "PYTHONNOUSERSITE=1" \
  "$IMAGE" python3 "$@"

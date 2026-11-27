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


# Find container runtime
CONTAINER_RUNTIME="$(command -v apptainer || command -v singularity || true)"

if [[ ! -f "$IMAGE" ]]; then
  echo "container image not found: $IMAGE" >&2
  exit 1
fi

# Resolve model directory
MODEL_DIR="$SCRATCH/models"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: Model directory not found: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$CACHE/hf" "$CACHE/vllm/triton"

echo "=========================================="
echo "Repository: $REPO_ROOT"
echo "Container: $IMAGE"
echo "Cache: $CACHE"
echo "Models: $MODEL_DIR"
echo "=========================================="

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

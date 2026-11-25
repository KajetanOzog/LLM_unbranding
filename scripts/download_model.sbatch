#!/bin/bash
#SBATCH --job-name=download-models
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/download-models-%j.out
#SBATCH --error=logs/download-models-%j.err

set -euo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# User-specific scratch directory
if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: SCRATCH environment variable is not set."
    exit 1
fi

MODEL_ROOT="$SCRATCH/models"
CONTAINER="$REPO_ROOT/containers/vllm-0.26.0.sif"

# Caches on user's scratch
export APPTAINER_CACHEDIR="$SCRATCH/.cache/apptainer"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$MODEL_ROOT"
mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p "$HF_HOME"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-4-12B-it"
    "microsoft/phi-4"
)

echo "=========================================="
echo "Repository: $REPO_ROOT"
echo "Scratch: $SCRATCH"
echo "Models directory: $MODEL_ROOT"
echo "Container: $CONTAINER"
echo "Checking ${#MODELS[@]} models"
echo "=========================================="

hostname
df -h "$SCRATCH"

export HF_HUB_DISABLE_XET=1

DOWNLOADED=0
SKIPPED=0

for MODEL_ID in "${MODELS[@]}"; do
    MODEL_NAME="${MODEL_ID//\//__}"
    MODEL_DIR="$MODEL_ROOT/$MODEL_NAME"

    echo
    echo "=========================================="
    echo "Model: $MODEL_ID"
    echo "Destination: $MODEL_DIR"
    echo "=========================================="

    # Skip if model directory already exists and is not empty
    if [[ -d "$MODEL_DIR" ]] && [[ -n "$(ls -A "$MODEL_DIR")" ]]; then
        echo "SKIPPING: Model directory already exists."
        du -sh "$MODEL_DIR"
        ((SKIPPED+=1))
        continue
    fi

    echo "Downloading model..."
    mkdir -p "$MODEL_DIR"

    apptainer exec --cleanenv \
        --bind "$SCRATCH:$SCRATCH" \
        --env HF_HOME="$HF_HOME" \
        --env HF_HUB_CACHE="$HF_HUB_CACHE" \
        --env HF_HUB_DISABLE_XET=1 \
        --env MODEL_ID="$MODEL_ID" \
        --env MODEL_DIR="$MODEL_DIR" \
        "$CONTAINER" \
        python3 -c '
from huggingface_hub import snapshot_download
import os

path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=os.environ["MODEL_DIR"],
)

print(f"Downloaded to: {path}")
'

    echo "Finished: $MODEL_ID"
    du -sh "$MODEL_DIR"
    ((DOWNLOADED+=1))
done

echo
echo "=========================================="
echo "ALL CHECKS FINISHED"
echo "Downloaded: $DOWNLOADED"
echo "Skipped: $SKIPPED"
echo "=========================================="

du -sh "$MODEL_ROOT"/*

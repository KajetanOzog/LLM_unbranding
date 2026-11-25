#!/bin/bash
#SBATCH --job-name=judge-llama
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --qos=normal
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/judge-%j.out
#SBATCH --error=logs/judge-%j.out
set -euo pipefail

# Model directory
INPUT_DIR="results/qwen3-8b"

shopt -s nullglob
RUNS=("$INPUT_DIR"/*.jsonl)

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "No JSONL files found in $INPUT_DIR" >&2
    exit 1
fi

if [[ ${#RUNS[@]} -ne 20 ]]; then
    echo "Expected 20 files, found ${#RUNS[@]}." >&2
    exit 1
fi

echo "Model directory: $INPUT_DIR"
echo "Number of files: ${#RUNS[@]}"
echo "Judge model: Qwen3-32B"

# One process, one model loading, all files
exec scripts/container.sh eval/judge.py --run "${RUNS[@]}"

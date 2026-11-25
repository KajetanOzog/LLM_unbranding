#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --array=0-19%4
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/gen-%A_%a.out
#SBATCH --error=logs/gen-%A_%a.out

set -euo pipefail

MODELS=(
    # "llama-3.1-8b-instruct"
    # "mistral-7b-instruct"
    # "qwen3-8b"
    "qwen3-14b"
)

BRANDS=(
    "auto/audi"
    "auto/bmw"
    "auto/porsche"
    "auto/tesla"
    "beverages/coca_cola"
    "beverages/monster"
    "beverages/pepsi"
    "beverages/red_bull"
    "food/dominos"
    "food/kfc"
    "food/mcdonalds"
    "food/subway"
    "sport/adidas"
    "sport/new_balance"
    "sport/nike"
    "sport/puma"
    "tech/apple"
    "tech/google"
    "tech/microsoft"
    "tech/samsung"
)

NUM_BRANDS=${#BRANDS[@]}

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_BRANDS))
BRAND_IDX=$((SLURM_ARRAY_TASK_ID % NUM_BRANDS))

MODEL="${MODELS[$MODEL_IDX]}"
BRAND_PATH="${BRANDS[$BRAND_IDX]}"
BRAND="${BRAND_PATH##*/}"

EVAL_DIR="dataset/eval/$BRAND_PATH"
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR/$MODEL" logs

echo "Model: $MODEL"
echo "Brand: $BRAND"
echo "Eval dir: $EVAL_DIR"
echo "Output: $OUTPUT_DIR/$MODEL/$BRAND.jsonl"

exec scripts/container.sh eval/generate.py \
    --model "$MODEL" \
    --eval-dir "$EVAL_DIR" \
    --out "$OUTPUT_DIR"
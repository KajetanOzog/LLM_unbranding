#!/bin/bash
#SBATCH --job-name=pull-container
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pull_container_%j.out
#SBATCH --error=logs/pull_container_%j.err
set -e

echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
cd "$SCRATCH/LLM_unbranding"

# Store Apptainer cache on scratch to avoid home quota
export APPTAINER_CACHEDIR="$SCRATCH/.cache/apptainer"

mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p logs

echo "APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"

bash containers/pull.sh

echo "Container pull finished successfully."
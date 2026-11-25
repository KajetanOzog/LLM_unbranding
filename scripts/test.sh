#!/bin/bash
#SBATCH --job-name=test-container
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=test-container-%j.log

cd /net/scratch/hscra/plgrid/plgvltkv/LLM_unbranding

echo "Architecture:"
uname -m

echo "Container test:"
scripts/container.sh -c '
import os
p = "/models/meta-llama__Meta-Llama-3.1-8B-Instruct"
print("Model exists:", os.path.isdir(p))
print("Config exists:", os.path.isfile(p + "/config.json"))
print("Tokenizer exists:", os.path.isfile(p + "/tokenizer.json"))
'
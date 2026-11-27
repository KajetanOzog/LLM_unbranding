#!/usr/bin/env bash
# Run the per-brand hyperparameter sweep used for the weight-space baselines
# in arXiv:2609.37127. Execute from open-unlearning/.
set -euo pipefail

MODEL_CONFIG="${MODEL_CONFIG:-Llama-3.1-8B-Instruct}"
case "$MODEL_CONFIG" in
  Llama-3.1-8B-Instruct) DEFAULT_MODEL=/models/meta-llama__Meta-Llama-3.1-8B-Instruct ;;
  Qwen3-8B) DEFAULT_MODEL=/models/Qwen__Qwen3-8B ;;
  Qwen3-14B) DEFAULT_MODEL=/models/Qwen__Qwen3-14B ;;
  Qwen3-32B) DEFAULT_MODEL=/models/Qwen__Qwen3-32B ;;
  Mistral-7B-Instruct-v0.3) DEFAULT_MODEL=/models/mistralai__Mistral-7B-Instruct-v0.3 ;;
  Gemma-4-12B-it) DEFAULT_MODEL=/models/google__gemma-4-12B-it ;;
  Phi-4) DEFAULT_MODEL=/models/microsoft__phi-4 ;;
  *) echo "Unknown MODEL_CONFIG: $MODEL_CONFIG" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DEFAULT_MODEL}"
METHODS="${METHODS:-NPO SimNPO GradAscent GradDiff}"
LRS="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
EPOCHS="${EPOCHS:-1 2 3 4 5 6 7 8}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f src/train.py || ! -d ../dataset/train ]]; then
  echo "Run this script from LLM_unbranding/open-unlearning." >&2
  exit 2
fi

while read -r brand category slug; do
  [[ -z "${brand:-}" ]] && continue
  for method in $METHODS; do
    for lr in $LRS; do
      for epochs in $EPOCHS; do
        task="unbranding/${MODEL_CONFIG}/${brand}/${method}/lr${lr}_ep${epochs}"
        cmd=(python src/train.py --config-name=unlearn.yaml
          experiment=unlearn/unbranding
          "brand=${brand}" "brand_slug=${slug}" "category=${category}"
          "trainer=${method}" "model=${MODEL_CONFIG}"
          "model.model_args.pretrained_model_name_or_path=${MODEL}"
          "model.tokenizer_args.pretrained_model_name_or_path=${MODEL}"
          "trainer.args.learning_rate=${lr}"
          "trainer.args.num_train_epochs=${epochs}"
          "task_name=${task}")
        printf -v rendered '%q ' "${cmd[@]}"
        if [[ "$DRY_RUN" == 1 ]]; then
          echo "$rendered"
        else
          "${cmd[@]}"
        fi
      done
    done
  done
done <<'BRANDS'
Audi auto audi
BMW auto bmw
Porsche auto porsche
Tesla auto tesla
Coca-Cola beverages coca-cola
Monster beverages monster
Pepsi beverages pepsi
Red-Bull beverages red-bull
Dominos food dominos
KFC food kfc
McDonalds food mcdonalds
Subway food subway
Adidas sport adidas
New-Balance sport new-balance
Nike sport nike
Puma sport puma
Apple tech apple
Google tech google
Microsoft tech microsoft
Samsung tech samsung
BRANDS

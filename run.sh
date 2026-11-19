#!/usr/bin/env bash
# Full pipeline: generate data, train baseline + fine-tuned DistilBERT,
# re-evaluate and plot the confusion matrix.
set -e
cd "$(dirname "$0")"
python src/generate_data.py
python src/train.py
python src/evaluate.py
echo "Done. See results/metrics.json and screenshots/confusion_matrix.png"

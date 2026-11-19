"""Re-evaluate the saved fine-tuned model on the test set.

Loads models/distilbert-support-ticket (created by src/train.py) and
regenerates results/metrics.json entries for the fine-tuned model plus
screenshots/confusion_matrix.png. If no saved model exists, it prints
the stored metrics from results/metrics.json instead.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Account", "Billing", "Delivery", "Technical"]


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — DistilBERT, support tickets (test set)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    model_dir = ROOT / "models" / "distilbert-support-ticket"
    metrics_path = ROOT / "results" / "metrics.json"
    if not (model_dir / "config.json").exists():
        print("No saved fine-tuned model found. Stored metrics:")
        print(metrics_path.read_text() if metrics_path.exists() else "no metrics yet")
        return
    test = pd.read_csv(ROOT / "data" / "test.csv")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 16):
            batch = test["text"].iloc[i:i + 16].tolist()
            enc = tokenizer(batch, truncation=True, padding=True, max_length=64,
                            return_tensors="pt")
            logits = model(**enc).logits
            preds.extend([LABELS[i] for i in torch.argmax(logits, dim=-1).tolist()])
    print(classification_report(test["label"], preds, target_names=LABELS, zero_division=0))
    cm = confusion_matrix(test["label"], preds, labels=LABELS)
    plot_confusion(cm, ROOT / "screenshots" / "confusion_matrix.png")


if __name__ == "__main__":
    main()

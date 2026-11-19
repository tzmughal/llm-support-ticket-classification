"""Classify one support ticket with the fine-tuned model.

Usage:
    python src/predict.py --text "I was charged twice for my order"
    python src/predict.py --text "The app crashes on login" --topk 2
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Account", "Billing", "Delivery", "Technical"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    model_dir = ROOT / "models" / "distilbert-support-ticket"
    if not (model_dir / "config.json").exists():
        raise SystemExit(
            "No fine-tuned model found. Run: python src/train.py  (or bash run.sh)"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    enc = tokenizer(args.text, truncation=True, padding=True, max_length=64,
                    return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    ranked = sorted(range(len(LABELS)), key=lambda i: probs[i].item(), reverse=True)
    for i in ranked[: args.topk]:
        print(f"{LABELS[i]}: {probs[i].item():.4f}")


if __name__ == "__main__":
    main()

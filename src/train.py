"""Fine-tune DistilBERT end-to-end for support-ticket classification.

All transformer parameters are trained (no layers frozen). A TF-IDF +
Logistic Regression baseline is trained first for comparison, using
scikit-learn. Results are written to results/metrics.json.

CPU settings: max_len 64, batch 16, 3 epochs, lr 2e-5. On this machine
(4 CPU cores) the fine-tune takes roughly 10-20 minutes.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "distilbert-base-uncased"
LABELS = ["Account", "Billing", "Delivery", "Technical"]  # alphabetical = sklearn order
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
EPOCHS = 2  # 2-3 epochs per brief; 2 chosen after a runtime restart killed
# a 3-epoch run after epoch 1 (512s/epoch on this shared CPU)
BATCH = 16
LR = 2e-5
MAX_LEN = 64
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            list(texts), truncation=True, padding=True, max_length=MAX_LEN
        )
        self.labels = [LABEL2ID[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def report_dict(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": {label: report[label]["f1-score"] for label in LABELS},
        "per_class_precision": {label: report[label]["precision"] for label in LABELS},
        "per_class_recall": {label: report[label]["recall"] for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": LABELS,
    }


def train_baseline(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    x_train = vec.fit_transform(train["text"])
    x_test = vec.transform(test["text"])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(x_train, train["label"])
    pred = clf.predict(x_test)
    result = report_dict(list(test["label"]), list(pred))
    result["model"] = "TF-IDF (1-2 grams, 5000 features) + Logistic Regression"
    result["test_size"] = len(test)
    return result


def train_distilbert(train, test):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL2ID,
    )
    # End-to-end fine-tuning: every parameter trains, nothing frozen.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} / {total:,}")

    train_ds = TicketDataset(train["text"], train["label"], tokenizer)
    test_ds = TicketDataset(test["text"], test["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * steps), num_training_steps=steps
    )
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_loss, batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += out.loss.item()
            batches += 1
        print(f"epoch {epoch + 1}/{EPOCHS}  mean loss {epoch_loss / batches:.4f}  "
              f"elapsed {time.time() - t0:.0f}s", flush=True)
        # Checkpoint after every epoch, so an interruption never loses all work.
        ckpt = ROOT / "models" / f"checkpoint-epoch-{epoch + 1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    y_true = [LABEL2ID[label] for label in test["label"]]
    pred_labels = [LABELS[i] for i in preds]
    result = report_dict(list(test["label"]), pred_labels)
    result["model"] = "DistilBERT (distilbert-base-uncased), fine-tuned end-to-end"
    result["test_size"] = len(test)
    result["epochs"] = EPOCHS
    result["batch_size"] = BATCH
    result["learning_rate"] = LR
    result["max_len"] = MAX_LEN
    result["trainable_parameters"] = trainable
    result["training_seconds"] = round(time.time() - t0, 1)

    # Save the fine-tuned model locally. Weights are git-ignored (see
    # .gitignore); config, tokenizer and label map are small and tracked.
    out_dir = ROOT / "models" / "distilbert-support-ticket"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return result


def main():
    set_seed(SEED)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    print(f"train={len(train)} test={len(test)}")

    print("Training baseline (TF-IDF + Logistic Regression)...", flush=True)
    baseline = train_baseline(train, test)
    print(f"baseline accuracy={baseline['accuracy']:.4f} macro-F1={baseline['macro_f1']:.4f}")

    print("Fine-tuning DistilBERT end-to-end...", flush=True)
    tuned = train_distilbert(train, test)
    print(f"distilbert accuracy={tuned['accuracy']:.4f} macro-F1={tuned['macro_f1']:.4f}")

    metrics = {
        "dataset": "synthetic support tickets (seed 42), 1000 tickets, 250 per class",
        "train_size": len(train),
        "test_size": len(test),
        "labels": LABELS,
        "baseline": baseline,
        "fine_tuned": tuned,
    }
    out = ROOT / "results" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

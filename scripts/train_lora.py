#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, random, json
import numpy as np
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset routing
# -------------------------
DATASETS = {
    "sst2":   {"hf": ("glue", "sst2")},
    "imdb":   {"hf": ("imdb",)},
    "ag_news":{"hf": ("ag_news",)},
}

def load_task(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Choose from {list(DATASETS.keys())}")
    ds = load_dataset(*DATASETS[name]["hf"])
    # text column
    if "sentence" in ds["train"].column_names:
        text_col = "sentence"
    elif "text" in ds["train"].column_names:
        text_col = "text"
    else:
        # fallback: first string column
        text_col = next(
            (c for c in ds["train"].column_names if isinstance(ds["train"][c][0], str)),
            None
        )
        if text_col is None:
            raise ValueError("Could not find a text column.")
    # label column
    if "label" in ds["train"].column_names:
        label_col = "label"
    elif "coarse_label" in ds["train"].column_names:
        label_col = "coarse_label"
    else:
        raise ValueError("Could not find a label column.")
    # num labels
    num_labels = len(set(ds["train"][label_col]))
    return ds, text_col, label_col, num_labels


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sst2","imdb","ag_news"], required=True)
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--limit_train", type=int, default=0, help="Use first N train rows (0 = all)")
    ap.add_argument("--limit_val", type=int, default=0, help="Use first N val rows (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--num_proc", type=int, default=2, help="Tokenization workers")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -------- Load + prepare splits
    ds, text_col, label_col, num_labels = load_task(args.dataset)

    # add validation if missing (IMDB/AG News)
    if "validation" not in ds:
        try:
            split = ds["train"].train_test_split(test_size=0.1, seed=args.seed, stratify_by_column=label_col)
        except Exception:
            split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
        ds = DatasetDict({"train": split["train"], "validation": split["test"], "test": ds["test"]})

    # IMPORTANT: GLUE/SST-2 public test has no labels → use validation for final metrics
    test_split = "validation" if args.dataset == "sst2" else "test"

    # -------- Tokenization
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok_fn(batch):
        return tok(batch[text_col], truncation=True, max_length=args.max_length)

    ds = ds.map(tok_fn, batched=True, num_proc=args.num_proc)
    ds = ds.rename_column(label_col, "labels")

    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    # Optional subsetting for speed
    if args.limit_train and args.limit_train > 0:
        n = min(args.limit_train, len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(n))
    if args.limit_val and args.limit_val > 0:
        n = min(args.limit_val, len(ds["validation"]))
        ds["validation"] = ds["validation"].shuffle(seed=args.seed).select(range(n))

    # -------- Model + LoRA
    base = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="SEQ_CLS", target_modules=["query","value"]
    )
    model = get_peft_model(base, lcfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")

    # -------- Trainer
    collator = DataCollatorWithPadding(tokenizer=tok)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")
        }

    run_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}_r{args.rank}_ml{args.max_length}_bs{args.batch_size}"
    )
    os.makedirs(run_dir, exist_ok=True)

    targs = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=[],
        fp16=torch.cuda.is_available(),  # enable fp16 if GPU available
        dataloader_num_workers=2,
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # -------- Final evaluation on held-out split
    metrics = trainer.evaluate(ds[test_split])
    print("Final metrics:", metrics)

    # -------- Persist artifacts
    trainer.save_model(run_dir)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # summary row (CSV)
    try:
        import pandas as pd
        row = {
            "dataset": args.dataset,
            "rank": args.rank,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "accuracy": float(metrics.get("eval_accuracy", 0.0)),
            "f1": float(metrics.get("eval_f1", 0.0)),
            "trainable_params_pct": round(trainable_pct, 2)
        }
        pd.DataFrame([row]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    except Exception as e:
        print("Warning: could not write summary.csv:", e)

    print(f"Saved model and metrics to: {run_dir}")


if __name__ == "__main__":
    main()

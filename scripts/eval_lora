import argparse, os, json, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score

def load_task(dataset_name: str):
    DATASETS = {
        "sst2": ("glue", "sst2"),
        "imdb": ("imdb",),
        "ag_news": ("ag_news",)
    }
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    ds = load_dataset(*DATASETS[dataset_name])
    text_col = "sentence" if "sentence" in ds["train"].column_names else "text"
    label_col = "label" if "label" in ds["train"].column_names else ("coarse_label" if "coarse_label" in ds["train"].column_names else None)
    if label_col is None:
        raise ValueError("Label column not found.")
    test_split = "test" if "test" in ds else "validation"
    return ds, text_col, label_col, test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sst2","imdb","ag_news"])
    ap.add_argument("--adapter_dir", required=True, help="Path to the trained adapter dir")
    ap.add_argument("--model_name", default="roberta-base")
    args = ap.parse_args()

    ds, text_col, label_col, test_split = load_task(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
def tokenize(batch):
    return tokenizer(batch[text_col], truncation=True, max_length=256)
ds = ds.map(tokenize, batched=True, num_proc=2)

ds = ds.rename_column(label_col, "labels")
keep_cols = ["input_ids", "attention_mask", "labels"]
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])

base = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, num_labels=len(set(ds["train"]["labels"]))
)
model = PeftModel.from_pretrained(base, args.adapter_dir)

from torch.utils.data import DataLoader
collator = DataCollatorWithPadding(tokenizer=tokenizer)
dl = DataLoader(ds[test_split], batch_size=32, shuffle=False, collate_fn=collator)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

all_preds, all_labels = [], []
with torch.inference_mode():
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        labels = batch["labels"].cpu().numpy().tolist()
        all_preds.extend(preds); all_labels.extend(labels)


    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    metrics = {"accuracy": acc, "f1": f1}
    print(metrics)

    with open(os.path.join(args.adapter_dir, "eval_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

if __name__ == "__main__":
    main()

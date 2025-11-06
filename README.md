# Parameter-Efficient Fine-Tuning with LoRA: Replication and Extension

**Course:** COMP8240 – Project Report  
**Author:** Muhammad Zaran Zahir (47997222)

## Overview
This project replicates the results of *Hu et al. (2022)* “Low-Rank Adaptation of Large Language Models (LoRA)” and extends them to new datasets (IMDB, AG News, TREC, IMDB-Mini-Paraphrased).  
It explores reproducibility and efficiency trade-offs of parameter-efficient fine-tuning (PEFT).

##  Structure
scripts/ → training & evaluation scripts
data_notes/ → dataset descriptions & sources
results/ → output metrics, logs, plots
report/ → LaTeX report + compiled PDF

##  Environment
Developed on Google Colab Pro (T4 GPU, 16 GB VRAM).  
Requirements (install via `pip install -r requirements.txt`):
torch>=2.0
transformers>=4.37
datasets
peft
scikit-learn
matplotlib


##  Usage
1. Clone repo and install requirements  
2. Run training:
   ```bash
   python scripts/train_lora.py --dataset sst2 --rank 8
   ```

## Results
| Dataset   | Accuracy (%) | F1 (%) | Trainable Params (%) |
| --------- | ------------ | ------ | -------------------- |
| SST-2     | 93.7         | 93.6   | 0.9                  |
| IMDB      | 88.0         | 87.8   | 0.9                  |
| IMDB-Mini | 89.6         | 89.2   | 0.9                  |

## Evaluate results:
  ```bash
  python scripts/eval_lora.py --dataset imdb
  ```
## 📓 Notebooks
- [`notebooks/LoRA.ipynb`](notebooks/LoRA.ipynb) — contains full replication (SST-2) and extension experiments (IMDB, AG News, TREC, IMDB-Mini).
  - Run directly in **Google Colab** using the "Open in Colab" badge below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/comp8240-lora-replication/blob/main/notebooks/LoRA.ipynb)

# Train LoRA on SST-2 (replication)
   ```bash
   python scripts/train_lora.py --dataset sst2 --rank 8 --epochs 3 --batch_size 16 --lr 2e-5 --output_dir results/lora_out
   ```
# Train LoRA on IMDB (extension)
```bash
   python scripts/train_lora.py --dataset imdb --rank 8 --epochs 3 --batch_size 16 --lr 2e-5 --output_dir results/lora_out
```
# Evaluate a trained adapter
```bash
   python scripts/eval_lora.py --dataset imdb --adapter_dir results/lora_out/imdb_r8 --model_name roberta-base
```
# Print environment
```bash
python scripts/check_env.py
```
## Reference
  Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

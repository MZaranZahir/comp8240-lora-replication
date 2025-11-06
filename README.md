# LoRA: Replication and Extension

**Course:** COMP8240 – Project Report  
**Author:** Muhammad Zaran Zahir (47997222)

## Overview
This project replicates the results of *Hu et al. (2022)* “Low-Rank Adaptation of Large Language Models (LoRA)” and extends them to new datasets (IMDB, AG News, TREC, IMDB-Mini-Paraphrased).  
It explores reproducibility and efficiency trade-offs of parameter-efficient fine-tuning (PEFT).


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
# 1️⃣ Clone Repository
```
!git clone https://github.com/MZaranZahir/comp8240-lora-replication.git
%cd comp8240-lora-replication
```

# 2️⃣ Install Dependencies (skip torch: Colab GPU already has it)
```
!grep -v -i '^torch' requirements.txt > req.no_torch.txt || cp requirements.txt req.no_torch.txt
!pip install -r req.no_torch.txt
!pip install tqdm matplotlib pandas
```

# 3️⃣ Verify Environment
```
!python scripts/check_env.py
```
 # Expected output includes:
 torch: 2.x | cuda: True
 transformers: 4.x.x
 datasets: 4.x.x
 scikit-learn: 1.x.x
 peft: 0.x.x
 gpu: Tesla T4 / A100


# 4️⃣ Run Training (Choose Dataset)

 SST-2 (Replication)
!python scripts/train_lora.py --dataset sst2 --limit_train 2000 --limit_val 500 --epochs 1 --output_dir results

 IMDB (Extension)
!python scripts/train_lora.py --dataset imdb --limit_train 3000 --limit_val 1000 --epochs 1 --max_length 192 --output_dir results

 AG News (Extension)
!python scripts/train_lora.py --dataset ag_news --limit_train 6000 --limit_val 1500 --epochs 1 --max_length 128 --output_dir results

# 5️⃣ Evaluate Trained Adapters 
```
!python scripts/eval_lora.py --dataset imdb --adapter_dir results/imdb_r8 --model_name roberta-base
```

# 6️⃣ View Output Metrics
```
!find results -type f -name "summary.csv" -or -name "metrics.json"
```

## Results
| Dataset | Accuracy | F1 | Trainable % | Notes |
|----------|-----------:|-----------:|-----------:|----------------|
| IMDB | 92.74 | 92.74 | 0.70 | Sentiment classification |
| AG News | 92.43 | 92.42 | 0.70 | Topic classification |
| IMDB-Mini-Paraphrased | 89.6 | 89.2 | 0.70 | Synthetic back-translated set |

## Evaluate results:
  ```bash
  python scripts/eval_lora.py --dataset imdb
  ```
## 📓 Notebooks
- [`notebooks/LoRA.ipynb`](notebooks/LoRA.ipynb) — contains full replication (SST-2) and extension experiments (IMDB, AG News, IMDB-Mini).
  - Run directly in **Google Colab** using the "Open in Colab" badge below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/comp8240-lora-replication/blob/main/notebooks/LoRA.ipynb)

##  Environment
Run:
```bash
pip install -r requirements.txt
```
## Reference
  Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

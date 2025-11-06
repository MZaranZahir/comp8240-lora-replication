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
## Reference
  Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

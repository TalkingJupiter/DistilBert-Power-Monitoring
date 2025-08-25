#!/bin/bash
#SBATCH --job-name=distill_preproc
#SBATCH --output=logs/preproc_%j.out
#SBATCH --error=logs/preproc_%j.err
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs data
source ~/.bashrc
conda activate distilbert

export TOKENIZERS_PARALLELISM=true

echo "[$(date)] Step 1/3: ensure data/dump.txt from Amazon Polarity…"
if [ ! -f "data/dump.txt" ]; then
  python generate_dataset.py
else
  echo "data/dump.txt already exists, skipping."
fi

echo "[$(date)] Step 2/3: binarize → data/binarized_text.bert-base-uncased.pickle…"
if [ ! -f "data/binarized_text.bert-base-uncased.pickle" ]; then
  python scripts/binarized_data.py \
    --file_path data/dump.txt \
    --tokenizer_type bert \
    --tokenizer_name bert-base-uncased \
    --dump_file data/binarized_text
else
  echo "binarized_text.bert-base-uncased.pickle exists, skipping."
fi

echo "[$(date)] Step 3/3: token counts → data/token_counts.bert-base-uncased.pickle…"
if [ ! -f "data/token_counts.bert-base-uncased.pickle" ]; then
  python scripts/token_counts.py \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts_dump data/token_counts.bert-base-uncased.pickle \
    --vocab_size 30522
else
  echo "token_counts.bert-base-uncased.pickle exists, skipping."
fi

echo "[$(date)] Preprocessing done."

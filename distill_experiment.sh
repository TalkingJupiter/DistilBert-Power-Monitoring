#!/bin/bash
#SBATCH --job-name=distill_train
#SBATCH --output=logs/%A/train_%A_%a.out
#SBATCH --error=logs/%A/train_%A_%a.err
#SBATCH --partition=h100         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3


set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs serialization_dir

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate distilbert
set -u

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true


# -------------------------
# Print SLURM job info
# -------------------------
echo "====================================================="
echo " SLURM JOB INFO"
echo "-----------------------------------------------------"
echo " Job ID           : $SLURM_JOB_ID"
echo " Array Job ID     : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo " Array Task Index : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo " Node List        : $SLURM_JOB_NODELIST"
echo " Partition        : $SLURM_JOB_PARTITION"
echo " Submitted From   : $SLURM_SUBMIT_DIR"
echo " Start Time       : $(date)"
echo "====================================================="
echo

# Verify required files from preprocessing
for f in \
  data/binarized_text.bert-base-uncased.pickle \
  data/token_counts.bert-base-uncased.pickle \
; do
  [[ -f "$f" ]] || { echo "Missing required file: $f" >&2; exit 2; }
done

mkdir -p "logs/${SLURM_JOB_ID}"


JOB_TS=$(date +%Y%m%d_%H%M%S)_${SLURM_ARRAY_TASK_ID:-0}
MONITOR_LOG="logs/${SLURM_JOB_ID}/monitor_${JOB_TS}.out"
MONITOR_JSON="logs/${SLURM_JOB_ID}/power_log_${JOB_TS}.jsonl"

echo "[$(date)] Starting GPU power monitoring…"
nohup python monitor.py --gpu 0 --interval 5 --log_path "$MONITOR_JSON" > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > "logs/${SLURM_JOB_ID}/monitor_${JOB_TS}.pid"
sleep 5
if ! ps -p "$MONITOR_PID" >/dev/null; then
  echo "monitor.py failed to start" >&2
  exit 1
fi

cleanup() {
  echo "[$(date)] Stopping GPU power monitoring (PID $MONITOR_PID)…"
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  echo "[$(date)] Monitoring stopped."
}
trap cleanup EXIT INT TERM

echo "[$(date)] Starting distillation training…"
python train.py \
  --student_type distilbert \
  --student_config training_configs/distilbert-base-uncased.json \
  --teacher_type bert \
  --teacher_name bert-base-uncased \
  --mlm \
  --alpha_ce 0.5 \
  --alpha_mlm 0.5 \
  --alpha_clm 0.0 \
  --mlm_mask_prop 0.15 \
  --batch_size 5 \
  --n_epoch 3 \
  --data_file data/binarized_text.bert-base-uncased.pickle \
  --token_counts data/token_counts.bert-base-uncased.pickle \
  --dump_path serialization_dir/exp_"$JOB_TS" \
  --force

echo "[$(date)] Training complete."

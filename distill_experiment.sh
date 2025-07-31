#!/bin/bash
#SBATCH --job-name=distill_monitor
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=h100
#SBATCH --nodelist=rpg-93-1
#SBATCH --exclusive

source ~/.bashrc
conda activate distilbert

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

JOB_TS=$(date +%Y%m%d_%H%M%S)
MONITOR_LOG="logs/monitor_${JOB_TS}.out"
MONITOR_JSON="logs/power_log_${JOB_TS}.jsonl"

echo "[$(date)] Starting GPU power monitoring..."
nohup python monitor.py --gpu 0 --interval 5 --log_path $MONITOR_JSON > $MONITOR_LOG 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > logs/monitor_${JOB_TS}.pid
echo "[$(date)] Monitoring PID: $MONITOR_PID"

# Ensure monitor.py actually started
sleep 5
if ! ps -p $MONITOR_PID > /dev/null; then
  echo "[$(date)] monitor.py failed to start. Exiting." >&2
  exit 1
fi

trap "echo 'Caught termination signal. Killing monitor PID $MONITOR_PID'; kill $MONITOR_PID; exit" INT TERM

echo "[$(date)] Starting distillation process..."
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
  --dump_path serialization_dir/exp_$JOB_TS \
  --force

echo "[$(date)] Stopping GPU power monitoring..."
kill $MONITOR_PID
wait $MONITOR_PID 2>/dev/null

echo "[$(date)] Monitoring stopped."
echo "[$(date)] Job complete."

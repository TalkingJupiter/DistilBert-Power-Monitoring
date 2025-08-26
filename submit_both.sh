#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

CPU_JOB_ID=$(sbatch --parsable preprocess.sh)
echo "Submitted CPU job: ${CPU_JOB_ID}"

# remove --dependency to run both in parallel
GPU_JOB_ID=$(sbatch --parsable --dependency=afterok:${CPU_JOB_ID} distill_experiment.sh)
echo "Submitted GPU job (afterok ${CPU_JOB_ID}): ${GPU_JOB_ID}"

#!/usr/bin/env bash
set -e -o pipefail

mkdir -p logs serialization_dir

# --- Conda bootstrap (non-interactive safe) ---
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "[ERROR] conda not found. Load your anaconda/miniconda module or install it." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_NAME="distilbert"
PYTHON_VER="${PYTHON_VER:-3.10}"
ENV_YML="${ENV_YML:-environment.yml}"
REQS_TXT="${REQS_TXT:-requirements.txt}"

# Prefer mamba if available (faster)
if command -v mamba >/dev/null 2>&1; then PM="mamba"; else PM="conda"; fi

# --- Create env if missing ---
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Creating env '${ENV_NAME}' (python=${PYTHON_VER})…"
  "${PM}" create -y -n "${ENV_NAME}" "python=${PYTHON_VER}"
else
  echo "[INFO] Env '${ENV_NAME}' already exists."
fi

set +u
conda activate "${ENV_NAME}"
set -u

# --- Apply environment.yml (optional) ---
if [ -f "${ENV_YML}" ]; then
  echo "[INFO] Applying ${ENV_YML}…"
  conda env update -n "${ENV_NAME}" -f "${ENV_YML}" --prune
fi

# --- Install pip requirements (optional) ---
if [ -f "${REQS_TXT}" ]; then
  echo "[INFO] Installing pip requirements from ${REQS_TXT}…"
  python -m pip install --upgrade pip
  pip install -r "${REQS_TXT}"
fi

# --- Ensure PyTorch (pick CPU/CUDA build) ---
if ! python -c "import torch" >/dev/null 2>&1; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  else
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
  fi
  echo "[INFO] Installing torch from ${TORCH_INDEX_URL}…"
  python -m pip install "torch==2.*" --index-url "${TORCH_INDEX_URL}"
fi

python -V
python -c 'import torch,transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'

# --- Slurm sanity ---
command -v sbatch >/dev/null 2>&1 || { echo "[ERROR] sbatch not found." >&2; exit 1; }
[ -x preprocess.sh ] || { echo "[ERROR] preprocess.sh not found/executable." >&2; exit 1; }
[ -x distill_experiment.sh ] || { echo "[ERROR] distill_experiment.sh not found/executable." >&2; exit 1; }

# --- Submit jobs ---
CPU_JOB_ID=$(sbatch --parsable preprocess.sh)
[ -n "${CPU_JOB_ID}" ] || { echo "[ERROR] CPU job submission failed." >&2; exit 1; }
echo "Submitted CPU job: ${CPU_JOB_ID}"

# Run GPU job only if CPU job succeeds; change to afterany if you want it to run regardless.
GPU_JOB_ID=$(sbatch --parsable --dependency=afterok:${CPU_JOB_ID} distill_experiment.sh)
[ -n "${GPU_JOB_ID}" ] || { echo "[ERROR] GPU job submission failed." >&2; exit 1; }
echo "Submitted GPU job (afterok ${CPU_JOB_ID}): ${GPU_JOB_ID}"

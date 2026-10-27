#!/bin/bash
# PLAN_SOLVE S3b: RoboPoint-13B on S1's frames (experiments/s3b_robopoint.py) with the isolated S3b venv.  S3BARGS: e.g. "--max 4".
#SBATCH --job-name=s3brun
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then echo "[guard] no GPU on $(hostname)"; scontrol requeue $SLURM_JOB_ID; exit 97; fi
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID|RESTART_COUNT)$'); do unset $v; done
export PYTHONUNBUFFERED=1 HF_HOME=/coc/flash10/ksingara3/s3b_ext/hf_home
cd /coc/flash10/ksingara3/vla-operator-binding
echo "[s3b run] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
/coc/flash10/ksingara3/s3b_ext/venv/bin/python -I experiments/s3b_robopoint.py ${S3BARGS:-}
echo "exit=$? | $(date)"

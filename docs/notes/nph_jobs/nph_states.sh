#!/bin/bash
# NPH: SIXTH fresh start states (n2 N/K only) (no policy runs).  WHAT = n2 | f7 | f6 | assemble.  Partition / account / exclude on the sbatch command line.  Copy of w5r_states.sh.
#SBATCH --gres=gpu:a40:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nph_states_%x_%j.out
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
WHAT=${WHAT:?WHAT n2}
echo "[nph states] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} what $WHAT"
cd $ROBOT
PY="examples/libero/.venv/bin/python"
export PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero
case $WHAT in
  n2) [ -f $REPO/results/flawed/n2_branches_w6.json ] || $PY $REPO/experiments/fc_n2_w6.py || { echo failed; exit 3; } ;;
esac
echo "[nph states] done $WHAT $(date)"

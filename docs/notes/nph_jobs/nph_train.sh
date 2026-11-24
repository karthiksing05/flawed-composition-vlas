#!/bin/bash
# Single-GPU training for plan_oracle_pi0 (ORC / PZ): copy of raafp0_train.sh calling c_train_nph.py; ARGS = its arguments joined by ~.  Resumable.
# words joined by '~'.  Resumable: the same command is resubmitted until the last step exists.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nph_train_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
x=${ARGS//\~/ }
echo "[nph-train] $(date) $(hostname) restart ${SLURM_RESTART_COUNT:-0}: $x"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${used:-0}" -gt 1000 ]; then
  echo "[nph-train] a GPU holds ${used} MiB before start; refusing this node"
  exit 97
fi
srun /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/c_train_nph.py $x
rc=$?
echo "exit=$rc | $(date)"
exit $rc

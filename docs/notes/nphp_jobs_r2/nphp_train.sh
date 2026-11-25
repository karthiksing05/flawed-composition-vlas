#!/bin/bash
# Single-GPU training for plan_oracle_pi0 (ORC / PZ): NPHP copy of nph_train.sh calling c_train_nphp.py (GPU-h cap guard, skips when the final step exists); ARGS = its arguments joined by ~.  Resumable.
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
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nphp_train_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CAP=${CAP_H:-150}
spent() { sacct -u $USER -S 2026-11-24 -X -n -P --format=JobName,Elapsed,AllocTRES | awk -F'|' '$1 ~ /^nphp_/ && $3 ~ /gres\/gpu/ {n=split($2,p,/[-:]/); if(n==4) s=p[1]*86400+p[2]*3600+p[3]*60+p[4]; else s=p[1]*3600+p[2]*60+p[3]; t+=s} END{printf "%.2f", t/3600}'; }
x=${ARGS//\~/ }
nm=$(echo $x | sed -E 's/.*--name ([^ ]+).*/\1/'); ex=$(echo $x | sed -E 's/.*--exp ([^ ]+).*/\1/')
if [ -d /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/$nm/$ex/6000 ]; then echo "[nphp-train] step 6000 exists for $nm/$ex: exit 0 without work"; exit 0; fi
R=/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/nphp8
if [ "$ex" != smoke ]; then
  [ -f /coc/flash10/ksingara3/.claude_tmp/pp/nphp_checks_ok ] || { echo "[nphp-train] checks_ok marker missing (cfgcheck/identity/smoke/pcheck not passed): exit 0 without work"; exit 0; }
  ls /coc/flash10/ksingara3/flawed-composition-vlas/ledger/m3_prereg/m3_prereg__libero_goal__nphp__s0__*.json > /dev/null 2>&1 || { echo "[nphp-train] no nphp prereg row: exit 0 without work"; exit 0; }
fi
s=$(spent); echo "[nphp-train] nphp_* GPU wall-h so far $s (cap $CAP)"
awk -v s=$s -v c=$CAP 'BEGIN{exit !(s<c)}' || { echo "[nphp-train] GPU-h cap reached: exit 0 without work"; exit 0; }
echo "[nphp-train] $(date) $(hostname) restart ${SLURM_RESTART_COUNT:-0}: $x"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${used:-0}" -gt 1000 ]; then
  echo "[nphp-train] a GPU holds ${used} MiB before start; refusing this node"
  exit 97
fi
srun /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/c_train_nphp.py $x
rc=$?
echo "exit=$rc | $(date)"
exit $rc

#!/bin/bash
# NPHJ single-GPU training link (copy of nphs_train.sh calling c_train_nphj.py; cap 110 GPU-h guard over nphj_* jobs; skips when the final step exists).  ARGS = c_train arguments joined by ~.  Resumable.
# The jointly trained branch (--name pi0_g8v_raaj_l_ae) runs ONLY if results/flawed/nphj7/checks_ok.marker exists (identity + 30-step smoke + export parity); otherwise exit 0 without work.
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
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nphj_train_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CAP=${CAP_H:-110}
spent() { sacct -u $USER -S 2026-11-24 -X -n -P --format=JobName,Elapsed,AllocTRES | awk -F'|' '$1 ~ /^nphj_/ && $3 ~ /gres\/gpu/ {n=split($2,p,/[-:]/); if(n==4) s=p[1]*86400+p[2]*3600+p[3]*60+p[4]; else s=p[1]*3600+p[2]*60+p[3]; t+=s} END{printf "%.2f", t/3600}'; }
x=${ARGS//\~/ }
nm=$(echo $x | sed -E 's/.*--name ([^ ]+).*/\1/'); ex=$(echo $x | sed -E 's/.*--exp ([^ ]+).*/\1/'); st=$(echo $x | sed -E 's/.*--steps ([0-9]+).*/\1/'); fin=$((st-1))
case $nm in pi0_g8v_raaj_l_ae) [ -f /coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/nphj7/checks_ok.marker ] || { echo "[nphj-train] no checks_ok marker: exit 0 without work"; exit 0; };; esac
if [ -d /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/$nm/$ex/$fin ]; then echo "[nphj-train] step $fin exists for $nm/$ex: exit 0 without work"; exit 0; fi
s=$(spent); echo "[nphj-train] nphj_* GPU wall-h so far $s (cap $CAP)"
awk -v s=$s -v c=$CAP 'BEGIN{exit !(s<c)}' || { echo "[nphj-train] GPU-h cap reached: exit 0 without work"; exit 0; }
echo "[nphj-train] $(date) $(hostname) restart ${SLURM_RESTART_COUNT:-0}: $x"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${used:-0}" -gt 1000 ]; then
  echo "[nphj-train] a GPU holds ${used} MiB before start; refusing this node"
  exit 97
fi
srun /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/c_train_nphj.py $x
rc=$?
echo "exit=$rc | $(date)"
exit $rc

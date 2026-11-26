#!/bin/bash
# XB CPU job: run one python script (CMD, relative to the repo) in the openpi venv (or PYENV).  Header per hard rules.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HOME=/coc/flash10/ksingara3/robot/bench/hf OMP_NUM_THREADS=4
PY=${PYENV:-/coc/flash10/ksingara3/robot/openpi/.venv/bin/python}
echo "[xb_cpu] $(date) $(hostname) $PY $CMD"
$PY $CMD
echo "exit=$? | $(date)"

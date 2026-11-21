#!/bin/bash
# RAA-F probe (docs/notes/raaf_probe_runbook.md): experiments/a2_probe_f.py on one GPU.  ARGS = a2_probe_f.py's arguments, words joined by '~' (sbatch --export splits on commas).
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/a2f_probe_%x_%j.out
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then echo "[guard] no GPU on $(hostname)"; exit 97; fi
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=4
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
x=${ARGS//\~/ }
echo "[a2f-probe] $(date) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1): a2_probe_f.py $x"
/coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/a2_probe_f.py $x
echo "exit=$? | $(date)"

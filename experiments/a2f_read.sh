#!/bin/bash
# RAA-F probe (docs/notes/raaf_probe_runbook.md): CPU read.  ARGS = raaf_probe_read.py's arguments, words joined by '~'.  Writes its JSON once (refuses to overwrite).
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/a2f_read_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
x=${ARGS//\~/ }
echo "[a2f-read] $(date) $(hostname): raaf_probe_read.py $x"
/coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/raaf_probe_read.py $x
echo "exit=$? | $(date)"

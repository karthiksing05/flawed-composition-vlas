#!/bin/bash
# Run one figure script of work package (b) (CPU, matplotlib only):  sbatch experiments/fig_b.sh fig_cues.py [args]
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J figb
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/figb_%j.out
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
cd $REPO
SCRIPT=${1:?script}; shift
$PY experiments/$SCRIPT "$@"

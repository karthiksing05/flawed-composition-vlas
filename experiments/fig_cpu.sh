#!/bin/bash
# CPU composition job for the figure scripts (docs/notes/fig_plan.md): openpi venv.   sbatch --export=ALL,CMD="fig_frames.py A2" fig_cpu.sh
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/figcpu_%x_%j.out
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
cd $REPO
export MPLBACKEND=Agg MPLCONFIGDIR=/coc/flash10/ksingara3/.claude_tmp/mpl
mkdir -p $MPLCONFIGDIR
ulimit -v 12000000
/coc/flash10/ksingara3/robot/openpi/.venv/bin/python $REPO/experiments/$CMD

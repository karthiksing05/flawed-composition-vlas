#!/bin/bash
#SBATCH -J fp_fit
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/fp_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
/coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/fp_fit.py

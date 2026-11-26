#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -J xb_fix_groot15
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
R=/coc/flash10/ksingara3/robot; B=$R/bench; export UV_CACHE_DIR=$R/.uv-cache PATH=$R/simpler/uvtool/bin:$PATH; P=$B/robocasa/groot15_venv/bin/python
uv pip install --python $P setuptools wheel 2>&1 | tail -1
cd $B/tmp; $P -c "
import torch, gr00t
from gr00t.model.policy import Gr00tPolicy
print('gr00t policy import ok', torch.__version__)" 2>&1 | tail -8

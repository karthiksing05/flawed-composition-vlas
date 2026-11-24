#!/bin/bash
# NPH model check (experiments/raah0_check.py) on one GPU (tail-lab).  Copy of rvh_check.sh.  EXTRA = optional extra args (e.g. --torch-ckpt~DIR), joined by ~.
#SBATCH -p tail-lab
#SBATCH -A tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nph_check_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
x=${EXTRA:-}
x=${x//\~/ }
srun /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/raah0_check.py $x

#!/bin/bash
# experiments/fc_n2_fresh.py on one GPU (EGL): fresh familiar-layout N2 states k = 36..55 and the harness files / branches that address them.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:40:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major
#SBATCH -o /coc/flash10/ksingara3/flawed-composition-vlas/slurm/n2_fresh_%j.out
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
cd $REPO
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$ROBOT/third_party/libero MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "[n2f] $(date) $(hostname)"
$ROBOT/examples/libero/.venv/bin/python experiments/fc_n2_fresh.py
r=$?
echo "exit=$r | $(date)"
exit $r

#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -J xb_reset_rc_heldout
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
source /coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/xb_jobs/xb_lib.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
cd $B/tmp; $B/robocasa/venv_sim/bin/python $REPO/experiments/xb_roll_robocasa.py --cell-type heldout --resetonly --model none 2>&1 | grep -v -i warn | tail -5

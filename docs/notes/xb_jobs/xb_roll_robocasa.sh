#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
# XB RoboCasa365 rollouts / placeability check.  env: MODE=check | roll ; for roll: MODEL=pi0|pi05 CELLTYPE=partner|heldout CHUNK=i/n NSHARD=parallel clients [SMOKE=1]
set -uo pipefail
source /coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/xb_jobs/xb_lib.sh
xb_guard
MODE=${MODE:-roll}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENPI_DATA_HOME=$R/openpi_cache HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
nvidia-smi -L
cd $B/tmp
if [ "$MODE" = check ]; then
  $B/robocasa/venv_sim/bin/python $REPO/experiments/xb_roll_robocasa.py --check 2>&1 | grep -v -i "warning"
  xb_log_job robocasa check; exit 0
fi
MODEL=${MODEL:?}; CELLTYPE=${CELLTYPE:?}; CHUNK=${CHUNK:-0/1}; NSHARD=${NSHARD:-3}
if [ "$CELLTYPE" = heldout ]; then [ -f $XBOUT/pred_robocasa.json ] || { echo "[xb] no pred_robocasa.json: held-out refused"; exit 4; }; fi
case $MODEL in
  groot15) CK=$B/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000;;
  pi05) CFG=pi05_pretrain_human300; CK=$B/robocasa/ckpt/pi05_pretrain_human300/multitask_learning/75000;;
  pi0)  CFG=pi0_robocasa_pretrain_human300; CK=$B/robocasa/ckpt/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000;;
  *) echo "model $MODEL not wired"; exit 2;;
esac
PORT=$((20000 + SLURM_JOB_ID % 20000))
if [ "$MODEL" = groot15 ]; then
  $B/robocasa/groot15_venv/bin/python $REPO/experiments/xb_serve_groot15.py $CK $PORT > $XBOUT/serve_rc_${MODEL}_$SLURM_JOB_ID.log 2>&1 &
else
PYTHONPATH=$B/robocasa/shim:$B/robocasa/openpi-robocasa/src:$B/robocasa/openpi-robocasa/packages/openpi-client/src \
  $R/openpi/.venv/bin/python $B/robocasa/openpi-robocasa/scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=$CFG --policy.dir=$CK > $XBOUT/serve_rc_${MODEL}_$SLURM_JOB_ID.log 2>&1 &
fi
SP=$!; trap 'kill $SP 2>/dev/null' EXIT
xb_wait_port $PORT $SP 900 || { tail -30 $XBOUT/serve_rc_${MODEL}_$SLURM_JOB_ID.log; exit 3; }
echo "[xb] server up ($PORT)"
cd $B/tmp; PIDS=""
for s in $(seq 0 $((NSHARD-1))); do
  $B/robocasa/venv_sim/bin/python $REPO/experiments/xb_roll_robocasa.py --model $MODEL --cell-type $CELLTYPE --chunk $CHUNK --shard $s/$NSHARD --port $PORT ${SMOKE:+--smoke} > $XBOUT/log_rc_${MODEL}_${CELLTYPE}_$(echo $CHUNK | tr / of)_s$s.txt 2>&1 &
  PIDS="$PIDS $!"
done
FAIL=0; for p in $PIDS; do wait $p || FAIL=1; done
xb_log_job robocasa $MODEL $CELLTYPE $CHUNK fail=$FAIL
exit $FAIL

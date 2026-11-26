#!/bin/bash
# XB MESA / BiMESA rollouts.  env: BENCH=mesa|bimesa MODEL=pi0|pi05|pi0fast|groot CELLTYPE=partner|heldout CHUNK=i/n NSHARD=parallel clients [SMOKE=1]
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
set -uo pipefail
source /coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/xb_jobs/xb_lib.sh
xb_guard
BENCH=${BENCH:?}; MODEL=${MODEL:?}; CELLTYPE=${CELLTYPE:?}; CHUNK=${CHUNK:-0/1}; NSHARD=${NSHARD:-4}
if [ "$CELLTYPE" = heldout ]; then
  [ -f $XBOUT/pred_$BENCH.json ] || { echo "[xb] no prediction file pred_$BENCH.json: held-out rollouts refused"; exit 4; }
fi
case "$BENCH:$MODEL" in
  mesa:pi0) CFG=pi0_mesa; CK=mesa-pi0;; mesa:pi05) CFG=pi05_mesa; CK=mesa-pi05;; mesa:pi0fast) CFG=pi0_fast_mesa_70; CK=mesa-pi0-fast;;
  bimesa:pi0) CFG=pi0_bimesa; CK=bimesa-pi0;; bimesa:pi05) CFG=pi05_bimesa; CK=bimesa-pi05;; bimesa:pi0fast) CFG=pi0_fast_bimesa; CK=bimesa-pi0-fast;;
  *:groot) CK=$BENCH-gr00t-n1.6; CFG=groot;;
  *) echo "bad model"; exit 2;;
esac
[ "$BENCH" = mesa ] && [ "$MODEL" = groot ] && CK=mesa-gr00t-n1.6
PORT=$((20000 + SLURM_JOB_ID % 20000))
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENPI_DATA_HOME=$R/openpi_cache HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
nvidia-smi -L; free -h | head -2
if [ "$MODEL" = groot ]; then
  cd $B/mesa/mesa-GR00T && UV_PROJECT_ENVIRONMENT=$B/mesa/groot_venv $R/simpler/uvtool/bin/uv run --no-sync python gr00t/eval/serve_mesa.py --model-path $B/mesa/checkpoints/$CK --port $PORT > $XBOUT/serve_${BENCH}_${MODEL}_$SLURM_JOB_ID.log 2>&1 &
else
  PYTHONPATH=$B/mesa/openpi-mesa/src:$B/mesa/openpi-mesa/packages/openpi-client/src $R/openpi/.venv/bin/python $B/mesa/openpi-mesa/scripts/serve_policy.py --port $PORT \
     policy:checkpoint --policy.config=$CFG --policy.dir=$B/mesa/checkpoints/$CK > $XBOUT/serve_${BENCH}_${MODEL}_$SLURM_JOB_ID.log 2>&1 &
fi
SP=$!; trap 'kill $SP 2>/dev/null' EXIT
xb_wait_port $PORT $SP 900 || { tail -30 $XBOUT/serve_${BENCH}_${MODEL}_$SLURM_JOB_ID.log; exit 3; }
echo "[xb] server up (port $PORT)"
cd $REPO
PIDS=""
for s in $(seq 0 $((NSHARD-1))); do
  $B/mesa/venv/bin/python experiments/xb_roll_mesa.py --bench $BENCH --model $MODEL --cell-type $CELLTYPE --chunk $CHUNK --shard $s/$NSHARD --port $PORT ${SMOKE:+--smoke} > $XBOUT/log_${BENCH}_${MODEL}_${CELLTYPE}_$(echo $CHUNK | tr / of)_s$s.txt 2>&1 &
  PIDS="$PIDS $!"
done
FAIL=0; for p in $PIDS; do wait $p || FAIL=1; done
xb_log_job mesa $BENCH $MODEL $CELLTYPE $CHUNK fail=$FAIL
exit $FAIL

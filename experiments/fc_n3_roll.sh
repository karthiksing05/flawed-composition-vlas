#!/bin/bash
# FLAWED_COMPOSITION N3 (section 4.12): fc_n2_roll.sh on the N3 suite.  One policy server (experiments/serve_guided.py, mode ours =
# the frozen policy) and, per task id in TIDS, one run of experiments/fc_n3_client.py (libero_client.py on the N3 suite, plus the
# client-side end-of-episode position log).  A task whose result file exists is skipped (resumable).
#   MODEL  pi05 (openpi's pi05_libero, PyTorch) | groot (ar0s/groot_libero, GR00T N1.5) | xvla (lerobot/xvla-libero, SmolVLA venv)
#   PHASE  smoke | main  (results/flawed/n3_branches_<PHASE>.json: the start states and noise seeds)
#   TIDS   0+1+5  (sbatch --export splits on commas)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
OUTD=$REPO/results/flawed/n3_roll
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
case "$MODEL" in
  pi05)  SPY=$ROBOT/.venv/bin/python; unset BACKBONE_CONFIG BACKBONE_CKPT ;;
  groot) SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python; export BACKBONE_CONFIG=groot_libero BACKBONE_CKPT=ar0s/groot_libero ;;
  xvla)  SPY=/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python; export BACKBONE_CONFIG=xvla BACKBONE_CKPT=lerobot/xvla-libero ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
VIDS=0; [ "$PHASE" = smoke ] && VIDS=1
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $REPO/experiments/serve_guided.py --port $PORT --mode ours --seed 13 > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[n3] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[n3] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[n3] no free port"; exit 3; }
echo "[n3] $MODEL server up on $PORT ($(date))"
rc=0
for tid in ${TIDS//+/ }; do
  out=$OUTD/${MODEL}_${PHASE}_${tid}.json
  [ -f $out ] && { echo "[n3] $out exists"; continue; }
  PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/fc_n3_client.py \
      --port $PORT --suite n3_spatial --tasks $tid --trials 1 --out $out --fresh $REPO/results/flawed/n3_states \
      --branches $REPO/results/flawed/n3_branches_${PHASE}.json --videos $VIDS --bind track
  r=$?; echo "[n3] $MODEL $PHASE task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
done
kill $SRV 2>/dev/null
exit $rc

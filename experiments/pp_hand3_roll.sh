#!/bin/bash
# The closed-loop locator's rollouts (docs/LOCATOR.md 10): pp_hand2_roll.sh's harness with pp_hand3_client.py.
#   MODEL  pi05 | groot     SEEDS  0+1+2 (locator seeds; each writes arm LS<seed>)     TIDS  18+20 ('+'-separated)     OUTD, BR  overridden only by the smoke job
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
OUTD=${OUTD:-$REPO/results/flawed/pp_hand3}
BR=${BR:-$REPO/results/flawed/n2_branches_main.json}
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
case "$MODEL" in
  pi05)  SPY=$ROBOT/.venv/bin/python ;;
  groot) SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $REPO/experiments/pp_serve_purify.py --port $PORT --model $MODEL --mode intact --comp "" --seed 13 > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[hand3] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[hand3] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[hand3] no free port"; exit 3; }
echo "[hand3] $MODEL server up on $PORT ($(date))"
rc=0
for s in ${SEEDS//+/ }; do
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_LS${s}_${tid}.json
    [ -f $out ] && { echo "[hand3] $out exists"; continue; }
    LOC_SEED=$s PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/pp_hand3_client.py \
        --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $REPO/results/flawed/n2_states \
        --branches $BR --videos 0 --bind track
    r=$?; echo "[hand3] $MODEL LS$s task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
exit $rc

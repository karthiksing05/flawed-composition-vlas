#!/bin/bash
# PLAN_PARTS targeted F3 closed loop (docs/PARTS_RESULTS.md section 7; pp_f3t): pp_f3_roll.sh with the targeted arms of
# experiments/pp_f3t.py (COMP f3t_T | f3t_R) and the output under results/flawed/pp_f3t_roll.  Same server, client, branch file
# (results/flawed/pp_f3_branches.json: N2's main states 4-13, N2's seeds) and episode protocol as pp_f3.
#   COMP  f3t_T | f3t_R     TIDS  16+18 ('+'-separated)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
OUTD=$REPO/results/flawed/pp_f3t_roll
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
MODEL=pi05
SPY=$ROBOT/.venv/bin/python
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${COMP}_${SLURM_JOB_ID:-x}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $REPO/experiments/pp_serve_purify.py --port $PORT --model $MODEL --mode purify --comp "$COMP" --seed 13 > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[f3t] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[f3t] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[f3t] no free port"; exit 3; }
echo "[f3t] $MODEL $COMP server up on $PORT ($(date))"
rc=0
for tid in ${TIDS//+/ }; do
  out=$OUTD/${MODEL}_${COMP}_${tid}.json
  [ -f $out ] && { echo "[f3t] $out exists"; continue; }
  PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/fc_n2_client.py \
      --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $REPO/results/flawed/n2_states \
      --branches $REPO/results/flawed/pp_f3_branches.json --videos 0 --bind track > /dev/null 2>&1
  r=$?; echo "[f3t] $MODEL $COMP task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
done
kill $SRV 2>/dev/null
exit $rc

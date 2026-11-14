#!/bin/bash
# TV-F on FRESH familiar-layout states (m3_prereg tv_fresh; docs/SCAFFOLD_RESULTS.md 47).  experiments/tv_frozen.py client, unchanged,
# run with only the states and branches moved, exactly as experiments/pp_atlas_fresh_roll.sh moved them for the intact arm I:
#   --fresh results/flawed/n2_states_fresh (rows 24-43 = fresh k = 36-55)   --branches results/flawed/n2_branches_fresh.json
# Server = pp_atlas_fresh_roll.sh's for pi05 / groot (pp_serve_purify.py --mode intact --comp= --seed 13); client seed default (7).
# Output <OUTD>/<MODEL>_<arm>_<tid>.json (+ _track.npz, _tv.json).  Skips finished outputs (overcap requeue safe).
#   MODEL  pi05 | groot    TIDS  '+'-separated (default N1-N6 + K1-K4, K6)    ARMS  '+'-separated (tv, tvp)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
FRESH=$REPO/results/flawed/n2_states_fresh
OUTD=${OUTD:-$REPO/results/flawed/tv_fresh}
BR=${BR:-$REPO/results/flawed/n2_branches_fresh.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-tv}
[ -f $FRESH/states.npy ] || { echo "[tvF] no fresh states"; exit 2; }
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
case "$MODEL" in
  pi05)  SPY=$ROBOT/.venv/bin/python ;;
  groot) SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[tvF] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD br $BR"
echo "[tvF] server: $SRVARGS"
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}_r${SLURM_RESTART_COUNT:-0}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $SRVARGS --port $PORT > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[tvF] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[tvF] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[tvF] no free port"; exit 3; }
echo "[tvF] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[tvF] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 5400 examples/libero/.venv/bin/python $REPO/experiments/tv_frozen.py client \
        --arm $arm --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[tvF] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

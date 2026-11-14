#!/bin/bash
# N46-PG (m3_prereg n46; docs/SCAFFOLD_RESULTS.md 51): the place phase alone.  experiments/n46_pg.py client on the atlas's main states
# (results/flawed/n2_states, n2_branches_main.json), tv_fresh_roll.sh's server (pp_serve_purify.py --model MODEL --mode intact --comp=
# --seed 13), client seed default (7), --bind track, --max-steps 450.
# Output <OUTD>/<MODEL>_<arm>_<tid>.json (+ _track.npz, _pg.json).  Skips finished outputs (overcap requeue safe).
# Main runs (OUTD under results/) refuse to start unless the n46 prereg ledger row exists.
#   MODEL pi05 | groot   TIDS '+'-separated (default N4 22, N6 26, N5 24)   ARMS '+'-separated (vla, servo, open)
#   FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
FRESH=${FRESH:-$REPO/results/flawed/n2_states}
OUTD=${OUTD:-$REPO/results/flawed/n46_pg}
BR=${BR:-$REPO/results/flawed/n2_branches_main.json}
TIDS=${TIDS:-22+26+24}
ARMS=${ARMS:-vla}
[ -d $FRESH ] || { echo "[pg] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[pg] no branches at $BR"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__n46__s0__*.json > /dev/null 2>&1 || { echo "[pg] no n46 prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
case "$MODEL" in
  pi05)  SPY=$ROBOT/.venv/bin/python ;;
  groot) SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[pg] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[pg] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[pg] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[pg] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[pg] no free port"; exit 3; }
echo "[pg] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[pg] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $REPO/experiments/n46_pg.py client \
        --arm $arm --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos 0 --bind track --max-steps 450
    r=$?; echo "[pg] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

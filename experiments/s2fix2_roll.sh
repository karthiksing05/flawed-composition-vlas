#!/bin/bash
# S2FIX2 (m3_prereg s2_fix2; docs/SCAFFOLD_RESULTS.md 62): frozen pi0.5 on scene 2, arms I / H0 / H / G, one server per job.
# s2fix_roll.sh's server and client flags, unchanged; the client per arm:
#   I  -> experiments/s2.py client --arm I (unchanged)       H0 -> experiments/s2fix.py client --arm hI (SR 54's hand-over, unchanged)
#   H, G -> experiments/s2fix2.py client --arm H|G --layout $LAYOUT
# Output <OUTD>/pi05_<arm>_<tid>.json (+ _track.npz; H0 / H / G also _h.json).  Skips finished outputs (overcap requeue safe).
# Main runs (OUTD under results/) refuse to start unless the s2_fix2 prereg ledger row exists.
#   LAYOUT O|PD2    TIDS '+'-separated (0 K1, 1 K2, 2 N1, 3 N2)    ARMS '+'-separated (I, H0, H, G)    FRESH, BR, OUTD overridable
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
LAYOUT=${LAYOUT:-O}
FRESH=${FRESH:-$REPO/results/flawed/s2fix2/$LAYOUT}
OUTD=${OUTD:-$REPO/results/flawed/s2fix2_roll/$LAYOUT}
BR=${BR:-$REPO/results/flawed/s2fix2/branches_main.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-0+1+2+3}
ARMS=${ARMS:-G}
MODEL=pi05
[ -d $FRESH ] || { echo "[s2fix2] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[s2fix2] no branches at $BR"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_10__s2_fix2__s0__*.json > /dev/null 2>&1 || { echo "[s2fix2] no s2_fix2 prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=2
SPY=$ROBOT/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
SRVARGS="$REPO/experiments/pp_serve_purify.py --model pi05 --mode intact --comp= --seed 13"
echo "[s2fix2] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} layout $LAYOUT arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[s2fix2] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[s2fix2] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[s2fix2] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[s2fix2] no free port"; exit 3; }
echo "[s2fix2] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  case "$arm" in
    I)    CL="$REPO/experiments/s2.py client --arm I" ;;
    H0)   CL="$REPO/experiments/s2fix.py client --arm hI" ;;
    H|G)  CL="$REPO/experiments/s2fix2.py client --arm $arm --layout $LAYOUT" ;;
    *) echo "[s2fix2] unknown arm $arm"; rc=2; continue ;;
  esac
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[s2fix2] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 10800 examples/libero/.venv/bin/python $CL \
        --port $PORT --suite s2_living --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?
    # SR 54's hI client writes <out>_h.json under its own name rule (out with .json -> _h.json): same as ours
    echo "[s2fix2] $LAYOUT $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

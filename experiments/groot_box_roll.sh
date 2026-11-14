#!/bin/bash
# GROOT-BOX (m3_prereg groot_box; docs/SCAFFOLD_RESULTS.md 56): frozen GR00T N1.5, no phrases, on the fourth fresh batch k = 156-195
# (experiments/fc_n2_fresh4.py; harness trials 24-63).  Arms:
#   I     experiments/pp_atlas_client.py                    (the atlas's intact client, unchanged)
#   tv    experiments/tv_frozen.py client --arm tv          (silhouette role views, SR 46 / 49, unchanged)
#   tvb   experiments/groot_box.py client --arm tvb         (tv + held object painted as its bounding box in the target phase)
#   tvg   experiments/groot_box.py client --arm tvg         (tv + finger qpos -> GR00T's K6 cheese hold while the detector is on)
#   tvbg  experiments/groot_box.py client --arm tvbg        (both; descriptive)
# This is tvpb_roll.sh with the GR00T server / venv of tv_g_roll.sh and the clients chosen per arm.  Same server (pp_serve_purify.py
# --model groot --mode intact --comp= --seed 13), same client flags (--suite n2_goal --tasks T --trials 1 --fresh F --branches B
# --videos 0 --bind track), client seed default (7).  Output <OUTD>/groot_<arm>_<tid>.json (+ _track.npz; non-I arms _tv.json).
# Skips finished outputs (overcap requeue safe).  Main runs (OUTD under results/) refuse to start unless the groot_box prereg ledger
# row exists.
#   TIDS '+'-separated (default N1-N6 + K1-K4, K6)    ARMS '+'-separated    FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=groot
FRESH=${FRESH:-$REPO/results/flawed/n2_states_fresh4}
OUTD=${OUTD:-$REPO/results/flawed/groot_box}
BR=${BR:-$REPO/results/flawed/n2_branches_fresh4.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-tvb}
[ -d $FRESH ] || { echo "[gbox] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[gbox] no branches at $BR"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__groot_box__s0__*.json > /dev/null 2>&1 || { echo "[gbox] no groot_box prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[gbox] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[gbox] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[gbox] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[gbox] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[gbox] no free port"; exit 3; }
echo "[gbox] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  case $arm in
    I)    CL="$REPO/experiments/pp_atlas_client.py" ;;
    tv)   CL="$REPO/experiments/tv_frozen.py client --arm tv" ;;
    tvb|tvg|tvbg) CL="$REPO/experiments/groot_box.py client --arm $arm" ;;
    *) echo "[gbox] unknown arm $arm"; rc=2; continue ;;
  esac
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[gbox] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $CL \
        --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[gbox] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

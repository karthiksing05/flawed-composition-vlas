#!/bin/bash
# TV-G (m3_prereg tv_g; docs/SCAFFOLD_RESULTS.md 49): frozen GR00T N1, arms I (intact), tv (silhouette role views) and w (AC-VLA-style
# whole-wrist grey), no phrases, on the second fresh batch k = 56-115 (experiments/fc_n2_fresh2.py; harness trials 24-83).
# experiments/tv_fresh_roll.sh with FRESH / BR moved to n2_states_fresh2 / n2_branches_fresh2.json and arm I added.  Arm I runs the
# atlas's client unchanged (experiments/pp_atlas_client.py, as pp_atlas_fresh_roll.sh ran it); tv and w run experiments/tv_frozen.py
# client unchanged.  Same server (pp_serve_purify.py --model groot --mode intact --comp= --seed 13), same client flags (--suite n2_goal
# --tasks T --trials 1 --fresh F --branches B --videos 0 --bind track), client seed default (7).
# Output <OUTD>/groot_<arm>_<tid>.json (+ _track.npz; tv / w also _tv.json).  Skips finished outputs (overcap requeue safe).
# Main runs (OUTD under results/) refuse to start unless the tv_g prereg ledger row exists.
#   TIDS '+'-separated (default N1-N6 + K1-K4, K6)    ARMS '+'-separated (I, tv, w)    FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=groot
FRESH=${FRESH:-$REPO/results/flawed/n2_states_fresh2}
OUTD=${OUTD:-$REPO/results/flawed/tv_g}
BR=${BR:-$REPO/results/flawed/n2_branches_fresh2.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-tv}
[ -d $FRESH ] || { echo "[tvG] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[tvG] no branches at $BR"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__tv_g__s0__*.json > /dev/null 2>&1 || { echo "[tvG] no tv_g prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[tvG] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[tvG] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[tvG] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[tvG] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[tvG] no free port"; exit 3; }
echo "[tvG] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[tvG] $out exists"; continue; }
    if [ $arm = I ]; then
      PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $REPO/experiments/pp_atlas_client.py \
          --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
          --branches $BR --videos $VIDEOS --bind track
    else
      PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $REPO/experiments/tv_frozen.py client \
          --arm $arm --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
          --branches $BR --videos $VIDEOS --bind track
    fi
    r=$?; echo "[tvG] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

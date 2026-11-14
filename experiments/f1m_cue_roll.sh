#!/bin/bash
# F1m-CUE (m3_prereg f1m_cue; docs/SCAFFOLD_RESULTS.md 53): arms sb (held object as its bounding box), sf (fixed-size box) and g (held
# finger qpos replaced by the cheese-held value) on top of tvp, frozen pi0.5.  experiments/tv_attr_roll.sh with only the client script
# (experiments/f1m_cue.py client) and the state set moved: SET=fresh -> n2_states_fresh + n2_branches_fresh.json (tv_fresh's tvp);
# SET=main -> n2_states + n2_branches_main.json (tv_frozen's tvp).  Same server (pp_serve_purify.py --mode intact --comp= --seed 13),
# same client flags (--suite n2_goal --tasks T --trials 1 --fresh F --branches B --videos 0 --bind track), client seed default (7).
# Output <OUTD>/pi05_<arm>_<tid>.json (+ _track.npz, _tv.json).  Skips finished outputs (requeue safe).
# Main runs (OUTD under results/) refuse to start unless the f1m_cue prereg ledger row exists.
#   SET fresh|main    TIDS '+'-separated    ARMS '+'-separated (sb, sf, g)    FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=pi05
SET=${SET:-fresh}
case "$SET" in
  fresh) FRESH=${FRESH:-$REPO/results/flawed/n2_states_fresh}; BR=${BR:-$REPO/results/flawed/n2_branches_fresh.json} ;;
  main)  FRESH=${FRESH:-$REPO/results/flawed/n2_states};       BR=${BR:-$REPO/results/flawed/n2_branches_main.json} ;;
  *) echo "[f1m] unknown SET=$SET"; exit 2 ;;
esac
OUTD=${OUTD:-$REPO/results/flawed/f1m_cue/$SET}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-22+26}
ARMS=${ARMS:-sb}
[ -d $FRESH ] || { echo "[f1m] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[f1m] no branches at $BR"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__f1m_cue__s0__*.json > /dev/null 2>&1 || { echo "[f1m] no f1m_cue prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
SPY=$ROBOT/.venv/bin/python
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[f1m] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} set $SET arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[f1m] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[f1m] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[f1m] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[f1m] no free port"; exit 3; }
echo "[f1m] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[f1m] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 5400 examples/libero/.venv/bin/python $REPO/experiments/f1m_cue.py client \
        --arm $arm --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[f1m] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

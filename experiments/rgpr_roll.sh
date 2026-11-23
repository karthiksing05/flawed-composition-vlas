#!/bin/bash
# RGPR (m3_prereg rgpr; docs/notes/rgpr.md) = rg_comp_roll.sh with experiments/rgpr.py's client, arm rgpr only (tasks 0-3), OUTD results/flawed/rgpr_roll.
# Original header follows.  RG_COMP (m3_prereg rg_comp; docs/SCAFFOLD_RESULTS.md 70): rollouts of experiments/rg_comp.py's client on the rgc suite
# (results/flawed/rg_comp).  One client process per (arm, task).  The policy is served as SR 45 served it:
#   pi05  pp_serve_purify.py --model pi05 --mode intact --comp '' --seed 13          (fx / fx_r / f6o2)
#   pi0   serve_guided.py --mode ours --seed 13, BACKBONE_CONFIG=pi0_libero          (fx_more, SR 45.3; s1_cross, SR 65)
# Client flags: --suite rgc --tasks T --trials 1 --fresh results/flawed/rg_comp/states --branches B --videos 0 --bind track, client
# seed default (7).  B: tasks 0-7 (F7 states) use BR7, tasks 8-10 (F6 states) BR6.  Output <OUTD>/<MODEL>_<arm>_<tid>.json
# (+ _track.npz, _rc.json).  Skips finished outputs (overcap requeue safe).  Main runs (OUTD under results/) refuse to start unless
# the rg_comp prereg row exists.
#   MODEL pi05|pi0   ARMS '+'-separated   TIDS '+'-separated   PHASE smoke|main (default branch files)   BR7 / BR6 / OUTD overridable
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=${MODEL:?MODEL pi05|pi0}
PHASE=${PHASE:-main}
ST=$REPO/results/flawed/rg_comp/states
BR7=${BR7:-$REPO/results/flawed/rg_comp/br7_${PHASE}.json}
BR6=${BR6:-$REPO/results/flawed/rg_comp/br6_${PHASE}.json}
OUTD=${OUTD:-$REPO/results/flawed/rgpr_roll}
VIDEOS=${VIDEOS:-0}
ARMS=${ARMS:?ARMS}
TIDS=${TIDS:?TIDS}
[ -d $ST ] || { echo "[rgc] no states at $ST"; exit 2; }
[ -f $BR7 ] && [ -f $BR6 ] || { echo "[rgc] no branch files $BR7 / $BR6"; exit 2; }
[ -z "${RG_STATS:-}" ] || { echo "[rgc] RG_STATS must be unset (ReGuide runs on its hashed results/flawed/rg_stats.json)"; exit 2; }
[ -z "${LEAF:-}" ] || { echo "[rgc] LEAF must be unset"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__rgpr__s0__*.json > /dev/null 2>&1 || { echo "[rgc] no rgpr prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=2
case "$MODEL" in
  pi05) SPY=$ROBOT/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
        SRVARGS="$REPO/experiments/pp_serve_purify.py --model pi05 --mode intact --comp '' --seed 13" ;;
  pi0)  SPY=$ROBOT/.venv/bin/python; export BACKBONE_CONFIG=pi0_libero; unset BACKBONE_CKPT
        SRVARGS="$REPO/experiments/serve_guided.py --mode ours --seed 13" ;;
  *) echo "[rgc] unknown MODEL=$MODEL"; exit 2 ;;
esac
echo "[rgc] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL phase $PHASE arms $ARMS tids $TIDS outd $OUTD"
echo "[rgc] server: BACKBONE_CONFIG=${BACKBONE_CONFIG:-} $SRVARGS"
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}_r${SLURM_RESTART_COUNT:-0}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  eval "$SPY $SRVARGS --port $PORT" > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[rgc] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[rgc] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[rgc] no free port"; exit 3; }
echo "[rgc] $MODEL server up on $PORT ($(date))"
valid() {   # the registered (arm, task) cells (rg_comp.TASKS)
  case $1 in
    rgpr) case $2 in 0|1|2|3) return 0 ;; esac ;;     # RGPR: ReGuide + the F7R resolver, the four F7 tasks only
  esac
  return 1
}
rc=0
for arm in ${ARMS//+/ }; do
  for tid in ${TIDS//+/ }; do
    valid $arm $tid || continue
    if [ $tid -le 7 ]; then BR=$BR7; else BR=$BR6; fi
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[rgc] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $REPO/experiments/rgpr.py client \
        --arm $arm --port $PORT --suite rgc --tasks $tid --trials 1 --out $out --fresh $ST --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[rgc] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

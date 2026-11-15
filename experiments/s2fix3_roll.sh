#!/bin/bash
# S2FIX3 (m3_prereg s2_fix3; docs/SCAFFOLD_RESULTS.md 63): scene 2, layout O, NEW states; one VLA server per job (SR 50's server lines,
# s2_roll.sh), then per ARM x TID one client:
#   I      -> experiments/s2.py client --arm I (unchanged)
#   H      -> experiments/s2fix2.py client --arm H --layout O (SR 62's arm, unchanged)
#   Hp     -> experiments/s2fix3.py client --arm Hp   (needs the OWLv2 sidecar experiments/s2fix3_det.py, started here when ARMS has Hp)
# Client flags as SR 50-62: --trials 1 --fresh --branches --videos 0 --bind track, client seed default (7).
# Output <OUTD>/<MODEL>_<arm>_<tid>.json (+ _track.npz; H / Hp also _h.json).  Skips finished outputs (overcap requeue safe).
# Main runs (OUTD under results/) refuse to start unless the s2_fix3 prereg ledger row exists.
#   MODEL pi05|pi0|groot    TIDS '+'-separated (0 K1, 1 K2, 2 N1, 3 N2)    ARMS '+'-separated    FRESH, BR, OUTD overridable
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
FRESH=${FRESH:-$REPO/results/flawed/s2fix3/O}
OUTD=${OUTD:-$REPO/results/flawed/s2fix3_roll/O}
BR=${BR:-$REPO/results/flawed/s2fix3/branches_main.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-0+1+2+3}
ARMS=${ARMS:-I}
[ -d $FRESH ] || { echo "[s2fix3] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[s2fix3] no branches at $BR"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_10__s2_fix3__s0__*.json > /dev/null 2>&1 || { echo "[s2fix3] no s2_fix3 prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=2
case "$MODEL" in
  pi05)  SPY=$ROBOT/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
         SRVARGS="$REPO/experiments/pp_serve_purify.py --model pi05 --mode intact --comp= --seed 13" ;;
  groot) SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
         SRVARGS="$REPO/experiments/pp_serve_purify.py --model groot --mode intact --comp= --seed 13" ;;
  pi0)   SPY=$ROBOT/.venv/bin/python; export BACKBONE_CONFIG=pi0_libero; unset BACKBONE_CKPT
         SRVARGS="$REPO/experiments/serve_guided.py --mode ours --seed 13" ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
echo "[s2fix3] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[s2fix3] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[s2fix3] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[s2fix3] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[s2fix3] no free port"; exit 3; }
echo "[s2fix3] $MODEL server up on $PORT ($(date))"
DET=""
case "+$ARMS+" in
  *+Hp+*)
    DPORT=$((9000 + RANDOM % 900)); for k in $(seq 1 50); do [ $DPORT != $PORT ] && free_port $DPORT && break; DPORT=$((9000 + RANDOM % 900)); done
    DLOG=$OUTD/det_${MODEL}_${SLURM_JOB_ID:-x}_r${SLURM_RESTART_COUNT:-0}.log
    ( unset PYTHONPATH; cd $REPO; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home TRANSFORMERS_OFFLINE=1
      exec $ROBOT/.venv/bin/python $REPO/experiments/s2fix3_det.py --port $DPORT ) > $DLOG 2>&1 &
    DET=$!
    dup=0
    for i in $(seq 1 120); do sleep 3; grep -q "detector listening" $DLOG 2>/dev/null && { dup=1; break; }; kill -0 $DET 2>/dev/null || break; done
    [ $dup = 1 ] || { echo "[s2fix3] detector died"; tail -30 $DLOG; kill $SRV 2>/dev/null; exit 3; }
    echo "[s2fix3] detector up on $DPORT: $(grep 'detector listening' $DLOG)"
    export S2FIX3_DET_PORT=$DPORT ;;
esac
rc=0
for arm in ${ARMS//+/ }; do
  case "$arm" in
    I)        CL="$REPO/experiments/s2.py client --arm I" ;;
    H)        CL="$REPO/experiments/s2fix2.py client --arm H --layout O" ;;
    Hp)       CL="$REPO/experiments/s2fix3.py client --arm Hp" ;;
    *) echo "[s2fix3] unknown arm $arm"; rc=2; continue ;;
  esac
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[s2fix3] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 10800 examples/libero/.venv/bin/python $CL \
        --port $PORT --suite s2_living --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?
    echo "[s2fix3] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
[ -n "$DET" ] && { kill $DET 2>/dev/null; wait $DET 2>/dev/null; }
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

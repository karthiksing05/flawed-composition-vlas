#!/bin/bash
# ReGuide on frozen pi0.5 (m3_prereg reguide; docs/SCAFFOLD_RESULTS.md 52): arms rgo (RG-oracle) and rgp (RG-perc).
# experiments/tv_attr_roll.sh with only the client script moved (experiments/rg.py client) and OUTD -> results/flawed/rg.  Same server
# (pp_serve_purify.py --mode intact --comp= --seed 13), same client flags (--suite n2_goal --tasks T --trials 1 --fresh n2_states_fresh
# --branches n2_branches_fresh.json --videos 0 --bind track), client seed default (7).  Output <OUTD>/<MODEL>_<arm>_<tid>.json
# (+ _track.npz, _rg.json).  Skips finished outputs.  Writing into results/ requires the m3_prereg reguide ledger row.
#   MODEL  pi05    TIDS  '+'-separated (default N1-N6 + K1-K4, K6)    ARMS  '+'-separated (rgo, rgp)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
FRESH=${FRESH:-$REPO/results/flawed/n2_states_fresh}
OUTD=${OUTD:-$REPO/results/flawed/rg}
BR=${BR:-$REPO/results/flawed/n2_branches_fresh.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-rgo}
[ -d $FRESH ] || { echo "[rg] no states at $FRESH"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__reguide__s0__*.json >/dev/null 2>&1 || { echo "[rg] no m3_prereg reguide row: refusing to write $OUTD"; exit 4; }
    [ -f $REPO/results/flawed/rg_stats.json ] || { echo "[rg] no rg_stats.json"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
case "$MODEL" in
  pi05)  SPY=$ROBOT/.venv/bin/python ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[rg] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR stats ${RG_STATS:-default}"
echo "[rg] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[rg] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[rg] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[rg] no free port"; exit 3; }
echo "[rg] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[rg] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 5400 examples/libero/.venv/bin/python $REPO/experiments/rg.py client \
        --arm $arm --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[rg] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

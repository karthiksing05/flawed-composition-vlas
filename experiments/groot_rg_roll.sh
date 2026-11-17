#!/bin/bash
# GROOT-RG (m3_prereg groot_rg; docs/SCAFFOLD_RESULTS.md 71): ReGuide transport on frozen GR00T N1.5, scene 1, on the fresh batch
# k = 500-539 (experiments/fc_n2_gx.py; harness trials 24-63).  Arms (every client existing and unchanged):
#   I     experiments/pp_atlas_client.py                    (the atlas's intact client)
#   rgp   experiments/rg.py client --arm rgp                (ReGuide RG-perc, SR 52 / 55 / 65; statistics results/flawed/rg_stats.json)
#   rgo   experiments/rg.py client --arm rgo                (ReGuide RG-oracle, simulator poses, PRIVILEGED reference)
# This is groot_box_roll.sh (GR00T server / venv) with s1_cross_roll.sh's ReGuide arms.  Same server (pp_serve_purify.py --model groot
# --mode intact --comp= --seed 13; GR00T returns LIBERO delta eef commands to every client), same client flags (--suite n2_goal --tasks T
# --trials 1 --fresh F --branches B --videos 0 --bind track), client seed default (7).  Output <OUTD>/groot_<arm>_<tid>.json
# (+ _track.npz; rg arms _rg.json).  Skips finished outputs (overcap requeue safe).  Main runs (OUTD under results/) refuse to start
# unless the groot_rg prereg ledger row exists.
#   TIDS '+'-separated (default N1-N6 + K1-K4, K6)    ARMS '+'-separated    FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=groot
FRESH=${FRESH:-$REPO/results/flawed/n2_states_gx}
OUTD=${OUTD:-$REPO/results/flawed/groot_rg}
BR=${BR:-$REPO/results/flawed/n2_branches_gx.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-rgp}
[ -d $FRESH ] || { echo "[grg] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[grg] no branches at $BR"; exit 2; }
[ -f $REPO/results/flawed/rg_stats.json ] || { echo "[grg] no rg_stats.json"; exit 2; }
[ -z "${RG_STATS:-}" ] || { echo "[grg] RG_STATS must be unset"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__groot_rg__s0__*.json > /dev/null 2>&1 || { echo "[grg] no groot_rg prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[grg] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[grg] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[grg] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[grg] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[grg] no free port"; exit 3; }
echo "[grg] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  case $arm in
    I)    CL="$REPO/experiments/pp_atlas_client.py" ;;
    rgp|rgo) CL="$REPO/experiments/rg.py client --arm $arm" ;;
    *) echo "[grg] unknown arm $arm"; rc=2; continue ;;
  esac
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[grg] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $CL \
        --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[grg] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

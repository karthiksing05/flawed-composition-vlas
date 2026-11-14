#!/bin/bash
# TVPB-RG (m3_prereg tvpb_rg; docs/SCAFFOLD_RESULTS.md 55): frozen pi0.5 head-to-head on the third fresh batch k = 116-155
# (experiments/fc_n2_fresh3.py; harness trials 24-63).  Every arm runs an existing client UNCHANGED:
#   I     experiments/pp_atlas_client.py                    (the atlas's intact client, as pp_atlas_fresh_roll.sh / tv_g_roll.sh)
#   tvp   experiments/tv_frozen.py client --arm tvp         (role views + role phrases, SR 46-47)
#   tvpb  experiments/f1m_cue.py client --arm sb            (tvp + held object painted as its bounding box in the target phase, SR 53)
#   rgp   experiments/rg.py client --arm rgp                (ReGuide RG-perc, SR 52; statistics results/flawed/rg_stats.json)
#   rgo   experiments/rg.py client --arm rgo                (ReGuide RG-oracle, PRIVILEGED, SR 52)
# This is rg_roll.sh / tv_g_roll.sh with the client chosen per arm and FRESH / BR moved.  Same server (pp_serve_purify.py --model pi05
# --mode intact --comp= --seed 13), same client flags (--suite n2_goal --tasks T --trials 1 --fresh F --branches B --videos 0 --bind
# track), client seed default (7).  Output <OUTD>/pi05_<arm>_<tid>.json (+ _track.npz; tvp / tvpb _tv.json; rgp / rgo _rg.json).
# Skips finished outputs (overcap requeue safe).  Main runs (OUTD under results/) refuse to start unless the tvpb_rg prereg ledger
# row exists.
#   TIDS '+'-separated (default N1-N6 + K1-K4, K6)    ARMS '+'-separated (I, tvp, tvpb, rgp, rgo)    FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=pi05
FRESH=${FRESH:-$REPO/results/flawed/n2_states_fresh3}
OUTD=${OUTD:-$REPO/results/flawed/tvpb_rg}
BR=${BR:-$REPO/results/flawed/n2_branches_fresh3.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-tvpb}
[ -d $FRESH ] || { echo "[tvpb] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[tvpb] no branches at $BR"; exit 2; }
[ -f $REPO/results/flawed/rg_stats.json ] || { echo "[tvpb] no rg_stats.json"; exit 2; }
[ -z "${RG_STATS:-}" ] || { echo "[tvpb] RG_STATS must be unset (ReGuide runs on its hashed results/flawed/rg_stats.json)"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__tvpb_rg__s0__*.json > /dev/null 2>&1 || { echo "[tvpb] no tvpb_rg prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home OMP_NUM_THREADS=2
SPY=$ROBOT/.venv/bin/python
SRVARGS="$REPO/experiments/pp_serve_purify.py --model $MODEL --mode intact --comp= --seed 13"
echo "[tvpb] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[tvpb] server: $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[tvpb] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[tvpb] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[tvpb] no free port"; exit 3; }
echo "[tvpb] $MODEL server up on $PORT ($(date))"
rc=0
for arm in ${ARMS//+/ }; do
  case $arm in
    I)    CL="$REPO/experiments/pp_atlas_client.py" ;;
    tvp)  CL="$REPO/experiments/tv_frozen.py client --arm tvp" ;;
    tvpb) CL="$REPO/experiments/f1m_cue.py client --arm sb" ;;
    rgp)  CL="$REPO/experiments/rg.py client --arm rgp" ;;
    rgo)  CL="$REPO/experiments/rg.py client --arm rgo" ;;
    *) echo "[tvpb] unknown arm $arm"; rc=2; continue ;;
  esac
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[tvpb] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $CL \
        --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[tvpb] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

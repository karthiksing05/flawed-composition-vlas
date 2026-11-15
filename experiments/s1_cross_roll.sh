#!/bin/bash
# S1-CROSS (m3_prereg s1_cross; docs/SCAFFOLD_RESULTS.md 65): does the scene-1 front end carry to a third model?  Frozen pi0 (openpi
# pi0_libero, PyTorch) or frozen SmolVLA (lerobot/smolvla_libero) on the NEW fresh batch k = 236-275 (experiments/fc_n2_s1x.py;
# harness trials 24-63).  Every arm runs an existing client UNCHANGED:
#   I     experiments/pp_atlas_client.py                    (the atlas's intact client)
#   tv    experiments/tv_frozen.py client --arm tv          (silhouette role views, SR 46 / 49)
#   tvpb  experiments/f1m_cue.py client --arm sb            (tvp + held object painted as its bounding box in the target phase, SR 53 / 55)
#   rgp   experiments/rg.py client --arm rgp                (ReGuide RG-perc, SR 52 / 55; statistics results/flawed/rg_stats.json)
# The policy is served exactly as the atlas served these models (experiments/pp_atlas_more_roll.sh: serve_guided.py --mode ours
# --seed 13; pi0's output transform adds the state back (AbsoluteActions), so every client receives LIBERO's delta eef commands, as
# from pi0.5).  Client flags as tvpb_roll.sh (--suite n2_goal --tasks T --trials 1 --fresh F --branches B --videos 0 --bind track),
# client seed default (7).  Output <OUTD>/<MODEL>_<arm>_<tid>.json (+ _track.npz; tv / tvpb _tv.json; rgp _rg.json).
# Skips finished outputs (overcap requeue safe).  Main runs (OUTD under results/) refuse to start unless the s1_cross prereg row exists.
#   MODEL pi0|smolvla    TIDS '+'-separated (default N1-N6 + K1-K4, K6)    ARMS '+'-separated    FRESH, BR, OUTD overridable (smoke)
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
MODEL=${MODEL:?MODEL pi0|smolvla}
FRESH=${FRESH:-$REPO/results/flawed/n2_states_s1x}
OUTD=${OUTD:-$REPO/results/flawed/s1_cross}
BR=${BR:-$REPO/results/flawed/n2_branches_s1x.json}
VIDEOS=${VIDEOS:-0}
TIDS=${TIDS:-16+18+20+22+24+26+0+2+4+6+10}
ARMS=${ARMS:-tvpb}
[ -d $FRESH ] || { echo "[s1x] no states at $FRESH"; exit 2; }
[ -f $BR ] || { echo "[s1x] no branches at $BR"; exit 2; }
[ -f $REPO/results/flawed/rg_stats.json ] || { echo "[s1x] no rg_stats.json"; exit 2; }
[ -z "${RG_STATS:-}" ] || { echo "[s1x] RG_STATS must be unset (ReGuide runs on its hashed results/flawed/rg_stats.json)"; exit 2; }
[ -z "${LEAF:-}" ] || { echo "[s1x] LEAF must be unset (serve_guided.py --mode ours, no leaf)"; exit 2; }
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__s1_cross__s0__*.json > /dev/null 2>&1 || { echo "[s1x] no s1_cross prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=2
[ -n "${HFH:-}" ] && export HF_HOME=$HFH
case "$MODEL" in
  pi0)     SPY=$ROBOT/.venv/bin/python; export BACKBONE_CONFIG=pi0_libero; unset BACKBONE_CKPT ;;
  smolvla) SPY=/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python; export BACKBONE_CONFIG=smolvla_libero BACKBONE_CKPT=lerobot/smolvla_libero ;;
  *) echo "[s1x] unknown MODEL=$MODEL"; exit 2 ;;
esac
SRVARGS="$REPO/experiments/serve_guided.py --mode ours --seed 13"
echo "[s1x] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} model $MODEL arms $ARMS tids $TIDS outd $OUTD fresh $FRESH br $BR"
echo "[s1x] server: BACKBONE_CONFIG=$BACKBONE_CONFIG BACKBONE_CKPT=${BACKBONE_CKPT:-} $SRVARGS"
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
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[s1x] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[s1x] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[s1x] no free port"; exit 3; }
echo "[s1x] $MODEL server up on $PORT ($(date))"; grep -m3 -E "Backbone|describe|n_params|absolute" $LOG
rc=0
for arm in ${ARMS//+/ }; do
  case $arm in
    I)    CL="$REPO/experiments/pp_atlas_client.py" ;;
    tv)   CL="$REPO/experiments/tv_frozen.py client --arm tv" ;;
    tvpb) CL="$REPO/experiments/f1m_cue.py client --arm sb" ;;
    rgp)  CL="$REPO/experiments/rg.py client --arm rgp" ;;
    *) echo "[s1x] unknown arm $arm"; rc=2; continue ;;
  esac
  for tid in ${TIDS//+/ }; do
    out=$OUTD/${MODEL}_${arm}_${tid}.json
    [ -f $out ] && { echo "[s1x] $out exists"; continue; }
    PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 7200 examples/libero/.venv/bin/python $CL \
        --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
        --branches $BR --videos $VIDEOS --bind track
    r=$?; echo "[s1x] $MODEL $arm task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
  done
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

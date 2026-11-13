#!/bin/bash
# The break atlas on FRESH familiar-layout states (m3_prereg pp_atlas_fresh; docs/BREAK_ATLAS.md, docs/ARCH_FAILURES.md).  The five
# models' harnesses, unchanged, with only the states and branches moved:
#   pi05 | groot          served as in experiments/pp_atlas_roll.sh       (pp_serve_purify.py --mode intact --comp "" --seed 13)
#   pi0 | xvla | smolvla  served as in experiments/pp_atlas_more_roll.sh  (serve_guided.py --mode ours --seed 13)
# and the unchanged client experiments/pp_atlas_client.py with --fresh results/flawed/n2_states_fresh (rows 0-23 = the stored states,
# 24-43 = fresh k = 36-55; experiments/fc_n2_fresh.py) and --branches results/flawed/n2_branches_fresh.json (trials 24-43,
# seed_base 13000000 + 1000 k).
#   MODEL  pi05 | groot | pi0 | xvla | smolvla      TIDS  0+2+4 ('+'-separated; sbatch --export splits on commas)
#   OUTD, BR  default results/flawed/pp_atlas_fresh and the fresh main branches; the smoke overrides both
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
FRESH=$REPO/results/flawed/n2_states_fresh
OUTD=${OUTD:-$REPO/results/flawed/pp_atlas_fresh}
BR=${BR:-$REPO/results/flawed/n2_branches_fresh.json}
[ -f $FRESH/states.npy ] || { echo "[atlasF] no fresh states"; exit 2; }
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
case "$MODEL" in
  pi05)    SPY=$ROBOT/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
           SRVARGS="$REPO/experiments/pp_serve_purify.py --model pi05 --mode intact --comp= --seed 13" ;;
  groot)   SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
           SRVARGS="$REPO/experiments/pp_serve_purify.py --model groot --mode intact --comp= --seed 13" ;;
  pi0)     SPY=$ROBOT/.venv/bin/python; export BACKBONE_CONFIG=pi0_libero; unset BACKBONE_CKPT
           SRVARGS="$REPO/experiments/serve_guided.py --mode ours --seed 13" ;;
  xvla)    SPY=/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python; export BACKBONE_CONFIG=xvla BACKBONE_CKPT=lerobot/xvla-libero
           SRVARGS="$REPO/experiments/serve_guided.py --mode ours --seed 13" ;;
  smolvla) SPY=/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python; export BACKBONE_CONFIG=smolvla_libero BACKBONE_CKPT=lerobot/smolvla_libero
           SRVARGS="$REPO/experiments/serve_guided.py --mode ours --seed 13" ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $SRVARGS --port $PORT > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[atlasF] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[atlasF] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[atlasF] no free port"; exit 3; }
echo "[atlasF] $MODEL server up on $PORT ($(date))"; grep -m3 -E "Backbone|describe|n_params" $LOG
rc=0
for tid in ${TIDS//+/ }; do
  [ $((tid % 2)) = 0 ] || { echo "[atlasF] tid $tid is not familiar-layout"; rc=2; continue; }
  out=$OUTD/${MODEL}_${tid}.json
  [ -f $out ] && { echo "[atlasF] $out exists"; continue; }
  PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/pp_atlas_client.py \
      --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $FRESH \
      --branches $BR --videos ${VIDS:-0} --bind track
  r=$?; echo "[atlasF] $MODEL task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

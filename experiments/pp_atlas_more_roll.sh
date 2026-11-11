#!/bin/bash
# The break atlas on more VLAs (docs/BREAK_ATLAS.md; PLAN_ARCH_FAILURES 4.2): pp_atlas_roll.sh's harness, unchanged, with the frozen
# policy served by experiments/serve_guided.py --mode ours (GuidedPolicy, no hooks: N2's server; pp_serve_purify.py --mode intact is
# the same policy for pi05 / groot) and the unchanged atlas client experiments/pp_atlas_client.py (N2's client + the simulator track).
# Branch seeds (seed_base + query) set the noise, so --seed 13 matches the atlas.
#   MODEL  pi0 (openpi pi0_libero, PyTorch; output transform adds the state back: AbsoluteActions)
#          | xvla (lerobot/xvla-libero, absolute eef control, SmolVLA venv) | smolvla (lerobot/smolvla_libero, SmolVLA venv)
#   TIDS   16+18 ('+'-separated; sbatch --export splits on commas)
#   OUTD, BR  default: results/flawed/pp_atlas (beside pi05_*, groot_*) and N2's main branches; the smoke job overrides both
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
OUTD=${OUTD:-$REPO/results/flawed/pp_atlas}
BR=${BR:-$REPO/results/flawed/n2_branches_main.json}
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
[ -n "${HFH:-}" ] && export HF_HOME=$HFH
case "$MODEL" in
  pi0)     SPY=$ROBOT/.venv/bin/python; export BACKBONE_CONFIG=pi0_libero; unset BACKBONE_CKPT ;;
  xvla)    SPY=/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python; export BACKBONE_CONFIG=xvla BACKBONE_CKPT=lerobot/xvla-libero ;;
  smolvla) SPY=/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python; export BACKBONE_CONFIG=smolvla_libero BACKBONE_CKPT=lerobot/smolvla_libero ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=${LOG:-$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}.log}
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $REPO/experiments/serve_guided.py --port $PORT --mode ours --seed 13 > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[atlas+] port $PORT taken (attempt $attempt)"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[atlas+] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[atlas+] no free port"; exit 3; }
echo "[atlas+] $MODEL server up on $PORT ($(date))"; grep -m3 -E "Backbone|describe|n_params" $LOG
rc=0
for tid in ${TIDS//+/ }; do
  out=$OUTD/${MODEL}_${tid}.json
  [ -f $out ] && { echo "[atlas+] $out exists"; continue; }
  PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero examples/libero/.venv/bin/python $REPO/experiments/pp_atlas_client.py \
      --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $out --fresh $REPO/results/flawed/n2_states \
      --branches $BR --videos ${VIDS:-0} --bind track
  r=$?; echo "[atlas+] $MODEL task $tid client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit $rc

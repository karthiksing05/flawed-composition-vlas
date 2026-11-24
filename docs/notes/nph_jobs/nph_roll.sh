#!/bin/bash
# NPH copy of rvw5_roll.sh (plan_necessity_power 3.2 step 8): Phase A rollouts on the 6th fresh set (n2_states_w6, n2_branches_w6.json).  RAAPY = raa.py (pi0-A0, pi0-sL) | raa_wh0.py (pi0-H*, WHERE_LOG=1).  Refuses without the nph prereg row.
#   ARM = rrel | raa.  CKPT = PyTorch policy dir.  CONFIG = served config (default pi05_g8v_<arm>_ae; the smoke serves R0 / R2's own config).
#   NAME, OUTD = result name and directory.  FRESH = states dir (n2_states = dev states 4-23 with n2_branches_main.json; n2_states_rx = the
#   40 new states with n2_branches_rx.json).  BR = branches file.  TIDS joined by '+'.  A result file that exists is skipped.
#   Per episode query the server appends the channel error vs simulator truth to $OUTD/chan_${NAME}_<tid>.jsonl (never fed to the model).
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nph_roll_%x_%j.out
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
WHERE_OWN=${WHERE_OWN:-}
case "$OUTD" in
  $REPO/results/*)
    ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__nph__s0__*.json > /dev/null 2>&1 || { echo "[raaw] no nph prereg row: refusing"; exit 4; } ;;
esac
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
SEGCKPT=${SEGCKPT:-/coc/flash10/ksingara3/robot/data/fa1/seg/ckpt/unet_main.npz}
FRESH=${FRESH:-$REPO/results/flawed/n2_states_w6}
BR=${BR:-$REPO/results/flawed/n2_branches_w6.json}
echo "[raaroll] $(date) $(hostname) job ${SLURM_JOB_ID:-x} arm $ARM name $NAME ckpt $CKPT config ${CONFIG:-default} fresh $FRESH br $BR tids $TIDS"
PORT=$((9000 + RANDOM % 900))
LOG=$OUTD/server_${NAME}_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}.log
SPY=$ROBOT/.venv/bin/python
cd $ROBOT
start_server() {
  $SPY $REPO/experiments/${RAAPY:-raa_w.py} serve --arm $ARM --port $PORT --ckpt $CKPT --seed 13 --seg-ckpt $SEGCKPT ${WHERE_OWN:+--where_own $WHERE_OWN} ${WHERE_LOG:+--where_log $OUTD/where_${NAME}_${SLURM_JOB_ID}.jsonl} \
      --chan-diag $OUTD/chan_${NAME}_${SLURM_JOB_ID}.jsonl --diag $OUTD/segdiag_${NAME}_${SLURM_JOB_ID}.json ${CONFIG:+--config $CONFIG} > $LOG 2>&1 &
  SRV=$!
}
up=0
for attempt in 1 2 3 4; do
  start_server
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  kill $SRV 2>/dev/null
  PORT=$((9000 + RANDOM % 900))
done
[ $up = 1 ] || { echo "[raaroll] no server"; tail -30 $LOG; exit 3; }
rc=0
for tid in ${TIDS//+/ }; do
  out=$OUTD/${NAME}_${PHASE:-main}_${tid}.json
  [ -f $out ] && { echo "[raaroll] $out exists"; continue; }
  tmp=$OUTD/${NAME}_${PHASE:-main}_${tid}.tmp.json
  PYTHONPATH=$ROBOT/third_party/libero timeout 5400 examples/libero/.venv/bin/python $REPO/experiments/${RAAPY:-raa_w.py} client --arm $ARM \
      --port $PORT --suite n2_goal --tasks $tid --trials 1 --out $tmp --fresh $FRESH \
      --branches $BR --videos 0 --bind track
  r=$?; echo "[raaroll] $ARM task $tid client exit=$r ($(date))"
  if [ $r = 0 ]; then mv $tmp $out; else rc=$r; fi
done
kill $SRV 2>/dev/null
echo "exit=$rc | $(date)"
exit $rc

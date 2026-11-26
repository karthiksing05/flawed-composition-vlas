#!/bin/bash
#SBATCH -J fp_probe_groot
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/fp_%x_%j.out
# Language-sensitivity probe (experiments/fp_probe_client.py): the atlas's frozen server for groot, one forward-pass probe per stored state. No rollouts.
set -uo pipefail
MODEL=groot
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
OUTD=$REPO/results/flawed/fp_dev/probe_raw
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=2
SPY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home; SRVARGS="$REPO/experiments/pp_serve_purify.py --model groot --mode intact --comp= --seed 13"
PORT=$((9000 + RANDOM % 900))
LOG=$OUTD/server_${MODEL}_${SLURM_JOB_ID:-x}.log
cd $ROBOT
up=0
for attempt in 1 2 3; do
  $SPY $SRVARGS --port $PORT > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  PORT=$((9000 + RANDOM % 900))
done
[ $up = 1 ] || { echo "[probe] server failed"; tail -30 $LOG; exit 3; }
echo "[probe] $MODEL server up on $PORT ($(date))"
CL="examples/libero/.venv/bin/python $REPO/experiments/fp_probe_client.py"
export PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero
FP_OUT=$OUTD/${MODEL}_s1_dev.jsonl timeout 7200 $CL --scene 1 --port $PORT --suite n2_goal --tasks fam --trials 1 --out $OUTD/${MODEL}_s1_dev.json --fresh $REPO/results/flawed/n2_states --branches $REPO/results/flawed/n2_branches_main.json --videos 0 --bind track
echo "[probe] s1 dev exit=$? ($(date))"
FP_OUT=$OUTD/${MODEL}_s1_fresh.jsonl timeout 7200 $CL --scene 1 --port $PORT --suite n2_goal --tasks fam --trials 1 --out $OUTD/${MODEL}_s1_fresh.json --fresh $REPO/results/flawed/n2_states_fresh --branches $REPO/results/flawed/n2_branches_fresh.json --videos 0 --bind track
echo "[probe] s1 fresh exit=$? ($(date))"
FP_OUT=$OUTD/${MODEL}_s2_std.jsonl timeout 7200 $CL --scene 2 --port $PORT --suite s2_living --tasks fam --trials 1 --out $OUTD/${MODEL}_s2_std.json --fresh $REPO/results/flawed/s2/states --branches $REPO/results/flawed/s2/branches_main.json --videos 0 --bind track
echo "[probe] s2 std exit=$? ($(date))"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
exit 0

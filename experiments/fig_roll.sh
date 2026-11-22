#!/bin/bash
# Display-frame rollouts for the figures (docs/notes/fig_plan.md 3 / 5(a)).  One server, then experiments/fig_roll_client.py around the
# source client of each (arm, tid): the server lines are copied from the source roll scripts, the client arguments are theirs; only the
# client script is wrapped (policy inputs untouched).  Dev states only (n2_states + n2_branches_main.json, rg_comp states + br7/br6 main,
# mt states).  Output under results/flawed/fig_frames/<OUT>/ only; finished outputs are skipped.
#   SERVER pi05 | pi0 | smolvla | groot | r2 | raa | raac      server lines of: pi05 = pp_atlas_roll.sh / rg_comp_roll.sh; pi0, smolvla =
#                       pp_atlas_more_roll.sh; groot = pp_atlas_roll.sh; r2, raa = mt_roll.sh; raac = raaw5_roll.sh (CKPT, CONFIG, WHERE_OWN)
#   HARN   n2 (suite n2_goal, tids of n2_tasks.json) | rgc (rg_comp states) | mt (cells in CELLS; tid 0 for K1 cells)
#   ARMS   '+'-separated.  n2: I (pp_atlas_client), rgp (rg.py client), g / sb (f1m_cue.py client), tvp (tv_frozen.py);
#          rgc: I rgp rc f7r mem (rg_comp.py client); mt: I r2 raa rgp (mt_client.py); raac: raac (raa_w.py client)
#   TIDS   '+'-separated (n2, rgc)     CELLS  '+'-separated (mt)
#   NFIRST first N entries of the branches file (default 10), SKIP skip M of them first (the 10-more rerun), OUT subdir name, SMOKE=1 -> 2 episodes
#   CKPT, CONFIG, WHERE_OWN (raac only), BR (override), PORT
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/figroll_%x_%j.out
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
SERVER=${SERVER:?SERVER}
HARN=${HARN:-n2}
ARMS=${ARMS:?ARMS}
OUT=${OUT:?OUT}
NFIRST=${NFIRST:-10}
SKIP=${SKIP:-0}
[ "${SMOKE:-0}" = 1 ] && { NFIRST=2; OUT=smoke/$OUT; }
OUTD=$REPO/results/flawed/fig_frames/$OUT
mkdir -p $OUTD
case "$OUTD" in $REPO/results/flawed/fig_frames/*) ;; *) echo "[fig] OUTD outside fig_frames"; exit 2 ;; esac
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot OMP_NUM_THREADS=2
CK=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch
SEGCKPT=${SEGCKPT:-/coc/flash10/ksingara3/robot/data/fa1/seg/ckpt/unet_main.npz}
SPY=$ROBOT/.venv/bin/python
case "$SERVER" in pi0) export BACKBONE_CONFIG=pi0_libero; unset BACKBONE_CKPT ;; smolvla) export BACKBONE_CONFIG=smolvla_libero BACKBONE_CKPT=lerobot/smolvla_libero ;; esac
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=$OUTD/server_${SERVER}_${SLURM_JOB_ID:-x}.log
server_cmd() {
  case "$SERVER" in
    pi05)    echo "$SPY $REPO/experiments/pp_serve_purify.py --port $PORT --model pi05 --mode intact --comp= --seed 13" ;;
    groot)   echo "/coc/flash10/ksingara3/robot/groot/.venv/bin/python $REPO/experiments/pp_serve_purify.py --port $PORT --model groot --mode intact --comp= --seed 13" ;;
    pi0)
             echo "$SPY $REPO/experiments/serve_guided.py --mode ours --seed 13 --port $PORT" ;;
    smolvla)
             echo "/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python $REPO/experiments/serve_guided.py --mode ours --seed 13 --port $PORT" ;;
    r2)      echo "$SPY $REPO/experiments/sc_rv.py serve --arm rview2 --port $PORT --ckpt $CK/pi05_g8v_rview2_ae_tl6000 --seed 13" ;;
    raa)     echo "$SPY $REPO/experiments/raa.py serve --arm raa --port $PORT --ckpt $CK/pi05_g8v_raa_ae_tl6000 --seed 13 --seg-ckpt $SEGCKPT --chan-diag $OUTD/chan_$SLURM_JOB_ID.jsonl --diag $OUTD/segdiag_$SLURM_JOB_ID.json" ;;
    raac)    echo "$SPY $REPO/experiments/raa_w.py serve --arm raac --port $PORT --ckpt $CKPT --seed 13 --seg-ckpt $SEGCKPT --where_own ${WHERE_OWN:?} --where_log $OUTD/where_$SLURM_JOB_ID.jsonl --chan-diag $OUTD/chan_$SLURM_JOB_ID.jsonl --diag $OUTD/segdiag_$SLURM_JOB_ID.json ${CONFIG:+--config $CONFIG}" ;;
    *) echo "unknown SERVER=$SERVER" >&2; exit 2 ;;
  esac
}
echo "[fig] $(date) $(hostname) job ${SLURM_JOB_ID:-x} server $SERVER harn $HARN arms $ARMS tids ${TIDS:-} cells ${CELLS:-} nfirst $NFIRST skip $SKIP out $OUTD"
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  CMD=$(server_cmd) || exit 2
  echo "[fig] server: $CMD"
  eval "$CMD" > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRV 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  kill $SRV 2>/dev/null
  if grep -q "address already in use" $LOG 2>/dev/null; then PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[fig] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[fig] no server"; exit 3; }
echo "[fig] $SERVER server up on $PORT ($(date))"
FC="examples/libero/.venv/bin/python $REPO/experiments/fig_roll_client.py --fig-first $NFIRST --fig-skip $SKIP"
LP=${PYTHONPATH:-}:$ROBOT/third_party/libero
rc=0
run() {   # run NAME CLIENT-WITH-ARGS...
  local out=$1; shift
  [ -f $out ] && { echo "[fig] $out exists"; return 0; }
  PYTHONPATH=$LP timeout 10000 $FC "$@" --out $out
  local r=$?; echo "[fig] $(basename $out) client exit=$r ($(date))"; [ $r = 0 ] || rc=$r
}
case "$HARN" in
  n2)
    FRESH=$REPO/results/flawed/n2_states; BRF=${BR:-$REPO/results/flawed/n2_branches_main.json}
    for arm in ${ARMS//+/ }; do
      case $arm in
        I)    CL="pp_atlas_client.py" ;;
        rgp)  CL="rg.py client --arm rgp" ;;
        g|sb|sf) CL="f1m_cue.py client --arm $arm" ;;
        tvp)  CL="tv_frozen.py client --arm tvp" ;;
        raac) CL="raa_w.py client --arm raac --fig-install-rec" ;;
        *) echo "[fig] unknown arm $arm"; rc=2; continue ;;
      esac
      for tid in ${TIDS//+/ }; do
        run $OUTD/${SERVER}_${arm}_${tid}.json $CL --port $PORT --suite n2_goal --tasks $tid --trials 1 --fresh $FRESH --branches $BRF --videos 0 --bind track
      done
    done ;;
  rgc)
    ST=$REPO/results/flawed/rg_comp/states
    for arm in ${ARMS//+/ }; do
      for tid in ${TIDS//+/ }; do
        case $arm in I) ;; rgp|rc) case $tid in 0|1|2|3|7|8|10) ;; *) continue ;; esac ;; f7r) case $tid in 0|1|2|3) ;; *) continue ;; esac ;; mem) [ $tid = 8 ] || continue ;; esac   # rg_comp_roll.sh's registered cells
        if [ $tid -le 7 ]; then BRF=$REPO/results/flawed/rg_comp/br7_main.json; else BRF=$REPO/results/flawed/rg_comp/br6_main.json; fi
        run $OUTD/${SERVER}_${arm}_${tid}.json rg_comp.py client --arm $arm --port $PORT --suite rgc --tasks $tid --trials 1 --fresh $ST --branches $BRF --videos 0 --bind track
      done
    done ;;
  s2)   # scene 2 (s2_roll.sh / s2rg_roll.sh): arms I (s2.py), rgp (s2rg.py, needs the s2fix3_det.py detector); tids 2+3 (N1, N2)
    S2F=${FRESH:-$REPO/results/flawed/s2/states}; S2B=${BR:-$REPO/results/flawed/s2/branches_main.json}
    DET=""
    case "+$ARMS+" in *+rgp+*)
      DPORT=$((9000 + RANDOM % 900)); for k in $(seq 1 50); do [ $DPORT != $PORT ] && free_port $DPORT && break; DPORT=$((9000 + RANDOM % 900)); done
      DLOG=$OUTD/det_${SERVER}_${SLURM_JOB_ID:-x}.log
      ( unset PYTHONPATH; cd $REPO; export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home TRANSFORMERS_OFFLINE=1
        exec $ROBOT/.venv/bin/python $REPO/experiments/s2fix3_det.py --port $DPORT ) > $DLOG 2>&1 &
      DET=$!
      for i in $(seq 1 120); do sleep 3; grep -q "detector listening" $DLOG 2>/dev/null && break; kill -0 $DET 2>/dev/null || break; done
      export S2FIX3_DET_PORT=$DPORT ;;
    esac
    for arm in ${ARMS//+/ }; do
      case $arm in I) CL="s2.py client --arm I" ;; rgp) CL="s2rg.py client --arm rgp" ;; *) echo "unknown s2 arm"; rc=2; continue ;; esac
      for tid in ${TIDS//+/ }; do
        run $OUTD/${SERVER}_${arm}_${tid}.json $CL --port $PORT --suite s2_living --tasks $tid --trials 1 --fresh $S2F --branches $S2B --videos 0 --bind track
      done
    done
    [ -n "$DET" ] && kill $DET 2>/dev/null ;;
  mt)
    for arm in ${ARMS//+/ }; do
      SEGF=""; [ $arm = r2 ] && SEGF="--seg"
      for cell in ${CELLS//+/ }; do
        case "$cell" in K1_in|T_near|T_far|O1_near|O1_far) tid=0 ;; K2_in|O2_near|O2_far) tid=2 ;; *) echo "bad cell $cell"; continue ;; esac
        run $OUTD/${arm}_${cell}.json mt_client.py --arm $arm $SEGF --port $PORT --suite n2_goal --tasks $tid --trials 1 --fresh $REPO/results/flawed/mt/states --branches $REPO/results/flawed/mt/mt_br_${cell}_main.json --videos 0 --bind track
      done
    done ;;
  *) echo "unknown HARN"; rc=2 ;;
esac
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "exit=$rc | $(date)"
exit $rc

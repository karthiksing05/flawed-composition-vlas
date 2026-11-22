#!/bin/bash
# plan_oracle_pi0 copy of a2f_capture.sh (a2f_capture.sh is not edited): a generic `*)` case, default OUTD results/flawed/pz/a2/cap_<POLICY>, own log prefix.
# RAA-F probe CAPTURE (docs/notes/raaf_probe_runbook.md): server (experiments/a2f_capture.py) + client on one GPU, N + K hold-phase queries (a2_cap.sh's nk mode, SR 83 design).
# A copy of a2_cap.sh for the servo-free state-token models; the F7 mode is dropped (the plan pools N + K).
#   POLICY  A | A0 | Aplus | B | raasl      label of the model; files are named <POLICY>_nk_<tid>_cap.jsonl, obs ids <POLICY>_<base>
#   PHASE   smoke | main                    smoke: OUTD must be outside results/ ; main: refuses without the a2 prereg row
#   TIDS    '+'-separated, default K 0+2+6+10 and N 16+22+24+26 (submit K and N as two jobs with the SAME OUTD)
#   OUTD    default (main) $REPO/results/flawed/raaf/a2/cap_<POLICY>        BRFILE overrides the branches file (smoke)
#   CONFIG / CKPT   override the registry below (Aplus: required; smoke: the scratch export)
#   Route B serves raa_b.py (crop in the third image) and the client a2f_cap_client.py with A2F_RAA=raa_b; the others serve raa.py --arm raac.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/a2fx_cap_%x_%j.out
set -uo pipefail
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
ROBOT=/coc/flash10/ksingara3/robot/openpi
POLICY=${POLICY:?POLICY}; PHASE=${PHASE:?PHASE}
CK=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch
SEGCKPT=/coc/flash10/ksingara3/robot/data/fa1/seg/ckpt/unet_main.npz
TAG=${TAG:-main6000}
case "$POLICY" in
  A)     SERVE=raa.py;   CONFIG=${CONFIG:-pi05_g8v_raac_lf_ae}; CKPT=${CKPT:-$CK/pi05_g8v_raac_lf_ae_$TAG}; export A2F_RAA=raa ;;
  A0)    SERVE=raa.py;   CONFIG=${CONFIG:-pi05_g8v_raac_l_ae};  CKPT=${CKPT:-$CK/pi05_g8v_raac_l_ae_$TAG};  export A2F_RAA=raa ;;
  Aplus) SERVE=raa.py;   CONFIG=${CONFIG:?Aplus: CONFIG (the A+ config name)}; CKPT=${CKPT:?Aplus: CKPT}; export A2F_RAA=raa ;;
  B)     SERVE=raa_b.py; CONFIG=${CONFIG:-pi05_g8v_raac_lb_ae}; CKPT=${CKPT:-$CK/pi05_g8v_raac_lb_ae_$TAG}; export A2F_RAA=raa_b ;;
  raasl) SERVE=raa.py;   CONFIG=${CONFIG:-pi05_g8v_raas_l_ae};  CKPT=${CKPT:-$CK/pi05_g8v_raas_l_ae_main6000}; export A2F_RAA=raa ;;   # reference only: SR 83's captures exist, not re-run
  *)     SERVE=raa.py;   CONFIG=${CONFIG:?POLICY $POLICY: CONFIG}; CKPT=${CKPT:?POLICY $POLICY: CKPT}; export A2F_RAA=raa ;;   # plan_oracle_pi0: pzsL / pzA0 / pzA / orc: any label, config and checkpoint given
esac
[ -f $CKPT/model.safetensors ] || { echo "[a2fcap] no checkpoint $CKPT"; exit 2; }
case "$PHASE" in
  smoke) OUTD=${OUTD:?smoke needs OUTD outside results/}; case "$OUTD" in */results/*) echo "[a2fcap] smoke OUTD inside results/"; exit 2;; esac ;;
  main)  OUTD=${OUTD:-$REPO/results/flawed/pz/a2/cap_${POLICY}}
         ls $REPO/ledger/m3_prereg/m3_prereg__libero_goal__a2__s0__*.json > /dev/null 2>&1 || { echo "[a2fcap] no a2 prereg row: refusing"; exit 4; } ;;
  *) echo "PHASE"; exit 2 ;;
esac
TIDS=${TIDS:-0+2+6+10+16+22+24+26}; FRESH=$REPO/results/flawed/n2_states; SUITE=n2_goal; BR=${BRFILE:-$REPO/results/flawed/n2_branches_main.json}
[ -f $BR ] && [ -d $FRESH ] || { echo "[a2fcap] no branches/states $BR"; exit 2; }
mkdir -p $OUTD
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=2
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export A2_OUT=$OUTD/obs
CAP=$REPO/experiments/a2f_capture.py
SPY=$ROBOT/.venv/bin/python
SRV="$CAP $REPO/experiments/$SERVE serve --arm raac --ckpt $CKPT --config $CONFIG --seed 13 --seg-ckpt $SEGCKPT"
echo "[a2fcap] $(date) $(hostname) job ${SLURM_JOB_ID:-x} restart ${SLURM_RESTART_COUNT:-0} $POLICY nk $PHASE tids $TIDS outd $OUTD br $BR"
echo "[a2fcap] server: $SRV"
PORT=${PORT:-$((9000 + RANDOM % 900))}
LOG=$OUTD/server_${SLURM_JOB_ID:-x}_r${SLURM_RESTART_COUNT:-0}.log
cd $ROBOT
free_port() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" $1 2>/dev/null; }
up=0
for attempt in 1 2 3 4; do
  for k in $(seq 1 50); do free_port $PORT && break; PORT=$((9000 + RANDOM % 900)); done
  $SPY $SRV --port $PORT > $LOG 2>&1 &
  SRVPID=$!
  for i in $(seq 1 240); do sleep 5; if grep -q -E "server listening" $LOG 2>/dev/null; then up=1; break; fi; if ! kill -0 $SRVPID 2>/dev/null; then break; fi; done
  [ $up = 1 ] && break
  if grep -q "address already in use" $LOG 2>/dev/null; then echo "[a2fcap] port taken"; PORT=$((9000 + RANDOM % 900)); continue; fi
  echo "[a2fcap] server died"; tail -30 $LOG; exit 3
done
[ $up = 1 ] || { echo "[a2fcap] no server"; tail -30 $LOG; exit 3; }
echo "[a2fcap] server up on $PORT ($(date))"
rc=0
for tid in ${TIDS//+/ }; do
  out=$OUTD/${POLICY}_nk_${tid}.json
  [ -f $out ] && { echo "[a2fcap] $out exists"; continue; }
  rm -f $out.part
  CLI="--port $PORT --suite $SUITE --tasks $tid --trials 1 --out $out --fresh $FRESH --branches $BR --videos 0 --bind track"
  PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero timeout 10800 examples/libero/.venv/bin/python $REPO/experiments/a2f_cap_client.py client \
      --arm RA --ra-arm raac --scene 1 --steer-out $out.part --max-steps ${MAXSTEPS:-240} $CLI
  r=$?; echo "[a2fcap] $POLICY nk task $tid client exit=$r ($(date))"
  if [ $r = 0 ] && [ -f $out ]; then [ -f $out.part ] && mv $out.part ${out%.json}_cap.jsonl; else rc=$r; fi
done
kill $SRVPID 2>/dev/null
wait $SRVPID 2>/dev/null
echo "[a2fcap] n obs files: $(ls $A2_OUT 2>/dev/null | wc -l)"
echo "[a2fcap] exit=$rc | $(date)"
exit $rc

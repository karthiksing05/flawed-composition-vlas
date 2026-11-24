#!/bin/bash
# NPHS pipeline job wrapper (post-registration orchestration only; no statistics).  KIND = export | pcheck | rollK | rollN | readgate | readmain | e1arm | e1pool.
# Env: ALAB (pzH|pzA0|pzsL), SEED (42-45), TIDS (roll groups joined by +).  Resources / partition come from the sbatch command line.
# Every kind exits 0 WITHOUT work when its gate is not met (training incomplete, K gate failed, GPU-h cap reached), so nothing downstream burns GPU.
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nphs_gated_%x_%j.out
REPO=/coc/flash10/ksingara3/flawed-composition-vlas; PP=/coc/flash10/ksingara3/.claude_tmp/pp; CK=/coc/flash10/ksingara3/robot/checkpoints_l90c
R=$REPO/results/flawed/nphs7; ROBOT=/coc/flash10/ksingara3/robot/openpi; PY=$ROBOT/.venv/bin/python; JD=$REPO/docs/notes/nphs_jobs
CAP=${CAP_H:-130}
echo "[nphs-gated] $(date) $(hostname) kind $KIND arm ${ALAB:-} seed ${SEED:-} job ${SLURM_JOB_ID:-x}"
skip() { echo "[nphs-gated] SKIP ($1): exit 0 without work"; exit 0; }
spent() { sacct -u $USER -S 2026-11-24 -X -n -P --format=JobName,Elapsed,AllocTRES | awk -F'|' '$1 ~ /^nphs_/ && $3 ~ /gres\/gpu/ {n=split($2,p,/[-:]/); if(n==4) s=p[1]*86400+p[2]*3600+p[3]*60+p[4]; else s=p[1]*3600+p[2]*60+p[3]; t+=s} END{printf "%.2f", t/3600}'; }
cap_ok() { s=$(spent); echo "[nphs-gated] nphs_* GPU wall-h so far $s (cap $CAP)"; awk -v s=$s -v c=$CAP 'BEGIN{exit !(s<c)}'; }
cfg_of() { case $1 in pzH) echo pi0_g8v_raah_l_ae;; pzA0) echo pi0_g8v_raac_l_ae;; pzsL) echo pi0_g8v_raas_l_ae;; esac; }
ck_of() { if [ "$2" = 42 ]; then echo $CK/pytorch/$(cfg_of $1)_tl6000; else echo $CK/pytorch/$(cfg_of $1)_s$2_6000; fi; }
gate_ok() { grep -q '"passes": true' $R/nphs_gate_s$1.json 2>/dev/null; }
cells() { for t in $2; do [ -f $R/a_$1/$1_main_$t.json ] || return 1; done; return 0; }
KT="0 2 4 6 8 10"; NT="16 18 20 22 24 26"
cd $REPO; unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
if [ -n "${ALAB:-}" ]; then CFG=$(cfg_of $ALAB); PT=$(ck_of $ALAB $SEED); fi
roll() {
  [ -f $PT/model.safetensors ] || skip "no exported checkpoint $PT"
  case $ALAB in pzH) RAAPY=raa_wh0.py; WHERE_LOG=1;; *) RAAPY=raa.py; unset WHERE_LOG;; esac
  export ARM=raac NAME=${ALAB}_s$SEED RAAPY CKPT=$PT CONFIG=$CFG OUTD=$R/a_${ALAB}_s$SEED TIDS
  bash $JD/nphs_roll.sh; exit $?
}
case $KIND in
  export)   [ -d $CK/jax/$CFG/tl_s$SEED/6000 ] || skip "training did not reach step 6000"; [ -d $PT ] && skip "already exported"
            ARGS="--name~$CFG~--exp~tl_s$SEED~--export~--step~6000~--tag~s${SEED}_6000" bash $JD/nphs_export.sh; exit $? ;;
  pcheck)   [ -f $PT/model.safetensors ] || skip "no exported checkpoint"; EXTRA="--torch-ckpt~$PT" bash $JD/nphs_check.sh; exit $? ;;
  rollK)    cap_ok || skip "GPU-h cap"; roll ;;
  rollN)    if [ $ALAB = pzH ]; then gate_ok $SEED || skip "K gate not passed / not read for seed $SEED"; fi; cap_ok || skip "GPU-h cap"; roll ;;
  readgate) cells pzH_s$SEED "$KT" || skip "pzH seed $SEED K cell missing (rollout failed or gated off)"; [ -f $R/nphs_gate_s$SEED.json ] && skip "gate json exists"
            $PY experiments/nphs_read.py gate --seed $SEED --out results/flawed/nphs7/nphs_gate_s$SEED.json; exit $? ;;
  readmain) for s in 42 43 44 45; do [ -f $R/nphs_gate_s$s.json ] || skip "gate json seed $s missing (no read until all expected cells exist, k4)"
              for a in pzA0 pzsL; do cells ${a}_s$s "$KT $NT" || skip "cell $a seed $s missing (k4)"; done
              if gate_ok $s; then cells pzH_s$s "$NT" || skip "pzH seed $s N cell missing (k4)"; fi; done
            $PY experiments/nphs_read.py main --out results/flawed/nphs7/nphs_read_main.json; exit $? ;;
  e1arm)    if [ $ALAB = pzH ]; then gate_ok $SEED || skip "K gate not passed seed $SEED"; fi; cells ${ALAB}_s$SEED "$NT" || skip "cells missing"
            $PY experiments/nphs_e1.py arm --arm $ALAB --seed $SEED --out results/flawed/nphs7/e1_${ALAB}_s$SEED.json; exit $? ;;
  e1pool)   $PY experiments/nphs_e1.py pool --arm $ALAB --out results/flawed/nphs7/e1_${ALAB}_pooled.json; exit $? ;;
  *) echo "bad KIND"; exit 2 ;;
esac

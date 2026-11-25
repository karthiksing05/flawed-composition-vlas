#!/bin/bash
# NPHJ pipeline job wrapper (post-registration orchestration only; no statistics).  KIND = export | rollK | rollN | readgate | readmain | e1arm | e1pool.
# Env: ALAB (pzA24 = pi0-A0 trained 24k steps | pzJ = jointly trained where-branch), SEED, TIDS (roll groups joined by +).  Resources / partition come from the sbatch command line.
# Every kind exits 0 WITHOUT work when its gate is not met (training incomplete, K gate failed, GPU-h cap reached), so nothing downstream burns GPU.
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nphj_gated_%x_%j.out
REPO=/coc/flash10/ksingara3/flawed-composition-vlas; PP=/coc/flash10/ksingara3/.claude_tmp/pp; CK=/coc/flash10/ksingara3/robot/checkpoints_l90c
R=$REPO/results/flawed/nphj7; ROBOT=/coc/flash10/ksingara3/robot/openpi; PY=$ROBOT/.venv/bin/python; JD=$REPO/docs/notes/nphj_jobs_r2
CAP=${CAP_H:-110}
echo "[nphj-gated] $(date) $(hostname) kind $KIND arm ${ALAB:-} seed ${SEED:-} job ${SLURM_JOB_ID:-x}"
skip() { echo "[nphj-gated] SKIP ($1): exit 0 without work"; exit 0; }
spent() { sacct -u $USER -S 2026-11-24 -X -n -P --format=JobName,Elapsed,AllocTRES | awk -F'|' '$1 ~ /^nphj_/ && $3 ~ /gres\/gpu/ {n=split($2,p,/[-:]/); if(n==4) s=p[1]*86400+p[2]*3600+p[3]*60+p[4]; else s=p[1]*3600+p[2]*60+p[3]; t+=s} END{printf "%.2f", t/3600}'; }
cap_ok() { s=$(spent); echo "[nphj-gated] nphj_* GPU wall-h so far $s (cap $CAP)"; awk -v s=$s -v c=$CAP 'BEGIN{exit !(s<c)}'; }
cfg_of() { case $1 in pzA24) echo pi0_g8v_raac_l_ae;; pzJ) echo pi0_g8v_raaj_l_ae;; esac; }
ck_of() { case $1 in pzA24) echo $CK/pytorch/pi0_g8v_raac_l_ae_t24k_s$2;; pzJ) echo $CK/pytorch/pi0_g8v_raaj_l_ae_s$2_6000;; esac; }
gate_ok() { grep -q '"passes": true' $R/nphj_gate_$1_s$2.json 2>/dev/null; }
cells() { for t in $2; do [ -f $R/a_$1/$1_main_$t.json ] || return 1; done; return 0; }
KT="0 2 4 6 8 10"; NT="16 18 20 22 24 26"
cd $REPO; unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
if [ -n "${ALAB:-}" ]; then CFG=$(cfg_of $ALAB); PT=$(ck_of $ALAB $SEED); fi
roll() {
  [ -f $PT/model.safetensors ] || skip "no exported checkpoint $PT"
  case $ALAB in pzJ) RAAPY=raa_wj.py; WHERE_LOG=1;; *) RAAPY=raa.py; unset WHERE_LOG;; esac
  export ARM=raac NAME=${ALAB}_s$SEED RAAPY CKPT=$PT CONFIG=$CFG OUTD=$R/a_${ALAB}_s$SEED TIDS
  bash $JD/nphj_roll.sh; exit $?
}
case $KIND in
  export)   case $ALAB in pzA24) fin=24000; exp=t24k_s$SEED; tag=t24k_s$SEED;; pzJ) fin=6000; exp=tj_s$SEED; tag=s${SEED}_6000;; esac
            [ -d $CK/jax/$CFG/$exp/$fin ] || skip "training did not reach step $fin"; [ -d $PT ] && skip "already exported"
            [ $ALAB = pzJ ] && { [ -f $R/checks_ok.marker ] || skip "no checks_ok marker"; }
            ARGS="--name~$CFG~--exp~$exp~--export~--step~$fin~--tag~$tag" bash $JD/nphj_export.sh; exit $? ;;
  rollK)    cap_ok || skip "GPU-h cap"; roll ;;
  rollN)    gate_ok $ALAB $SEED || skip "K gate not passed / not read for $ALAB seed $SEED"; cap_ok || skip "GPU-h cap"; roll ;;
  readgate) cells ${ALAB}_s$SEED "$KT" || skip "$ALAB seed $SEED K cell missing (rollout failed or gated off)"; [ -f $R/nphj_gate_${ALAB}_s$SEED.json ] && skip "gate json exists"
            $PY experiments/nphj_read.py gate --arm $ALAB --seed $SEED --out results/flawed/nphj7/nphj_gate_${ALAB}_s$SEED.json; exit $? ;;
  readmain) for s in 42 43 44 45; do for a in pzA24 pzJ; do
              [ $a = pzA24 ] && [ $s -gt 43 ] && continue
              [ -f $R/nphj_gate_${a}_s$s.json ] || skip "gate json $a seed $s missing (no read until all expected cells exist, k4)"
              cells ${a}_s$s "$KT" || skip "K cell $a seed $s missing (k4)"
              if gate_ok $a $s; then cells ${a}_s$s "$NT" || skip "N cell $a seed $s missing (k4)"; fi; done; done
            $PY experiments/nphj_read.py main --out results/flawed/nphj7/nphj_read_main.json; exit $? ;;
  e1arm)    gate_ok $ALAB $SEED || skip "K gate not passed $ALAB seed $SEED"; cells ${ALAB}_s$SEED "$NT" || skip "cells missing"
            $PY experiments/nphj_e1.py arm --arm $ALAB --seed $SEED --out results/flawed/nphj7/e1_${ALAB}_s$SEED.json; exit $? ;;
  e1pool)   $PY experiments/nphj_e1.py pool --arm $ALAB --out results/flawed/nphj7/e1_${ALAB}_pooled.json; exit $? ;;
  *) echo "bad KIND"; exit 2 ;;
esac

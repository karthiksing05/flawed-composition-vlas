#!/bin/bash
# NPH pipeline job wrapper (post-registration orchestration only; no statistics).  KIND = export | pcheck | rollK | rollN | readgate | readmain | e1arm | e1dec.  Resources come from the sbatch command line.
# Every kind exits 0 WITHOUT work when its gate is not met (training incomplete, K gate failed, GPU-h cap reached), so nothing downstream burns GPU.
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nph_gated_%x_%j.out
REPO=/coc/flash10/ksingara3/flawed-composition-vlas; PP=/coc/flash10/ksingara3/.claude_tmp/pp; CK=/coc/flash10/ksingara3/robot/checkpoints_l90c
R=$REPO/results/flawed/nph6; PT=$CK/pytorch/pi0_g8v_raah_l_ae_tl6000; ROBOT=/coc/flash10/ksingara3/robot/openpi; PY=$ROBOT/.venv/bin/python
CAP=${CAP_H:-21.5}
echo "[nph-gated] $(date) $(hostname) kind $KIND job ${SLURM_JOB_ID:-x}"
skip() { echo "[nph-gated] SKIP ($1): exit 0 without work"; exit 0; }
spent() { sacct -u $USER -S 2026-11-23 -X -n -P --format=JobName,Elapsed | awk -F'|' '$1 ~ /^nph_/ {n=split($2,p,/[-:]/); if(n==4) s=p[1]*86400+p[2]*3600+p[3]*60+p[4]; else s=p[1]*3600+p[2]*60+p[3]; t+=s} END{printf "%.2f", t/3600}'; }
gate_ok() { grep -q '"passes": true' $R/nph_read_gate.json 2>/dev/null; }
cap_ok() { s=$(spent); echo "[nph-gated] nph_* wall-h so far $s (cap $CAP)"; awk -v s=$s -v c=$CAP 'BEGIN{exit !(s<c)}'; }
cd $REPO; unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
case $KIND in
  export)   [ -d $CK/jax/pi0_g8v_raah_l_ae/tl/6000 ] || skip "training did not reach step 6000"; [ -d $PT ] && skip "already exported"
            ARGS='--name~pi0_g8v_raah_l_ae~--exp~tl~--export~--step~6000~--tag~tl6000' bash $PP/nph_export.sh; exit $? ;;
  pcheck)   [ -f $PT/model.safetensors ] || skip "no exported checkpoint"; EXTRA="--torch-ckpt~$PT" bash $PP/nph_check.sh; exit $? ;;
  rollK)    [ -f $PT/model.safetensors ] || skip "no exported checkpoint"; cap_ok || skip "GPU-h cap"; bash $PP/nph_roll.sh; exit $? ;;
  rollN)    [ -f $PT/model.safetensors ] || skip "no exported checkpoint"; gate_ok || skip "K gate not passed / not read"; cap_ok || skip "GPU-h cap"; bash $PP/nph_roll.sh; exit $? ;;
  readgate) for t in 0 2 4 6 8 10; do [ -f $R/a_pzH/pzH_main_$t.json ] || skip "pzH K cell $t missing (rollout failed or gated off)"; done
            [ -f $R/nph_read_gate.json ] && skip "gate json exists"; $PY experiments/nph_read.py read --part gate --out results/flawed/nph6/nph_read_gate.json; exit $? ;;
  readmain) gate_ok || skip "K gate not passed"; for a in pzH pzA0 pzsL; do for t in 0 2 4 6 8 10 16 18 20 22 24 26; do [ -f $R/a_$a/${a}_main_$t.json ] || skip "cell $a $t missing: no read until all cells exist (k4)"; done; done
            $PY experiments/nph_read.py read --part main --out results/flawed/nph6/nph_read_main.json; exit $? ;;
  e1arm)    gate_ok || skip "K gate not passed"; [ -f $R/a_$ARMX/${ARMX}_main_26.json ] || skip "cells missing"; $PY experiments/nph_e1.py arm --arm $ARMX --out results/flawed/nph6/e1_$ARMX.json; exit $? ;;
  e1dec)    gate_ok || skip "K gate not passed"; [ -f $R/a_$ARMA/${ARMA}_main_26.json ] && [ -f $R/a_$ARMB/${ARMB}_main_26.json ] || skip "cells missing"; $PY experiments/nph_e1.py decomp --a $ARMA --b $ARMB --out results/flawed/nph6/e1_dec_${ARMA}_${ARMB}.json; exit $? ;;
  *) echo "bad KIND"; exit 2 ;;
esac

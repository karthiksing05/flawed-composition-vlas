#!/bin/bash
# RAA-F probe: JAX -> PyTorch export (CPU).  A copy of t16_export.sh with the full node exclusion list, plus a scratch mode.
#   registered export:  --export=ALL,ARGS='--g8v~raac_lf~--exp~main~--export~--step~6000~--tag~main6000'   (c_train_pi05.py's arguments, joined by '~')
#   scratch export:     --export=ALL,SRC=<jax step dir>,DST=<scratch pytorch dir>   (raac_export.py directly + assets copy; used for the smoke checkpoint)
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nphp_export_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
if [ -n "${SRC:-}" ]; then
  [ -n "${DST:-}" ] && [ -d "$SRC" ] || { echo "[nphp-export] SRC / DST"; exit 2; }
  case "$DST" in /coc/flash10/ksingara3/.claude_tmp/*) ;; *) echo "[nphp-export] scratch DST must be under .claude_tmp"; exit 2;; esac
  echo "[nphp-export] $(date) $(hostname) scratch $SRC -> $DST"
  (cd /coc/flash10/ksingara3/robot/openpi && $PY /coc/flash10/ksingara3/flawed-composition-vlas/experiments/raac_export.py --checkpoint_dir $SRC --output_path $DST)
  rc=$?
  [ $rc = 0 ] && [ ! -d $DST/assets ] && cp -r $SRC/assets $DST/assets
else
  x=${ARGS//\~/ }
  echo "[nphp-export] $(date) $(hostname): $x"
  $PY experiments/c_train_nphp.py $x
  rc=$?
fi
echo "exit=$rc | $(date)"
exit $rc

#!/bin/bash
# NPHJ pre-training checks on one overcap GPU: (1) identity check (raaj_check.py identity), (2) 30-step smoke training of pi0_g8v_raaj_l_ae, (3) export (raaj_export.py: exact tensors, jax vs torch parity < 1e-4 on 100 states),
# (4) post check on the exported smoke checkpoint.  Writes results/flawed/nphj7/checks_ok.marker ONLY if all four exit 0.  The J training links exit 0 without work unless the marker exists.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/nphj_check_%x_%j.out
cd /coc/flash10/ksingara3/flawed-composition-vlas
unset PYTHONPATH
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
CK=/coc/flash10/ksingara3/robot/checkpoints_l90c; NM=pi0_g8v_raaj_l_ae; TAG=smoke_${SLURM_JOB_ID}
MK=results/flawed/nphj7/checks_ok.marker
mkdir -p results/flawed/nphj7
[ -f $MK ] && { echo "[nphj-check] marker exists"; exit 0; }
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "[nphj-check] 1 identity"; srun $PY experiments/raaj_check.py identity || { echo "[nphj-check] FAIL identity"; exit 11; }
echo "[nphj-check] 2 smoke training (30 steps)"; srun $PY experiments/c_train_nphj.py --name $NM --exp $TAG --steps 31 --batch 32 --save-interval 30 --keep-period 30 --seed 42 || { echo "[nphj-check] FAIL smoke"; exit 12; }
[ -d $CK/jax/$NM/$TAG/30 ] || { echo "[nphj-check] FAIL no step-30 checkpoint"; exit 12; }
echo "[nphj-check] 3 export + parity"; JAX_PLATFORMS=cpu $PY experiments/c_train_nphj.py --name $NM --exp $TAG --export --step 30 --tag $TAG || { echo "[nphj-check] FAIL export"; exit 13; }
echo "[nphj-check] 4 post check"; srun $PY experiments/raaj_check.py post --jax_dir $CK/jax/$NM/$TAG/30 --torch_dir $CK/pytorch/${NM}_$TAG || { echo "[nphj-check] FAIL post"; exit 14; }
date > $MK; echo "[nphj-check] ALL CHECKS OK; marker written"
rm -rf $CK/jax/$NM/$TAG $CK/pytorch/${NM}_$TAG results/flawed/nphj7/jmag_$TAG.json
exit 0

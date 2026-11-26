#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
# XB setup: MESA/BiMESA checkpoints (not yet downloaded) + BiMESA assets.  CPU only.
R=/coc/flash10/ksingara3/robot; B=$R/bench; export HF_HOME=$B/hf PYTHONUNBUFFERED=1
$R/openpi/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
B="/coc/flash10/ksingara3/robot/bench"
for r in ["mesa-pi0","mesa-pi0-fast","bimesa-pi0","bimesa-pi05","bimesa-pi0-fast","mesa-gr00t-n1.6","bimesa-gr00t-n1.6"]:
    try:
        p=snapshot_download("albertwilcox/"+r, local_dir=f"{B}/mesa/checkpoints/{r}"); print("downloaded",r,p, flush=True)
    except Exception as e: print("FAILED",r,repr(e), flush=True)
PY
cd $B/mesa/mesa-env; export UV_PROJECT_ENVIRONMENT=$B/mesa/venv UV_CACHE_DIR=$R/.uv-cache UV_PYTHON_INSTALL_DIR=$R/.uv-python PATH=$R/simpler/uvtool/bin:$PATH
uv run --no-sync python scripts/setup/download_assets.py 2>&1 | tr '\r' '\n' | grep -v "MB$" | tail -5
du -sh $B/mesa/checkpoints/* $B/mesa/mesa-env/mesa/sim/assets
echo SETUP_MESA_DONE

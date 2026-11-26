#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -J xb_setup_rtextra
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
# XPolicyLab websocket deps for the pi0.5 (openpi venv) and SmolVLA (lerobot venv) policy servers, installed to a separate target dir (the shared envs are not modified)
R=/coc/flash10/ksingara3/robot; B=$R/bench; export UV_CACHE_DIR=$R/.uv-cache PATH=$R/simpler/uvtool/bin:$PATH
uv pip install --python $R/openpi/.venv/bin/python --target $B/robotwin/extra_pylibs --no-deps msgpack-numpy "websockets>=14" msgpack h5py pyyaml "pydantic>=2.5" opencv-python-headless 2>&1 | tail -3
ls $B/robotwin/extra_pylibs | head

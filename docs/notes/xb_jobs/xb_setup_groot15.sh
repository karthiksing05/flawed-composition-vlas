#!/bin/bash
# XB setup: GR00T N1.5 inference env (robocasa-benchmark/Isaac-GR00T fork).  CPU only.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
R=/coc/flash10/ksingara3/robot; B=$R/bench; export UV_CACHE_DIR=$R/.uv-cache UV_PYTHON_INSTALL_DIR=$R/.uv-python PATH=$R/simpler/uvtool/bin:$PATH HF_HOME=$B/hf PYTHONUNBUFFERED=1
V=$B/robocasa/groot15_venv; P=$V/bin/python
uv venv --clear --python 3.10 $V
cd $B/robocasa/gr00t-robocasa
uv pip install --python $P torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
uv pip install --python $P --no-deps -e . 2>&1 | tail -1
uv pip install --python $P albumentations==1.4.18 av==12.3.0 blessings==1.7 dm_tree==0.1.8 einops==0.8.1 gymnasium==1.0.0 h5py hydra-core==1.3.2 imageio==2.34.2 kornia==0.7.4 matplotlib numpy==1.26.4 numpydantic==1.6.7 omegaconf==2.3.0 opencv_python_headless==4.11.0.86 pandas==2.2.3 pydantic==2.10.6 PyYAML Requests timm==1.0.14 tqdm transformers==4.51.3 typing_extensions pyarrow fastparquet accelerate protobuf==3.20.3 tyro peft==0.17.0 pyzmq diffusers==0.30.2 fastapi uvicorn json-numpy requests safetensors 2>&1 | tail -3
uv pip install --python $P https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 2>&1 | tail -2
$P -c "import torch, gr00t; from gr00t.model.policy import Gr00tPolicy; print('gr00t ok', torch.__version__)" 2>&1 | tail -3
$P -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1 | tail -1
du -sh $V; echo SETUP_G15_DONE

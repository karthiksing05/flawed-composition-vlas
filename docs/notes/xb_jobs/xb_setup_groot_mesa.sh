#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
# XB setup: GR00T-N1.6 inference env for MESA (mesa-GR00T, uv, Python 3.10).  CPU build of flash-attn needs nvcc: borrow the RoboTwin conda env's CUDA 12.1 + gcc 12.
R=/coc/flash10/ksingara3/robot; B=$R/bench; E=$B/robotwin/env
export HF_HOME=$B/hf PYTHONUNBUFFERED=1 UV_CACHE_DIR=$R/.uv-cache UV_PYTHON_INSTALL_DIR=$R/.uv-python PATH=$E/bin:$R/simpler/uvtool/bin:$PATH
export CUDA_HOME=$E TORCH_CUDA_ARCH_LIST="8.6" FLASH_ATTN_CUDA_ARCHS="80" MAX_JOBS=8 CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++ NVCC_PREPEND_FLAGS="-ccbin $E/bin/x86_64-conda-linux-gnu-g++"
export CPATH=$E/targets/x86_64-linux/include LIBRARY_PATH=$E/targets/x86_64-linux/lib
cd $B/mesa/mesa-GR00T
UV_PROJECT_ENVIRONMENT=$B/mesa/groot_venv uv sync 2>&1 | tail -15
$B/mesa/groot_venv/bin/python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)" 2>&1 | tail -2
du -sh $B/mesa/groot_venv
echo SETUP_GROOT_DONE

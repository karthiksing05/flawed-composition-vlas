#!/bin/bash
# PLAN_SOLVE S3b setup: RoboPoint (github wentaoyuan/RoboPoint @ 206cb1e) + weights wentao-yuan/robopoint-v1-vicuna-v1.5-13b into
# their own directory /coc/flash10/ksingara3/s3b_ext (untrusted downloads; no repo code lives there).  CPU job.
#SBATCH --job-name=s3bsetup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID)$'); do unset $v; done
set -x
EXT=/coc/flash10/ksingara3/s3b_ext
cd $EXT
export HF_HOME=$EXT/hf_home PIP_CACHE_DIR=$EXT/pip_cache
[ -d RoboPoint ] || git clone -q https://github.com/wentaoyuan/RoboPoint RoboPoint
git -C RoboPoint checkout -q 206cb1e086fc3960a8673725a8540766655cd9c5
sed -n 1,60p RoboPoint/pyproject.toml
ls RoboPoint/robopoint RoboPoint/robopoint/model RoboPoint/robopoint/eval 2>&1 | head -60
[ -x venv/bin/python ] || ~/.local/bin/python3.10 -m venv venv
venv/bin/python -I -m pip install -q --upgrade pip
venv/bin/python -I -m pip install -q -e "$EXT/RoboPoint" 2>&1 | tail -5
venv/bin/python -I -m pip install -q "numpy<2" 2>&1 | tail -2
venv/bin/python -I -c "import torch, transformers, robopoint; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
venv/bin/python -I -c "
from huggingface_hub import snapshot_download
p = snapshot_download('wentao-yuan/robopoint-v1-vicuna-v1.5-13b', local_dir='$EXT/weights/robopoint-v1-vicuna-v1.5-13b')
print('weights at', p)"
du -sh $EXT/weights/* $EXT/venv
grep -rn "def load_pretrained_model\|def process_images\|def tokenizer_image_token\|conv_templates = \|IMAGE_TOKEN_INDEX =" RoboPoint/robopoint | head
grep -rn "Locate several points\|Your answer should be formatted" RoboPoint | head -5
echo "exit=$? | $(date)"

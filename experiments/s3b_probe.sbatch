#!/bin/bash
# PLAN_SOLVE S3b: availability probe for RoboPoint / RoboRefer (no download of weights).  Outputs to slurm log only.
#SBATCH --job-name=s3bprobe
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID)$'); do unset $v; done
echo "host $(hostname)"; which uv git; ls ~/.local/bin 2>/dev/null | head; df -h /coc/flash10/ksingara3 | tail -1
quota -s 2>/dev/null | tail -3
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
cd /coc/flash10/ksingara3/s3b_ext
$PY -I - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
for rid in ["wentao-yuan/robopoint-v1-vicuna-v1.5-13b", "wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora", "wentao-yuan/robopoint-v1-llama-2-13b",
            "Zhoues/RoboRefer-2B-SFT", "Zhoues/RoboRefer-2B-Depth-Align", "Zhoues/RoboRefer-8B-SFT", "allenai/Molmo-7B-D-0924"]:
    try:
        info = api.model_info(rid, files_metadata=True)
        tot = sum((s.size or 0) for s in info.siblings)
        print("OK", rid, f"{tot/1e9:.1f} GB", [s.rfilename for s in info.siblings][:8])
    except Exception as e:
        print("NO", rid, type(e).__name__, str(e)[:120])
PYEOF
for r in wentaoyuan/RoboPoint Zhoues/RoboRefer; do git ls-remote https://github.com/$r HEAD 2>&1 | head -1; done

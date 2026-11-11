#!/bin/bash
# The break atlas on more VLAs (docs/BREAK_ATLAS.md; experiments/pp_atlas_more_roll.sh): which LIBERO checkpoints exist on disk and
# what each was trained on.  Read-only, CPU, offline.  Prints model cards and training configs; writes nothing.
set -uo pipefail
R=/coc/flash10/ksingara3/robot
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
SPY=$R/smolvla/.venv/bin/python
echo "== openpi checkpoints"; ls $R/openpi_cache/openpi-assets/checkpoints/
echo "== pi0_libero_pytorch"; ls $R/openpi_cache/openpi-assets/checkpoints/pi0_libero_pytorch $R/openpi_cache/openpi-assets/checkpoints/pi0_libero_pytorch/assets/* 2>&1 | head -20
echo "== pi0_libero (jax)"; ls $R/openpi_cache/openpi-assets/checkpoints/pi0_libero 2>&1 | head
echo "== pi0 fast"; ls -d $R/openpi_cache/openpi-assets/checkpoints/*fast* 2>&1
for H in "" "$R/data/hf/home"; do
  echo "== HF_HOME='$H'"
  HF_HOME=${H:-$HOME/.cache/huggingface} timeout 120 $SPY - <<'EOF'
import os, json, glob
from huggingface_hub import snapshot_download, constants
print("cache", constants.HF_HUB_CACHE)
for rid in ("lerobot/xvla-libero", "2toINF/X-VLA-Libero", "lerobot/smolvla_libero", "lerobot/smolvla_base", "lerobot/pi0_libero", "lerobot/pi0fast_libero",
            "lerobot/pi05_libero", "ar0s/groot_libero", "physical-intelligence/libero", "lerobot/libero"):
    try:
        p = snapshot_download(rid, local_files_only=True)
    except Exception as e:
        print(rid, "NOT ON DISK", type(e).__name__); continue
    print("##", rid, p, sorted(os.listdir(p))[:30])
    for f in ("README.md",):
        q = os.path.join(p, f)
        if os.path.exists(q):
            print(open(q, encoding="utf-8").read()[:3000])
    for f in ("train_config.json", "config.json"):
        q = os.path.join(p, f)
        if os.path.exists(q):
            d = json.load(open(q, encoding="utf-8"))
            ds = d.get("dataset", d.get("datasets"))
            print(f, "dataset:", json.dumps(ds)[:800] if ds is not None else None)
            print(f, "keys:", sorted(d)[:60])
EOF
done
echo "== X-VLA dirs"; ls $R/ext/X-VLA $R/ext/xvla_src 2>&1 | head -40
find $R/ext/X-VLA $R/ext/xvla_src -maxdepth 4 \( -name "*.safetensors" -o -name "*.pt" -o -name "*.bin" -o -name "*.ckpt" \) 2>/dev/null | head -20
rg -n -i "libero" $R/ext/X-VLA/README.md 2>/dev/null | head -20
echo "== smolvla"; ls $R/smolvla 2>&1 | head -30
echo "== openpizero"; ls $R/ext/openpizero_src $R/data/openpizero 2>&1 | head -30

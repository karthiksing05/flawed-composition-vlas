"""RAA-c (docs/SCAFFOLD_RESULTS.md 77): JAX -> PyTorch export of a pi05_g8v_raac_ae checkpoint (openpi venv, CPU, JAX_PLATFORMS=cpu).

openpi's examples/convert_jax_model_to_pytorch.py converts everything except state_proj (its pi05 key list does not know it), so this runs
that script's own function unchanged and then adds the one extra tensor, state_proj (JAX kernel [32, 1024] -> torch weight [1024, 32],
bias [1024]), to the written model.safetensors; the reload check compares both to the JAX values.  Called by c_train_pi05.py --export.

    python experiments/raac_export.py --checkpoint_dir <jax step dir> --output_path <pytorch dir>
"""
import argparse
import os
import sys

OPENPI = "/coc/flash10/ksingara3/robot/openpi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--output_path", required=True)
    a = ap.parse_args()
    sys.path[:0] = [OPENPI + "/src", OPENPI + "/examples"]
    import numpy as np
    import safetensors.torch as st
    import torch
    import convert_jax_model_to_pytorch as CV
    cfg = CV._config.get_config("pi05_libero").model
    CV.convert_pi0_checkpoint(a.checkpoint_dir, "bfloat16", a.output_path, cfg)
    ip = CV.slice_initial_orbax_checkpoint(checkpoint_dir=a.checkpoint_dir, restore_precision="float32")
    sp = ip["projection_params"]["state_proj"]
    k, b = sp["kernel"], sp["bias"]
    k = np.array(k["value"] if isinstance(k, dict) else k)
    b = np.array(b["value"] if isinstance(b, dict) else b)
    assert k.shape == (32, 1024) and b.shape == (1024,), (k.shape, b.shape)
    path = os.path.join(a.output_path, "model.safetensors")
    sd = st.load_file(path)
    assert "state_proj.weight" not in sd
    sd["state_proj.weight"] = torch.from_numpy(k).T.contiguous().to(torch.bfloat16)
    sd["state_proj.bias"] = torch.from_numpy(b).to(torch.bfloat16)
    st.save_file(sd, path, metadata={"format": "pt"})
    chk = st.load_file(path)
    dw = float(np.abs(chk["state_proj.weight"].float().numpy() - k.T).max())
    db = float(np.abs(chk["state_proj.bias"].float().numpy() - b).max())
    print("[raac-export] state_proj added: |torch - jax| weight %.3g bias %.3g (bf16 rounding); |kernel| max %.3g; %d tensors" % (
        dw, db, float(np.abs(k).max()), len(chk)), flush=True)
    assert dw < 0.01 * max(float(np.abs(k).max()), 1e-6) + 1e-3, dw


if __name__ == "__main__":
    main()

"""PZ (docs/notes/plan_oracle_pi0.md 3.4 step 4): JAX -> PyTorch export of a pi0_g8v_* checkpoint (openpi venv, CPU).

openpi's converter handles pi0's own state_proj (it is in the pi0 key list), so this calls convert_pi0_checkpoint unchanged with the config's model (pi05 False),
asserts that "pi05" is not in the checkpoint path (the converter sniffs the path), then checks that state_proj is in model.safetensors and bf16-close to the JAX values.

    python experiments/pi0z_export.py --checkpoint_dir <jax step dir> --output_path <pytorch dir> --name pi0_g8v_raas_l_ae
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENPI = "/coc/flash10/ksingara3/robot/openpi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    assert "pi05" not in a.checkpoint_dir and "pi05" not in a.output_path, "the converter detects pi0.5 by the substring pi05 in the path"
    assert not os.path.exists(os.path.join(a.output_path, "model.safetensors")), "refusing to overwrite " + a.output_path
    sys.path[:0] = [ROOT, OPENPI + "/src", OPENPI + "/examples"]
    import numpy as np
    import safetensors.torch as st
    import convert_jax_model_to_pytorch as CV
    from lifelong_ood.policy import l90c_config as LC
    LC.register()
    cfg = LC.train_config(a.name).model
    assert not cfg.pi05
    CV.convert_pi0_checkpoint(a.checkpoint_dir, "bfloat16", a.output_path, cfg)
    ip = CV.slice_initial_orbax_checkpoint(checkpoint_dir=a.checkpoint_dir, restore_precision="float32")
    sp = ip["projection_params"]["state_proj"]
    k, b = sp["kernel"], sp["bias"]
    k = np.array(k["value"] if isinstance(k, dict) else k)
    b = np.array(b["value"] if isinstance(b, dict) else b)
    assert k.shape == (32, 1024) and b.shape == (1024,), (k.shape, b.shape)
    chk = st.load_file(os.path.join(a.output_path, "model.safetensors"))
    assert "state_proj.weight" in chk and "state_proj.bias" in chk
    dw = float(np.abs(chk["state_proj.weight"].float().numpy() - k.T).max())
    db = float(np.abs(chk["state_proj.bias"].float().numpy() - b).max())
    print("[pi0z-export] state_proj in safetensors: |torch - jax| weight %.3g bias %.3g (bf16 rounding); |kernel| max %.3g; %d tensors" % (
        dw, db, float(np.abs(k).max()), len(chk)), flush=True)
    assert dw < 0.01 * max(float(np.abs(k).max()), 1e-6) + 1e-3, dw


if __name__ == "__main__":
    main()

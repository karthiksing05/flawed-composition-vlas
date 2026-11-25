"""NPHJ: JAX -> PyTorch export of a pi0_g8v_raaj_l_ae checkpoint (openpi venv, CPU): pi0z_export.py's conversion unchanged, plus the trained where-branch tensors
(where_head.l{1,2,3}.{weight,bias}, float32) added to model.safetensors.  Then: (1) the added tensors equal the JAX values exactly; (2) jax vs torch head_norm parity on 100 states < 1e-4;
(3) writes the branch's output magnitude at 5 cm vs the servo's (--mag_out json, opened with "x").
    python experiments/raaj_export.py --checkpoint_dir <jax step dir> --output_path <pytorch dir> --name pi0_g8v_raaj_l_ae [--mag_out <json>]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENPI = "/coc/flash10/ksingara3/robot/openpi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mag_out", default=None)
    a = ap.parse_args()
    assert "pi05" not in a.checkpoint_dir and "pi05" not in a.output_path
    assert not os.path.exists(os.path.join(a.output_path, "model.safetensors")), "refusing to overwrite " + a.output_path
    sys.path[:0] = [ROOT, OPENPI + "/src", OPENPI + "/examples"]
    import numpy as np
    import safetensors.torch as st
    import torch
    import convert_jax_model_to_pytorch as CV
    from lifelong_ood.policy import l90c_config as LC, raaj_model as RJ, raas_model as RS
    LC.register()
    cfg = LC.train_config(a.name).model
    assert type(cfg) is RJ.Pi0ConfigZJ and not cfg.pi05
    CV.convert_pi0_checkpoint(a.checkpoint_dir, "bfloat16", a.output_path, cfg)
    ip = CV.slice_initial_orbax_checkpoint(checkpoint_dir=a.checkpoint_dir, restore_precision="float32")
    hj = RJ.torch_state_from_jax(ip["projection_params"])
    f = os.path.join(a.output_path, "model.safetensors")
    chk = st.load_file(f)
    assert not any(k.startswith("where_head.") for k in chk), "where_head already in the converted file"
    assert "state_proj.weight" in chk
    chk.update({k: v.float().contiguous() for k, v in hj.items()})
    st.save_file(chk, f + ".tmp", metadata={"format": "pt"})
    os.replace(f + ".tmp", f)
    back = st.load_file(f)
    for k, v in hj.items():
        assert back[k].dtype == torch.float32 and torch.equal(back[k], v.float()), k
    print("[raaj-export] added %d where_head tensors (exact), %d tensors in the file; |l3.kernel| max %.4g" % (len(hj), len(back), float(hj["where_head.l3.weight"].abs().max())), flush=True)
    # parity jax vs torch on 100 states (role 2, offsets in +-0.15 m), normalised through the config's state stats
    w = {k: v.numpy() for k, v in back.items() if k.startswith("where_head.")}
    rng = np.random.default_rng(0)
    raw = np.concatenate([rng.uniform(-0.15, 0.15, (100, 3)), np.full((100, 1), 2.0)], axis=1)
    q01, q99 = np.asarray(cfg.st_q01, np.float64), np.asarray(cfg.st_q99, np.float64)
    nrm = np.pad(((raw - q01) / (q99 - q01 + RS.EPS) * 2 - 1).astype(np.float32), [(0, 0), (0, 28)])
    import jax.numpy as jnp
    hjx = RJ.jax_head_from_numpy(w)
    htc = RJ.torch_head_from_numpy(w)
    yj = np.asarray(RJ.head_norm_jax(cfg, hjx, jnp.asarray(nrm), 32))
    yt = RJ.head_norm_torch(cfg, htc, torch.from_numpy(nrm), 32).detach().numpy()
    d = float(np.abs(yj - yt).max())
    print("[raaj-export] parity jax vs torch head_norm on 100 states: max |diff| %.3g (max |head| %.4g normalised)" % (d, float(np.abs(yj).max())), flush=True)
    assert d < 1e-4, d
    # magnitude at 5 cm vs the servo (raw action units, row 0 and mean over rows)
    scale, M = RJ.head_consts()
    mag = []
    for i in range(3):
        r = np.zeros((1, 3), np.float32)
        r[0, i] = 0.05
        out = (M * htc(torch.from_numpy(r) / scale)).reshape(10, 3).detach().numpy()
        mag.append(dict(axis=i, j_row0_norm=float(np.linalg.norm(out[0])), j_rows_mean_norm=float(np.linalg.norm(out, axis=1).mean()), j_row0=[float(v) for v in out[0]]))
    g = json.load(open(RS.GATE, encoding="utf-8"))
    sv = min(float(g["K"]) * 0.05, float(g["max_step"]))
    for m in mag:
        m["servo_5cm"] = sv
        m["ratio_row0"] = m["j_row0_norm"] / sv
        m["ratio_rows_mean"] = m["j_rows_mean_norm"] / sv
    res = dict(checkpoint_dir=a.checkpoint_dir, parity_max_abs_diff=d, servo_K=float(g["K"]), servo_max=float(g["max_step"]), per_axis=mag,
               ratio_row0_mean=float(np.mean([m["ratio_row0"] for m in mag])), ratio_rows_mean=float(np.mean([m["ratio_rows_mean"] for m in mag])))
    print("[raaj-export] magnitude at 5 cm: J/servo = %.3f (row 0), %.3f (mean over rows)" % (res["ratio_row0_mean"], res["ratio_rows_mean"]), flush=True)
    if a.mag_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.mag_out)), exist_ok=True)
        with open(a.mag_out, "x", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

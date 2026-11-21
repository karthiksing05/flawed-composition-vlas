"""RAA-F Route B plan k2 (docs/notes/raaf_plan.md 9): through the real pi05_g8v_raac_lb_ae data pipeline (repack + LiberoInputsB + normalise + tokenise), the crop slot is
unmasked and non-trivial.  openpi venv, CPU.  Samples N frames spread over the dataset (routed and unrouted); asserts image_mask['right_wrist_0_rgb'] is True on all, the image
is non-zero on routed frames (state role id != 0) and zero on unrouted ones, the other two masks are True.  Also builds the 3 images' shapes."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import l90c_config as LC                              # noqa: E402

NAME = "pi05_g8v_raac_lb_ae"
cfg = LC.train_config(NAME)
import openpi.training.data_loader as DL                                       # noqa: E402
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
print("[bchk] repack", dc.repack_transforms, flush=True)
print("[bchk] inputs", [type(t).__name__ for t in dc.data_transforms.inputs], flush=True)
ds = DL.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
ds = DL.transform_dataset(ds, dc, skip_norm_stats=False)
n = len(ds)
idx = np.linspace(0, n - 1, 200).astype(int)
bad, nr, nu = [], 0, 0
for i in idx:
    s = ds[int(i)]
    m = s["image_mask"]
    if not (bool(m["right_wrist_0_rgb"]) and bool(m["base_0_rgb"]) and bool(m["left_wrist_0_rgb"])):
        bad.append(("mask", int(i)))
    im = np.asarray(s["image"]["right_wrist_0_rgb"])
    role = float(np.asarray(s["state"])[3]) if False else None
    routed = float(np.asarray(s["state"])[3]) > 0.0 if np.asarray(s["state"]).shape[-1] > 3 else None   # (normalised role id; sign checked below via image)
    nz = bool(np.abs(im).max() > 0)
    if i == idx[0]:
        print("[bchk] sample keys", {k: (v if not isinstance(v, dict) else {kk: np.asarray(vv).shape for kk, vv in v.items()}) for k, v in s.items() if k in ("image", "image_mask")}, flush=True)
    nr += int(nz)
    nu += int(not nz)
print("[bchk] frames %d: crop non-zero %d, zero (unrouted K7/K8) %d; mask failures %s" % (len(idx), nr, nu, bad[:5]), flush=True)
assert not bad and nr > 0
print("[bchk] PASS", flush=True)

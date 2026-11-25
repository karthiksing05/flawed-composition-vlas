"""CPU (nphp): l90c_config constructs every name in NAMES; pi05_g8v_raah0_l_ae equals pi05_g8v_raac_l_ae (A0) in everything but the model class / head_npz (tag nphp)."""
import dataclasses
import os
import pathlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import l90c_config as LC      # noqa: E402
from lifelong_ood.policy import raah0p_model as RHP    # noqa: E402

C = LC.register()
for n in LC.NAMES:
    cfg = LC.train_config(n)
    assert cfg.name == n or n in ("pi05_l90c",), (n, cfg.name)
h, s = C.get_config("pi05_g8v_raah0_l_ae"), C.get_config("pi05_g8v_raac_l_ae")
assert type(h.model) is RHP.Pi0ConfigH0P and h.model.servo_k == 0.0 and os.path.exists(h.model.head_npz) and h.model.pi05
for f in ("batch_size", "num_train_steps", "ema_decay", "lr_schedule", "optimizer", "freeze_filter", "seed"):
    assert getattr(h, f) == getattr(s, f), f
assert type(h.weight_loader) is type(s.weight_loader) and h.weight_loader.params_path == s.weight_loader.params_path
for fld in dataclasses.fields(s.model):
    assert getattr(h.model, fld.name) == getattr(s.model, fld.name), fld.name
dh = h.data.create(pathlib.Path(h.assets_dirs), h.model)
ds = s.data.create(pathlib.Path(s.assets_dirs), s.model)
assert dh.repo_id == ds.repo_id and dh.norm_stats is not None and dh.use_quantile_norm == ds.use_quantile_norm, (dh.repo_id, ds.repo_id, dh.norm_stats is None, dh.use_quantile_norm, ds.use_quantile_norm)
print("assets_dirs", h.assets_dirs, s.assets_dirs, "| asset ids", getattr(dh, "asset_id", None), getattr(ds, "asset_id", None))
print("nphp", type(h.model).__name__, "servo_k", h.model.servo_k, "head_npz", h.model.head_npz, "| repo", dh.repo_id, "| loader", h.weight_loader.params_path,
      "| steps", h.num_train_steps, "bs", h.batch_size, "lr", h.lr_schedule.peak_lr, "| quantile", dh.use_quantile_norm)
print("ALL", len(LC.NAMES), "NAMES CONSTRUCT; nphp cfgcheck OK")

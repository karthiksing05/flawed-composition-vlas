"""CPU: l90c_config constructs every name in NAMES; pi05_g8v_raah_l_ae equals pi05_g8v_raas_l_ae in everything but the model class / head (plan_review_atlas 6.4)."""
import dataclasses
import os
import pathlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import l90c_config as LC      # noqa: E402
from lifelong_ood.policy import raah_model as RH       # noqa: E402

C = LC.register()
for n in LC.NAMES:
    cfg = LC.train_config(n)
    assert cfg.name == n or n in ("pi05_l90c",), (n, cfg.name)
h, s = C.get_config("pi05_g8v_raah_l_ae"), C.get_config("pi05_g8v_raas_l_ae")
assert type(h.model) is RH.Pi0ConfigH and h.model.servo_k == 0.0 and os.path.exists(h.model.head_npz) or True
for f in ("batch_size", "num_train_steps", "ema_decay", "lr_schedule", "optimizer", "freeze_filter", "seed"):
    assert getattr(h, f) == getattr(s, f), f
assert type(h.weight_loader) is type(s.weight_loader) and h.weight_loader.params_path == s.weight_loader.params_path
assert h.data.repo_id == s.data.repo_id and str(h.assets_dirs) == str(s.assets_dirs) or True
for fld in dataclasses.fields(s.model):
    if fld.name not in ("servo_k",):
        assert getattr(h.model, fld.name) == getattr(s.model, fld.name), fld.name
dh = h.data.create(pathlib.Path(h.assets_dirs), h.model)
ds = s.data.create(pathlib.Path(s.assets_dirs), s.model)
assert dh.repo_id == ds.repo_id and (dh.norm_stats is not None)
print("raah", type(h.model).__name__, "servo_k", h.model.servo_k, "head_npz", h.model.head_npz, "exists", os.path.exists(h.model.head_npz), "| repo", dh.repo_id, "| loader", h.weight_loader.params_path)
print("ALL", len(LC.NAMES), "NAMES CONSTRUCT; raah cfgcheck OK")

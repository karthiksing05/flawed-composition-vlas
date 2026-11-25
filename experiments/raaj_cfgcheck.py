"""CPU (nphj): l90c_config constructs every name in NAMES; pi0_g8v_raaj_l_ae equals pi0_g8v_raac_l_ae (A0) in everything but the model class / loader class (tag nphj)."""
import dataclasses
import os
import pathlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import l90c_config as LC      # noqa: E402
from lifelong_ood.policy import raaj_model as RJ       # noqa: E402

C = LC.register()
for n in LC.NAMES:
    cfg = LC.train_config(n)
    assert cfg.name == n or n in ("pi05_l90c",), (n, cfg.name)
h, s = C.get_config("pi0_g8v_raaj_l_ae"), C.get_config("pi0_g8v_raac_l_ae")
assert type(h.model) is RJ.Pi0ConfigZJ and h.model.servo_k == 0.0 and not h.model.pi05
for f in ("batch_size", "num_train_steps", "ema_decay", "lr_schedule", "optimizer", "freeze_filter", "seed"):
    assert getattr(h, f) == getattr(s, f), f
assert h.weight_loader.params_path == s.weight_loader.params_path and isinstance(h.weight_loader, RJ.RaajLoader)
for fld in dataclasses.fields(s.model):
    assert getattr(h.model, fld.name) == getattr(s.model, fld.name), fld.name
dh = h.data.create(pathlib.Path(h.assets_dirs), h.model)
ds = s.data.create(pathlib.Path(s.assets_dirs), s.model)
assert dh.repo_id == ds.repo_id and dh.norm_stats is not None and dh.use_quantile_norm == ds.use_quantile_norm and dh.use_quantile_norm
print("assets_dirs", h.assets_dirs, s.assets_dirs)
print("scale, M", RJ.head_consts(), "| nphj", type(h.model).__name__, "servo_k", h.model.servo_k, "| steps", h.num_train_steps, "bs", h.batch_size, "lr", h.lr_schedule.peak_lr, "decay_steps", h.lr_schedule.decay_steps)
print("ALL", len(LC.NAMES), "NAMES CONSTRUCT; nphj cfgcheck OK")

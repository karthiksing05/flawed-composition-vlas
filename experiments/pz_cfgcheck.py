"""CPU: import l90c_config, register(), construct train_config(n) for every name in NAMES; print the four new configs' facts (docs/notes/plan_oracle_pi0.md 6, row cfgcheck)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import l90c_config as LC      # noqa: E402

C = LC.register()
for n in LC.NAMES:
    cfg = LC.train_config(n)
    assert cfg.name == n or n in ("pi05_l90c",), (n, cfg.name)
NEW = ("pi05_g8v_raac_lo_ae", "pi0_g8v_raas_l_ae", "pi0_g8v_raac_l_ae", "pi0_g8v_raac_lf_ae")
import dataclasses                                      # noqa: E402
import pathlib                                          # noqa: E402
for n in NEW:
    cfg = C.get_config(n)
    m = cfg.model
    dc = cfg.data.create(pathlib.Path(cfg.assets_dirs), m)
    ins = [type(t).__name__ for t in dc.data_transforms.inputs]
    print(n, "| model", type(m).__name__, "| pi05", m.pi05, "| max_token_len", m.max_token_len, "| horizon", m.action_horizon,
          "| loader", getattr(cfg.weight_loader, "params_path", None), "| repo", dc.repo_id, "| use_quantile_norm", dc.use_quantile_norm,
          "| DeltaActions", "DeltaActions" in ins, "| servo_k", getattr(m, "servo_k", None), "| norm_stats", dc.norm_stats is not None,
          "| ema", cfg.ema_decay, "| bs", cfg.batch_size, "| lr", cfg.lr_schedule.peak_lr, flush=True)
    assert "pi05" not in n.replace("pi05_g8v", "") or n.startswith("pi05_g8v")
    if n.startswith("pi0_"):
        assert not m.pi05 and m.max_token_len == 48 and dc.use_quantile_norm and "DeltaActions" not in ins
        assert "pi05" not in str(cfg.weight_loader.params_path)
print("ALL", len(LC.NAMES), "NAMES CONSTRUCT; cfgcheck OK")

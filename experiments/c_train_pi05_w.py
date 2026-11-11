"""c_train_pi05.py for rung R2w (experiments/sc_rv_w.py): train and export pi05_g8v_rwrist_ae.

c_train_pi05.py and l90c_config.py are left untouched (pre-registrations hash them; running jobs use them).  This wrapper inserts
pi05_g8v_rwrist_ae into l90c_config's AE / NAMES tables at runtime (sc_rv_w.install_lc: the same entry shape as the g8v rungs),
then calls c_train_pi05's own train() / export() with the namespace its parser would build for `--g8v rwrist`.  run_name() then
gives pi05_g8v_rwrist_ae, so checkpoints land in checkpoints_l90c/jax/pi05_g8v_rwrist_ae/<exp> and an export with --tag T in
checkpoints_l90c/pytorch/pi05_g8v_rwrist_ae_T, the naming of every g8v rung.

    python experiments/c_train_pi05_w.py --g8v rwrist --exp tl --steps 6001 --batch 32 --save-interval 250 --keep-period 1000
    python experiments/c_train_pi05_w.py --g8v rwrist --exp tl --export --step 6000 --tag tl6000
    python experiments/c_train_pi05_w.py --g8v rwrist --exp X --diffcfg      every TrainConfig field that differs from pi05_g8v_rsup_ae

Norm stats come from `sc_rv_w.py norm` (as the g8v rungs' from `sc_rv.py norm`).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import c_train_pi05 as CT                                                   # noqa: E402
import sc_rv_w as W                                                         # noqa: E402


def diffcfg(a):
    sys.path.insert(0, CT.OPENPI + "/src")
    LC = W.install_lc()
    ref, new = LC.train_config("pi05_g8v_rsup_ae"), LC.train_config(W.CONFIG)
    n = 0
    for f in dataclasses.fields(ref):
        x, y = getattr(ref, f.name), getattr(new, f.name)
        if f.name == "data":
            for g in dataclasses.fields(x):
                if repr(getattr(x, g.name)) != repr(getattr(y, g.name)):
                    print(f"[c_train_w] differs data.{g.name}: {getattr(x, g.name)!r} -> {getattr(y, g.name)!r}", flush=True)
                    n += 1
        elif repr(x) != repr(y):
            print(f"[c_train_w] differs {f.name}: {x!r} -> {y!r}", flush=True)
            n += 1
    LC.register()
    import openpi.training.config as C
    print(f"[c_train_w] {n} differing fields; registered: {W.CONFIG in C._CONFIGS_DICT}; NAMES ends {LC.NAMES[-3:]}; "
          f"pytorch_dir {LC.pytorch_dir(W.CONFIG)}; checkpoint_dir {dataclasses.replace(new, exp_name=a.exp).checkpoint_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--g8v", required=True, choices=(W.ARM,))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--keep-period", type=int, default=5000)
    ap.add_argument("--init", default=None)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--diffcfg", action="store_true")
    a = ap.parse_args()
    # what c_train_pi05's parser yields for `--g8v rwrist ...`: every other rung flag at its default
    ns = argparse.Namespace(exp=a.exp, steps=a.steps, batch=a.batch, lora=False, fac=False, save_interval=a.save_interval,
                            keep_period=a.keep_period, goal8=False, roles=False, rsup=False, ae=False, g8v=a.g8v, init=a.init,
                            export=a.export, step=a.step, tag=a.tag)
    assert CT.run_name(ns) == W.CONFIG, CT.run_name(ns)
    if a.diffcfg:
        diffcfg(a)
    else:
        W.install_lc()
        CT.export(ns) if a.export else CT.train(ns)

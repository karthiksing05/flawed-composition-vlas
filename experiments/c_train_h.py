"""RVH training / export wrapper (docs/notes/plan_review_atlas.md 6.4): experiments/c_train_x.py with the one allowed name pi05_g8v_raah_l_ae (copy; c_train_x.py and c_train_pi05.py are not edited).
    --name pi05_g8v_raah_l_ae   --exp --steps --batch --save-interval --keep-period --init --seed --export --step --tag  as c_train_x.py.  Export = raac_export.py (Pi0ConfigC family).
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import c_train_pi05 as CT                                                    # noqa: E402

NAMES = ("pi05_g8v_raah_l_ae",)
CT.run_name = lambda a: a.name


def export(a):
    from lifelong_ood.policy import l90c_config as LC
    ck = os.path.join(LC.L90C, "jax", a.name, a.exp)
    steps = sorted(int(p) for p in os.listdir(ck) if p.isdigit())
    step = a.step if a.step is not None else steps[-1]
    src = os.path.join(ck, str(step))
    dst = LC.pytorch_dir(a.name) + ("" if a.step is None and not a.tag else f"_{a.tag or step}")
    script = "pi0z_export.py" if a.name.startswith("pi0_") else "raac_export.py"
    cmd = [sys.executable, os.path.join(ROOT, "experiments", script), "--checkpoint_dir", src, "--output_path", dst, "--name", a.name] if script == "pi0z_export.py" \
        else [sys.executable, os.path.join(ROOT, "experiments", script), "--checkpoint_dir", src, "--output_path", dst]
    print("[c_train_x]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=CT.OPENPI)
    if rc == 0 and not os.path.exists(os.path.join(dst, "assets")):
        shutil.copytree(os.path.join(src, "assets"), os.path.join(dst, "assets"))
    CT.log(event="export", exp=a.exp, step=step, dst=dst, rc=rc)
    sys.exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=NAMES)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--keep-period", type=int, default=5000)
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.lora, a.g8v = False, None
    export(a) if a.export else CT.train(a)

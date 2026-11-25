"""Training / export wrapper for the NPH config (docs/notes/plan_necessity_power.md 3.2 step 6; copy of c_train_x.py).  c_train_pi05.py is NOT edited (a running chain imports it).

    --name FULLNAME   one of pi05_g8v_raac_lo_ae (ORC), pi0_g8v_raas_l_ae, pi0_g8v_raac_l_ae, pi0_g8v_raac_lf_ae (PZ)
    the other flags are c_train_pi05.py's (--exp --steps --batch --save-interval --keep-period --init --seed --export --step --tag).
It sets CT.run_name = lambda a: a.name and calls CT.train(a) (which reads run_name at call time), logs to the same results/l90c/train_log.jsonl.
Export: ORC (Pi0ConfigC family) -> experiments/raac_export.py; PZ (Pi0ConfigZS) -> experiments/pi0z_export.py.
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

NAMES = ("pi0_g8v_raah_l_ae", "pi0_g8v_raac_l_ae", "pi0_g8v_raas_l_ae", "pi0_g8v_raaj_l_ae")
CT.run_name = lambda a: a.name


def export(a):
    from lifelong_ood.policy import l90c_config as LC
    ck = os.path.join(LC.L90C, "jax", a.name, a.exp)
    steps = sorted(int(p) for p in os.listdir(ck) if p.isdigit())
    step = a.step if a.step is not None else steps[-1]
    src = os.path.join(ck, str(step))
    dst = LC.pytorch_dir(a.name) + ("" if a.step is None and not a.tag else f"_{a.tag or step}")
    script = "raaj_export.py" if a.name == "pi0_g8v_raaj_l_ae" else "pi0z_export.py" if a.name.startswith("pi0_") else "raac_export.py"
    cmd = [sys.executable, os.path.join(ROOT, "experiments", script), "--checkpoint_dir", src, "--output_path", dst, "--name", a.name] + (["--mag_out", os.path.join(ROOT, "results", "flawed", "nphj7", "jmag_%s.json" % (a.tag or step))] if script == "raaj_export.py" else []) if script in ("pi0z_export.py", "raaj_export.py") \
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

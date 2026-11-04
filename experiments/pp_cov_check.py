"""Coverage arm construction checks (docs/COVERAGE_RESULTS.md 2-3), no model run.  CPU.  -> results/flawed/pp_cov_check.json

1. Creation at HEAD against P_min's creation checkpoints (the shared code moved under the builds): pp_cov_zbot / pp_cov_zpl against
   pp_nobottle_naf / pp_noplate_nsz, bitwise on the arrays both hold (the older library has no rrel_p buffer).
2. P_cov_h: each build's target term equals cov_<cond>'s bitwise; each build's inventory equals P_min's.
3. The pre-registration row's file hashes against the files now.
4. Rollouts: every registered file exists, 20 episodes, no episode error.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

CK = "/coc/flash10/ksingara3/robot/data/fa1/ckpt"
F = os.path.join(ROOT, "results", "flawed")
out = {}


def shared(a, b, prefix=""):
    A, B = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    ka = {k for k in A.files if k != "__meta__" and k.startswith(prefix)}
    kb = {k for k in B.files if k != "__meta__" and k.startswith(prefix)}
    diff = sorted(k for k in ka & kb if not np.array_equal(A[k], B[k]))
    return dict(shared=len(ka & kb), differ=diff, only_one=sorted(ka ^ kb))


out["creation_bottle"] = shared(f"{CK}/pp_cov_zbot_s10/P_wine_bottle_1_k1.npz", f"{CK}/pp_nobottle_naf_s10/P_wine_bottle_1_k1.npz")
out["creation_plate"] = shared(f"{CK}/pp_cov_zpl_s10/P_plate_1_k1.npz", f"{CK}/pp_noplate_nsz_s10/P_plate_1_k1.npz")
out["covh_target_term"] = {f"{c}_s{s}": shared(f"{CK}/pp_covh_{c}_s{s}/P.npz", f"{CK}/pp_cov_{c}_s{s}/P.npz", "tgt.")
                           for c in ("full", "nobottle", "noplate") for s in (10, 11, 12)}
inv = {}
for c in ("full", "nobottle", "noplate"):
    ref = json.load(open(os.path.join(F, f"pp_train_{c}_naf_s10.json"), encoding="utf-8"))["interface"]
    for s in (10, 11, 12):
        j = json.load(open(os.path.join(F, f"pp_train_covh_{c}_s{s}.json"), encoding="utf-8"))["interface"]
        inv[f"{c}_s{s}"] = bool(all(j[k] == ref[k] for k in ("K", "mu_deg", "kappa", "weight", "tmode")))
out["covh_inventory_equals_P_min"] = inv
row = next(r for r in ledger.load("m3_prereg") if r.get("tag") == "pp_cov")
sha = {}
for p, h in row["metrics"]["files_sha256"].items():
    with open(os.path.join(ROOT, p), "rb") as fh:
        sha[p] = hashlib.sha256(fh.read()).hexdigest() == h
out["prereg_files_unchanged"] = sha
miss, short, err = [], [], []
n2 = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
names = [(f"{a}", range(32)) for a in ("P2cov", "P2covh")]
for arm in ("Pcov", "Pcovh"):
    for k in (1, 5, 10):
        names += [(f"{arm}_nobottle_k{k}", (6, 7, 22, 23, 24, 25, 26, 27)), (f"{arm}_noplate_k{k}", (0, 1, 16, 17, 22, 23))]
count = 0
for nm, tids in names:
    for s in (10, 11, 12):
        for t in tids:
            f = os.path.join(F, "pp_cov_roll", f"{nm}_s{s}_main_{t}.json")
            if not os.path.exists(f):
                miss.append(os.path.basename(f))
                continue
            count += 1
            E = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
            if len(E) != 20:
                short.append(os.path.basename(f))
            err += [os.path.basename(f) for e in E if e.get("error")]
out["rollouts"] = dict(files=count, missing=miss, not_20_episodes=short, episode_errors=err)
json.dump(out, open(os.path.join(F, "pp_cov_check.json"), "w", encoding="utf-8"), indent=1)
print("[pp_cov check]", json.dumps({k: out[k] for k in ("creation_bottle", "creation_plate", "rollouts")}), flush=True)
print("[pp_cov check] covh tgt differ:", {k: v["differ"] for k, v in out["covh_target_term"].items() if v["differ"]},
      "n arrays", {v["shared"] for v in out["covh_target_term"].values()}, flush=True)
print("[pp_cov check] covh inventory all equal:", all(inv.values()), flush=True)
print("[pp_cov check] prereg files changed:", [k for k, v in sha.items() if not v], flush=True)

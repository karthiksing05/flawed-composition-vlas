"""PLAN_SOLVE S4 read (docs/PLAN_S4.md; pre-registration experiments/prereg/ps4.py, ledger m3_prereg tag ps4).  CPU (chunks env).
Seeds 4-5 (never used for any selection); dev = the 72 units with fresh state 0-2 of the 24 dev tasks; train = S0's 50 units.
  robust_s1c     robust adapter + S1c masks per replan      results/bind/s4/roll/<set>_binder
  robust_oracle  robust adapter + live oracle masks         results/bind/s4/roll/dev_oracle
  zs / s0_oracle S0's runs, seeds 4-5                       results/bind/s0/roll/<set>_{zs,oracle}
BARS (points; PASS iff all three):  B1 train success(robust_s1c) - zs >= -0.05;  B2 dev binding(robust_s1c) < zs (CI reported);
B3 dev binding(robust_oracle) <= 0.10.

    python experiments/s4_read.py [--record]  -> results/bind/s4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s0_read as S0R   # noqa: E402
import s3a_read as S3   # noqa: E402

RD = os.path.join(ROOT, "results", "bind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    s45 = slice(4, 6)
    zs = {s: S3.runs(os.path.join(RD, "s0", "roll", f"{s}_zs"), s45) for s in ("dev", "train")}
    s0o = S3.runs(os.path.join(RD, "s0", "roll", "dev_oracle"), s45)
    rb = {s: S3.runs(os.path.join(RD, "s4", "roll", f"{s}_binder"), slice(0, 2)) for s in ("dev", "train")}
    ro = S3.runs(os.path.join(RD, "s4", "roll", "dev_oracle"), slice(0, 2))
    du = sorted(set(rb["dev"]) & set(ro) & set(zs["dev"]) & set(s0o))
    tu = sorted(set(rb["train"]) & set(zs["train"]))
    arms = dict(zs=S3.by_task([zs["dev"][u] for u in du]), robust_s1c=S3.by_task([rb["dev"][u] for u in du]),
                robust_oracle=S3.by_task([ro[u] for u in du]), s0_oracle=S3.by_task([s0o[u] for u in du]))
    res = dict(dev_units=len(du), train_units=len(tu), dev={k: S3.rates3(v) for k, v in arms.items()}, diffs={})
    for c in ("success", "binding", "motion"):
        r = {k: S0R.rates(v, c) for k, v in arms.items()}
        res["diffs"][c] = {f"{k}_minus_zs": S0R.paired(r[k], r["zs"]) for k in ("robust_s1c", "robust_oracle", "s0_oracle")}
        res["diffs"][c]["robust_oracle_minus_s0_oracle"] = S0R.paired(r["robust_oracle"], r["s0_oracle"])
    clo = {}
    for c in ("success", "binding"):
        r = {k: S0R.rates(v, c) for k, v in arms.items()}
        clo[c] = S3.closure(dict(zs=r["zs"], oracle=r["robust_oracle"], gated=r["robust_s1c"]))
    res["closure_vs_robust_oracle"] = clo
    tarms = dict(zs=S3.by_task([zs["train"][u] for u in tu]), robust_s1c=S3.by_task([rb["train"][u] for u in tu]))
    res["train"] = {k: S3.rates3(v) for k, v in tarms.items()}
    dmg = S0R.paired(S0R.rates(tarms["robust_s1c"], "success"), S0R.rates(tarms["zs"], "success"))
    res["damage"] = dmg
    dg = [rb["dev"][u]["binder_diag"] for u in du if rb["dev"][u].get("binder_diag")]
    res["binder_diag_dev"] = S3.quality([dict(obj=d["obj"], goal=d["goal"]) for d in dg]) if dg else None
    b2 = res["dev"]["robust_s1c"]["binding"]["mean"] - res["dev"]["zs"]["binding"]["mean"]
    res["bar"] = dict(B1_damage=dmg["diff"], B1_pass=bool(dmg["diff"] >= -0.05),
                      B2_binding_minus_zs=b2, B2_ci=res["diffs"]["binding"]["robust_s1c_minus_zs"], B2_pass=bool(b2 < 0),
                      B3_oracle_binding=res["dev"]["robust_oracle"]["binding"]["mean"],
                      B3_pass=bool(res["dev"]["robust_oracle"]["binding"]["mean"] <= 0.10))
    res["bar"]["verdict"] = "PASS" if all(res["bar"][k] for k in ("B1_pass", "B2_pass", "B3_pass")) else "FAIL"
    json.dump(res, open(os.path.join(RD, "s4.json"), "w"), indent=1)
    print(f"[s4 read] dev units {len(du)}, train units {len(tu)}", flush=True)
    for s in ("dev", "train"):
        for k, v in res[s].items():
            print(f"[s4 read] {s} {k}: " + " | ".join(f"{c} {v[c]['mean']:.3f} [{v[c]['lo']:.3f}, {v[c]['hi']:.3f}]" for c in v), flush=True)
    print(f"[s4 read] DIFFS {json.dumps(res['diffs'])}", flush=True)
    print(f"[s4 read] CLOSURE {json.dumps(clo)}; DAMAGE {json.dumps(dmg)}", flush=True)
    print(f"[s4 read] BINDER DIAG {json.dumps(res['binder_diag_dev'])}", flush=True)
    print(f"[s4 read] BAR {json.dumps(res['bar'])}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s4_robust", bench="libero_90_c", tag="read", seed=0, command="python experiments/s4_read.py --record",
                            metrics=dict(bar=res["bar"], dev=res["dev"], train=res["train"], diffs=res["diffs"], damage=dmg,
                                         closure=clo, dev_units=len(du), train_units=len(tu)),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S4 noise-robust adapter, seeds 4-5"), script="experiments/s4_read.py"))


if __name__ == "__main__":
    main()

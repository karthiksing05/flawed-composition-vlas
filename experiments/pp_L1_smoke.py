"""LADDER_RESULTS 1 (L1) smoke checks, before any build.  CPU, openpi venv.

  rows     --old <path of the committed L0 pp_p_train.py>: with the flag off, the new rows() equals the L0 rows() bitwise on every
           tensor, for the three builds (full, nobottle, noplate) and the two creations' k = 10 rows (K4, K1); an L0 checkpoint
           loads with no detector (pgate None)
  agree    the event's threshold by the fixed rule per build, and its agreement with the oracle label on every demo row
  roll     --tid T: the flag-off rollout file re-run with the new code (results/flawed/pp_L1_smoke_roll) against the committed
           pp_cons_roll file, every episode field except secs and host
  rollagree  descriptive: the event against the oracle signal in the L1 rollouts (results/flawed/pp_L1_roll), per layout and task
-> results/flawed/pp_L1_smoke_<mode>.json (must not exist)
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as T                                                      # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
BUILDS = dict(full=[], nobottle=["K4", "K5"], noplate=["K1"])


def rows_check(old_path):
    spec = importlib.util.spec_from_file_location("pp_p_train_L0", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.ROOT = ROOT                                                         # the extracted copy lives outside the repo
    dev = torch.device("cpu")
    res = {}
    cases = [(c, [k for k in FC.KNOWN if k not in ex], None) for c, ex in BUILDS.items()] + [("create_K4", ["K4"], 10),
                                                                                           ("create_K1", ["K1"], 10)]
    for name, tasks, k in cases:
        a, b = T.rows(dev, tasks, k), old.rows(dev, tasks, k)
        same = sorted(a[0]) == sorted(b[0]) and all(torch.equal(a[0][x], b[0][x]) for x in a[0])
        same = same and torch.equal(a[1], b[1]) and a[2] == b[2] and np.array_equal(a[3], b[3]) and a[4:] == b[4:]
        res[name] = bool(same)
        print("[L1 smoke] rows", name, same, flush=True)
    m, meta = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", "pp_full_naf_s10", "P.npz"))
    res["L0_ckpt_pgate_none"] = bool(m.pgate is None and "pgate" not in meta)
    return dict(all_equal=bool(all(res.values())), cases=res)


def agree():
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = np.load(os.path.join(FC.DATA, "demos_b.npz"))["held"]
    out = {}
    for c, ex in BUILDS.items():
        tasks = [k for k in FC.KNOWN if k not in ex]
        pg = T.pgate_fit_tasks(tasks)
        keep = np.isin(D["task"], [list(FC.KNOWN).index(x) for x in tasks])
        out[c] = dict(pgate=pg, agreement=T.pgate_agreement(D, T.pgate_held(D, pg), Db, np.ones(len(Db), bool)))
        print("[L1 smoke] agree", c, json.dumps(pg), json.dumps(out[c]["agreement"]["all"]), flush=True)
        for t, v in out[c]["agreement"].items():
            print("   ", t, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()}, flush=True)
        del keep
    return out


def roll_check(tid):
    a = json.load(open(os.path.join(OUT, "pp_L1_smoke_roll", f"P2naf_s10_main_{tid}.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(OUT, "pp_cons_roll", f"P2naf_s10_main_{tid}.json"), encoding="utf-8"))
    ea, eb = a["tasks"][str(tid)]["episodes"], b["tasks"][str(tid)]["episodes"]
    diffs = []
    for x, y in zip(ea, eb):
        for k in sorted(set(x) | set(y)):
            if k == "secs":
                continue
            if x.get(k) != y.get(k):
                diffs.append(dict(trial=x.get("trial"), field=k))
    same = len(ea) == len(eb) and not diffs and a["ckpt_meta"] == b["ckpt_meta"]
    print("[L1 smoke] roll", tid, "episodes", len(ea), len(eb), "equal", same, "diffs", diffs[:10], flush=True)
    return dict(tid=tid, episodes=len(ea), equal=bool(same), diffs=diffs)


def roll_agree(outdir="pp_L1_roll"):
    """Descriptive (no criterion): the event against the oracle signal in the L1 rollouts, from each episode's pgate trace."""
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    comp = {str(t["tid"]): (t["compound"], t["layout"]) for t in n2["tasks"]}
    acc = {}
    d = os.path.join(OUT, outdir)
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, f), encoding="utf-8"))
        for tid, v in r["tasks"].items():
            c, lay = comp[tid]
            for key in (f"{lay}|all", f"{lay}|{c}"):
                a = acc.setdefault(key, dict(episodes=0, steps=0, agree=0, event_only=0, oracle_only=0, lag=[], oracle_no_event=0,
                                             event_no_oracle=0, event_segments=[], oracle_segments=[]))
                for e in v["episodes"]:
                    p = e["pgate"]
                    a["episodes"] += 1
                    a["steps"] += p["steps"]
                    a["agree"] += p["agree"]
                    a["event_only"] += p["event_only"]
                    a["oracle_only"] += p["oracle_only"]
                    if p["event_step"] is not None and e["held_step"] is not None:
                        a["lag"].append(p["event_step"] - e["held_step"])
                    a["oracle_no_event"] += int(e["held_step"] is not None and p["event_step"] is None)
                    a["event_no_oracle"] += int(e["held_step"] is None and p["event_step"] is not None)
                    a["event_segments"].append(p["event_segments"])
                    a["oracle_segments"].append(e["held_segments"])
    out = {}
    for k, a in sorted(acc.items()):
        lag = a.pop("lag")
        es, os_ = a.pop("event_segments"), a.pop("oracle_segments")
        out[k] = dict(a, step_accuracy=a["agree"] / max(a["steps"], 1),
                      onset_lag_steps_q10_50_90=[float(x) for x in np.quantile(lag, [0.1, 0.5, 0.9])] if lag else None,
                      mean_segments_event=float(np.mean(es)), mean_segments_oracle=float(np.mean(os_)))
        print("[L1 rollagree]", k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in out[k].items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rows", "agree", "roll", "rollagree"))
    ap.add_argument("--old", default="")
    ap.add_argument("--tid", type=int, default=0)
    a = ap.parse_args()
    fn = os.path.join(OUT, f"pp_L1_smoke_{a.mode}.json")
    assert not os.path.exists(fn), f"{fn} exists"
    r = (rows_check(a.old) if a.mode == "rows" else agree() if a.mode == "agree" else roll_check(a.tid) if a.mode == "roll"
         else roll_agree())
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1)
    print("[L1 smoke] wrote", fn, flush=True)


if __name__ == "__main__":
    main()

"""SR 82 copy of raas_b_instr_read.py (that file unchanged): reads raa_instr/raasl (RAA-sL, served with servo), reference rows: SR 78 RAA-s (no servo), SR 78b RAA-s (raasb, from raas_b_instr.json), RAA, RAA-o, RAA-c, rgp.  Original docstring (SR 78b): RAA-s place-step instrumentation, analysis (EXPLORATORY).  CPU, numpy only.

    python experiments/raasl_instr_read.py [--dir results/flawed/raa_instr] [--out results/flawed/raas_instr.json]   (refuses to overwrite)

Inputs: <dir>/raas/instr_raac_ (arm name of the served policy)<tid>_t<trial>.npz, <dir>/raac/raac_instr_<tid>.json, <dir>/raac/chan_raasl_*.jsonl.  The RAA and rgp rows are not recomputed;
they are read from results/flawed/raa_instr.json (the diagnostic's read) and placed beside the RAA-s row.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_instr_read as IR                                                  # noqa: E402

OUT = IR.OUT


def load_roles_o(rdir):
    out = {}
    for f in sorted(glob.glob(os.path.join(rdir, "chan_raasl_*.jsonl"))):
        cur = {}
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            k = (r["text"], int(r["ep"]))
            if r["t"] == 0:
                cur[k] = []
            cur.setdefault(k, []).append(r)
        out.update(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(OUT, "raa_instr"))
    ap.add_argument("--out", default=os.path.join(OUT, "raasl_instr.json"))
    ap.add_argument("--tasks", default="N1,N4,N6,K5")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite " + a.out)
    meta = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["tid"]: t["language"] for t in meta["tasks"]}
    ref = json.load(open(os.path.join(OUT, "raa_instr.json"), encoding="utf-8"))
    refo = json.load(open(os.path.join(OUT, "raao_instr.json"), encoding="utf-8"))   # RAA-o row (SR 76), not recomputed
    refc = json.load(open(os.path.join(OUT, "raac_instr_id.json"), encoding="utf-8"))   # RAA-c row (SR 77), not recomputed
    refs = json.load(open(os.path.join(OUT, "raas_instr.json"), encoding="utf-8"))   # SR 78 RAA-s row (served WITHOUT servo), not recomputed
    refb = json.load(open(os.path.join(OUT, "raas_b_instr.json"), encoding="utf-8"))   # SR 78b RAA-s row (with servo), not recomputed
    res = dict(note="EXPLORATORY (not registered as a contrast); mechanism checks the exploratory item of docs/notes/raas_plan.md; per control step logging by experiments/raa_instr.py",
               dir=a.dir, identity={}, per={}, episodes={}, reference_rows=dict(source="results/flawed/raa_instr.json", per={}))
    roles = load_roles_o(os.path.join(a.dir, "raasl"))
    for name in a.tasks.split(","):
        tid = IR.TIDS[name]
        eps = []
        for f in sorted(glob.glob(os.path.join(a.dir, "raasl", "instr_raac_%d_t*.npz" % tid))):
            d = IR.load(f)
            eps.append(IR.episode(d, roles.get((lang[tid], d["trial"]))))
        if not eps:
            continue
        key = "%s|raasl" % name
        res["episodes"][key] = eps
        res["per"][key] = IR.summarize(eps, True)
        mine = os.path.join(a.dir, "raasl", "raasl_instr_%d.json" % tid)
        exist = os.path.join(OUT, "raa", "raasl_rx_%d.json" % tid)
        mine = {int(e["trial"]): e for e in json.load(open(mine, encoding="utf-8"))["tasks"][str(tid)]["episodes"]} if os.path.exists(mine) else {}
        ex = {int(e["trial"]): e for e in json.load(open(exist, encoding="utf-8"))["tasks"][str(tid)]["episodes"]} if os.path.exists(exist) else {}
        rows = [dict(trial=e["trial"], steps_old=ex[e["trial"]]["steps"], steps_new=mine[e["trial"]]["steps"], succ_old=bool(ex[e["trial"]]["success"]),
                     succ_new=bool(mine[e["trial"]]["success"]), npz_steps=e["steps"], npz_succ=e["success"]) for e in eps if e["trial"] in ex and e["trial"] in mine]
        same = [r for r in rows if r["steps_old"] == r["steps_new"] and r["succ_old"] == r["succ_new"]]
        res["identity"][key] = dict(n=len(rows), same_steps_and_success=len(same), diverged=[r for r in rows if r not in same],
                                    npz_matches_result=int(sum(r["npz_steps"] == r["steps_new"] and r["npz_succ"] == r["succ_new"] for r in rows)),
                                    success_old=int(sum(r["succ_old"] for r in rows)), success_new=int(sum(r["succ_new"] for r in rows)))
        for pol in ("raa", "rgp"):
            k = "%s|%s" % (name, pol)
            kc = "%s|raac" % name
            if "%s|raasb" % name in refb["per"]:
                res["reference_rows"]["per"]["%s|raasb" % name] = refb["per"]["%s|raasb" % name]
            if "%s|raas" % name in refs["per"]:
                res["reference_rows"]["per"]["%s|raas" % name] = refs["per"]["%s|raas" % name]
            if kc in refc["per"]:
                res["reference_rows"]["per"][kc] = refc["per"][kc]
            ko = "%s|raao" % name
            if ko in refo["per"]:
                res["reference_rows"]["per"][ko] = refo["per"][ko]
            if k in ref["per"]:
                res["reference_rows"]["per"][k] = ref["per"][k]
    chk = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "raasl", "instr_*.npz"))):
        d = IR.load(f)
        m = (d["on_full"] > 0.5) == (d["on_terms"] > 0.5)
        s = (d["succ"] > 0.5) == (d["done"] > 0.5)
        chk[os.path.basename(f)] = dict(on_full_eq_terms=bool(m.all()), succ_eq_done_prefix=bool(s[:-1].all()), errs=int(d["errs"]))
    res["checks"] = chk
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raas-b-instr-read] -> %s" % a.out, flush=True)
    for name in a.tasks.split(","):
        for pol in ("raa", "raao", "raac", "rgp", "raas", "raasb", "raasl"):
            k = "%s|%s" % (name, pol)
            p = res["per"].get(k) or res["reference_rows"]["per"].get(k)
            if not p:
                continue
            fr = p.get("fail_ep_releases", {})
            print(k, "succ %d/%d" % (p["n_success"], p["n"]), "fail-ep releases n=%s" % fr.get("n"), "xy_rel p50 %s" % (fr.get("xy_rel") or {}).get("p50"),
                  "r(xy_rel,hand) fail %s" % fr.get("pearson_xy_rel_vs_hand"), "all %s" % p.get("all_releases", {}).get("pearson_xy_rel_vs_hand"),
                  "rel/ep %s" % p["releases_per_ep_mean"], "eps w/ release %s" % p["eps_with_release"], flush=True)


if __name__ == "__main__":
    main()

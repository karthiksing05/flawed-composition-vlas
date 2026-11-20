"""SR 86 Q1 part b (exploratory): the instrumented K5 rollouts of experiments/sr86_instr.py (W3 = results/flawed/sr86/s86w3, RAA-sL = s86w0): contacts by name, rack geometry, stall analysis,
the module / expert xy per query, the demo release spot.   python experiments/sr86_q1b.py --out results/flawed/sr86_q1b.json   (refuses to overwrite)"""
import argparse, collections, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed", "sr86")


def jl(f):
    return [json.loads(l) for l in open(f, encoding="utf-8")]


def parse(cs):
    o, r = cs.split("||") if "||" in cs else (cs, "")
    return set(x for x in o.split(";") if x), set(x for x in r.split(";") if x)


def ep(f, chan, where):
    d = np.load(f, allow_pickle=True)
    obj, tgt, eef, a = (np.asarray(d[k], float) for k in ("obj", "tgt", "eef", "a"))
    cons = [parse(str(s)) for s in d["cons"]]
    n = len(obj); z0 = obj[0, 2]
    rel = tgt - obj; xy = np.linalg.norm(rel[:, :2], axis=1)
    succ = bool(np.asarray(d["succ"]).reshape(-1).any())
    under = np.asarray(d["under"]).reshape(-1); con = np.asarray(d["contact"]).reshape(-1)
    nongrip = [set(x for x in o if "gripper0" not in x and "robot0" not in x) for o, r in cons]
    lifted = np.where(obj[:, 2] > z0 + 0.02)[0]
    r = dict(succ=succ, n=n, tgt=[float(v) for v in tgt[0]], obj0=[float(v) for v in obj[0]], under_end=float(under[-1]), contact_end=float(con[-1]))
    if len(lifted):
        s0 = int(lifted[0]); r["lift"] = s0
        sp = np.r_[0, np.linalg.norm(np.diff(obj[:, :2], axis=0), axis=1)]
        cmd = np.r_[0, np.linalg.norm(a[1:, :2], axis=1)]
        st = (np.arange(n) >= s0) & (cmd > 0.5) & (sp < 0.0015) & (xy > 0.05)
        r["stall_steps"] = int(st.sum())
        k = int(np.argmax(st)) if st.any() else n - 1
        r["k"] = k
        r["at_k"] = dict(rel_cm=[100 * float(v) for v in rel[k]], xy_cm=100 * float(xy[k]), obj=[float(v) for v in obj[k]], eef=[float(v) for v in eef[k]], a=[float(v) for v in a[k]],
                         obj_contacts=sorted(cons[k][0]), robot_contacts=sorted(cons[k][1]))
        r["own_contact_names_after_lift"] = collections.Counter(x for i in range(s0, n) for x in nongrip[i]).most_common(6)
        r["robot_contact_names_after_lift"] = collections.Counter(x for i in range(s0, n) for x in cons[i][1]).most_common(6)
        fc = {}
        for i in range(s0, n):
            for x in nongrip[i]:
                if x not in fc: fc[x] = i
        r["first_contact_step"] = fc
        if fc:
            first = min(fc.values()); r["first_nongripper_contact"] = dict(step=first, names=sorted(nongrip[first]), rel_cm=[100 * float(v) for v in rel[first]], c_cm=100 * float(obj[first, 2] - tgt[first, 2]))
        # velocity: obj xy displacement per step median while commanded
        r["min_xy_cm"] = 100 * float(xy[s0:].min())
        r["obj_xy_path_after_lift_cm"] = 100 * float(sp[s0:].sum())
    # world-frame check of the channel: true_cm of the last target-role query vs tgt - obj at the same step (index n_settle offset 10)
    return r, d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    res = {}
    for arm, dd in (("W3", "s86w3"), ("SL", "s86w0")):
        cf = sorted(glob.glob(os.path.join(D, dd, "chan_*.jsonl")))
        ch = jl(cf[0]) if cf else []
        wf = sorted(glob.glob(os.path.join(D, dd, "where_*.jsonl")))
        wh = jl(wf[0]) if wf else []
        wh = wh[len(wh) - len(ch):] if wh else []
        byep = collections.defaultdict(list)
        for i, c in enumerate(ch): byep[c["ep"]].append((c, wh[i] if wh else None))
        E, stat = {}, None
        for f in sorted(glob.glob(os.path.join(D, dd, "instr_raac_8_t*.npz"))):
            t = int(f.rsplit("_t", 1)[1][:-4])
            r, d = ep(f, ch, wh)
            q = byep.get(t, [])
            tq = [(c, w) for c, w in q if c["role"] == "target"]
            r["n_target_q"] = len(tq)
            if tq and tq[0][1] is not None:
                own = [(c, w) for c, w in tq if w["own"]]
                r["own_q"] = len(own); r["own_first_t"] = int(own[0][0]["t"]) if own else None
                if own:
                    r["own_median_rel_xy_cm"] = 100 * float(np.median([w["rel_xy"] for c, w in own]))
                    r["own_median_exp_xy_norm"] = float(np.median([np.linalg.norm(w["exp_xy"]) for c, w in own])); r["own_median_w_xy_norm"] = float(np.median([np.linalg.norm(w["w_xy"]) for c, w in own]))
                    r["own_exp_cos_w_med"] = float(np.median([np.dot(w["exp_xy"], w["w_xy"]) / (np.linalg.norm(w["exp_xy"]) * np.linalg.norm(w["w_xy"]) + 1e-9) for c, w in own]))
            if tq:
                r["chan_last_target_q"] = dict(t=int(tq[-1][0]["t"]), rel_cm=tq[-1][0]["rel_cm"], true_cm=tq[-1][0]["true_cm"], err=float(tq[-1][0]["err_cm"]))
            E[t] = r
            if stat is None:
                E["_static"] = json.loads(str(d["static"]))
                stat = 1
        res[arm] = E
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    for arm in res:
        E = {k: v for k, v in res[arm].items() if k != "_static"}
        print(arm, "n", len(E), "succ", sum(e["succ"] for e in E.values()))
        for t, e in sorted(E.items()):
            print(arm, t, "S" if e["succ"] else "F", "lift", e.get("lift"), "k", e.get("k"), "stall", e.get("stall_steps"), "xy@k %.1f" % e["at_k"]["xy_cm"] if "at_k" in e else "", "relk", [round(v) for v in e["at_k"]["rel_cm"]] if "at_k" in e else "",
                  "objC", e.get("at_k", {}).get("obj_contacts"), "robC", e.get("at_k", {}).get("robot_contacts"), "first", e.get("first_nongripper_contact", {}).get("names"), "ownq", e.get("own_q"), "minxy %.1f" % e.get("min_xy_cm", -1))
        st = res[arm].get("_static")
        if st: print(arm, "STATIC parent", st["parent"], "pos", st["parent_pos"]); [print("  ", k, v) for k, v in st["geoms"].items()]


if __name__ == "__main__":
    main()

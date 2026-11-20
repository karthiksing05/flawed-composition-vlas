"""SR 86 Q1 part d (exploratory): (a) the demos' release offset in the channel frame, per target class; (b) K5: the channel's object reference vs the body origin, hover-at-reference stalls, arm-rack contact,
and the channel offset at success for RAA-sL.   python experiments/sr86_q1d.py --out results/flawed/sr86_q1d.json  (refuses to overwrite)"""
import argparse, collections, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed", "sr86")
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"


def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {("p%d" % p): round(float(np.percentile(v, p)), 2) for p in ps} if len(v) else {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    res = {}
    eps = [json.loads(l) for l in open(os.path.join(LR, "meta", "episodes.jsonl"), encoding="utf-8")]
    cls = collections.defaultdict(list)
    for e in eps:
        tg = [t for t in e["tasks"] if t.startswith("target:")]
        if not tg: continue
        f = os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % e["episode_index"])
        if not os.path.exists(f): continue
        t = pq.read_table(f).to_pydict(); st = np.asarray(t["state"], float); ac = np.asarray(t["actions"], float)
        h = np.where(np.isclose(st[:, 3], 2.0))[0]
        if len(h) < 5: continue
        g = ac[h, 6]; op = np.where(g < 0)[0]; r = int(op[0]) if len(op) else len(h) - 1
        rel = 100 * st[h[r], :3]
        cls[tg[0]].append(dict(rel=[float(v) for v in rel], xy=float(np.hypot(rel[0], rel[1]))))
    res["demo_release"] = {k: dict(n=len(v), dx=q([x["rel"][0] for x in v]), dy=q([x["rel"][1] for x in v]), dz=q([x["rel"][2] for x in v]), xy=q([x["xy"] for x in v])) for k, v in cls.items()}
    for arm, dd in (("W3", "s86w3"), ("SL", "s86w0")):
        cf = sorted(glob.glob(os.path.join(D, dd, "chan_*.jsonl")))[0]; ch = [json.loads(l) for l in open(cf, encoding="utf-8")]
        wf = cf.replace("chan_", "where_"); wh = [json.loads(l) for l in open(wf, encoding="utf-8")] if os.path.exists(wf) else None
        if wh: wh = wh[len(wh) - len(ch):]
        byep = collections.defaultdict(list)
        for i, c in enumerate(ch): byep[c["ep"]].append((c, wh[i] if wh else None))
        out = []
        for f in sorted(glob.glob(os.path.join(D, dd, "instr_raac_8_t*.npz"))):
            t = int(f.rsplit("_t", 1)[1][:-4]); d = np.load(f, allow_pickle=True)
            obj, tgt, A = (np.asarray(d[k], float) for k in ("obj", "tgt", "a")); cons = [str(s) for s in d["cons"]]; succ = np.asarray(d["succ"]).reshape(-1)
            n = len(obj); rel = 100 * (tgt - obj)
            tq = [(c, w) for c, w in byep[t] if c["role"] == "target"]
            r = dict(trial=t, success=bool(succ.any()), n=n)
            offs = []
            for c, w in tq:
                row = int(c["t"]) + 10
                if row < n: offs.append([c["rel_cm"][i] - rel[row, i] for i in range(3)])      # channel rel (target - object reference) minus body-origin rel
            if offs: r["chan_minus_origin_rel_cm_med"] = [float(np.median([o[i] for o in offs])) for i in range(3)]; r["chan_minus_origin_rel_cm_sd"] = [float(np.std([o[i] for o in offs])) for i in range(3)]
            rob_rack = [i for i in range(n) if any(">wine_rack" in x for x in cons[i].split("||")[-1].split(";"))]
            r["steps_robot_rack_contact"] = len(rob_rack); r["steps_robot_rack_contact_last100"] = sum(1 for i in rob_rack if i >= n - 100)
            if wh:
                hov = [(c, w) for c, w in tq if w["own"] and w["rel_xy"] < 0.015]
                r["hover_queries(own & chan rel_xy<1.5cm)"] = len(hov)
                r["hover_c_cm_med"] = 100 * float(np.median([w["c"] for c, w in hov])) if hov else None
                r["hover_chan_rel_cm_med"] = [float(np.median([c["rel_cm"][i] for c, w in hov])) for i in range(3)] if hov else None
                r["own_q"] = sum(1 for c, w in tq if w["own"])
            if succ.any():
                k = int(np.argmax(succ > 0)); r["success_step"] = k
                row_t = k - 10
                cq = [c for c, w in tq if c["t"] <= row_t]
                r["origin_rel_at_success_cm"] = [float(v) for v in rel[k]]
                r["chan_rel_at_success_cm"] = cq[-1]["rel_cm"] if cq else None
            else:
                last = [(c, w) for c, w in tq][-1:] 
                r["chan_rel_last_target_q_cm"] = last[0][0]["rel_cm"] if last else None; r["origin_rel_final_cm"] = [float(v) for v in rel[-1]]
            out.append(r)
        res[arm] = out
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("DEMO release rel (target - object, channel frame, cm) by target class:")
    for k, v in res["demo_release"].items(): print(" ", k, v["n"], "dx", v["dx"].get("p50"), "dy", v["dy"].get("p50"), "dz", v["dz"].get("p50"), "xy p50/p95", v["xy"].get("p50"), v["xy"].get("p95"))
    for arm in ("W3", "SL"):
        o = res[arm]
        print(arm, "n", len(o), "succ", sum(r["success"] for r in o), "chan-origin offset med", np.median([r["chan_minus_origin_rel_cm_med"] for r in o if "chan_minus_origin_rel_cm_med" in r], axis=0).round(1).tolist(),
              "mean sd", np.mean([r["chan_minus_origin_rel_cm_sd"] for r in o if "chan_minus_origin_rel_cm_sd" in r], axis=0).round(1).tolist())
        print(arm, "eps with robot-rack contact in last 100 steps:", sum(r["steps_robot_rack_contact_last100"] > 0 for r in o), " median steps", np.median([r["steps_robot_rack_contact"] for r in o]))
        if arm == "W3":
            print(arm, "eps hovering at reference (>=10 queries own & chan rel_xy<1.5cm):", sum(r.get("hover_queries(own & chan rel_xy<1.5cm)", 0) >= 10 for r in o), "median queries", np.median([r.get("hover_queries(own & chan rel_xy<1.5cm)", 0) for r in o]),
                  "median c", np.median([r["hover_c_cm_med"] for r in o if r.get("hover_c_cm_med") is not None]))
        else:
            sx = [r for r in o if "chan_rel_at_success_cm" in r and r["chan_rel_at_success_cm"]]
            print(arm, "chan rel at success (cm) median", np.median([r["chan_rel_at_success_cm"] for r in sx], axis=0).tolist(), "origin rel at success", np.median([r["origin_rel_at_success_cm"] for r in sx], axis=0).round(1).tolist())
        if arm == "W3":
            for r in o:
                print(" ", r["trial"], "S" if r["success"] else "F", "hover q", r.get("hover_queries(own & chan rel_xy<1.5cm)"), "c", r.get("hover_c_cm_med") and round(r["hover_c_cm_med"], 1), "chan rel", r.get("hover_chan_rel_cm_med"), "robot-rack steps", r["steps_robot_rack_contact"], "last100", r["steps_robot_rack_contact_last100"])


if __name__ == "__main__":
    main()

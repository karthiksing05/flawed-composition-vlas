"""SR 86 Q1 (exploratory) part a: (1) where the rack demos release relative to the channel's target reference (state dx, dy, dz = target - object, held frames, role id 2);
(2) the SR 85 K5 instrumented rollouts (W3 vs RAA-sL, results/flawed/raa_instr/raasw3K and raaslK, 20 states) in the site frame; (3) channel error on held queries.
    python experiments/sr86_q1a.py --out results/flawed/sr86_q1a.json   (refuses to overwrite)"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed", "raa_instr")
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"


def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {("p%d" % p): round(float(np.percentile(v, p)), 3) for p in ps} if len(v) else {}


def demos():
    out = []
    for ep in range(200, 250):
        f = os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % ep)
        if not os.path.exists(f): continue
        t = pq.read_table(f).to_pydict()
        st = np.asarray(t["state"], float); ac = np.asarray(t["actions"], float)
        h = np.where(np.isclose(st[:, 3], 2.0))[0]
        if len(h) < 5: continue
        rel = st[h, :3] * 100; a = ac[h]
        xy = np.linalg.norm(rel[:, :2], axis=1)
        g = a[:, 6]
        rel_i = np.where(g < 0)[0]                                         # first open command while target role
        r = int(rel_i[0]) if len(rel_i) else len(h) - 1
        n_pre = min(r, 20)
        out.append(dict(ep=ep, n_held=len(h), start_xy=float(xy[0]), start_dz=float(rel[0, 2]), rel_at_release=[float(v) for v in rel[r]], xy_at_release=float(xy[r]),
                        min_xy=float(xy.min()), xy_end=float(xy[-1]), rel_end=[float(v) for v in rel[-1]], rel_before_release=[float(v) for v in rel[max(r - 10, 0)]],
                        mean_axy_last10=[float(v) for v in a[max(r - 10, 0):r + 1, :2].mean(0)] if r > 0 else None, axy_max_last30=float(np.abs(a[max(r - 30, 0):r + 1, :2]).max()),
                        descent_cm=float(rel[0, 2] - rel[r, 2]) if False else float(rel[max(r - 30, 0), 2] - rel[r, 2]), c_min=float((-rel[:, 2]).min()), c_max=float((-rel[:, 2]).max()),
                        closed_after_first_open=float((g[r:] > 0).mean())))
    return out


def ep_feats(f, site_hint=None):
    d = np.load(f, allow_pickle=True)
    obj, tgt, eef, a = (np.asarray(d[k], float) for k in ("obj", "tgt", "eef", "a"))
    ds = np.asarray(d["dsite"], float).reshape(len(obj), -1); under = np.asarray(d["under"]).reshape(-1); con = np.asarray(d["contact"]).reshape(-1)
    succ = bool(np.asarray(d["succ"]).reshape(-1).any())
    n = len(obj); z0 = obj[0, 2]
    rel = tgt - obj; xy = np.linalg.norm(rel[:, :2], axis=1)
    lifted = np.where(obj[:, 2] > z0 + 0.02)[0]
    r = dict(n=n, succ=succ, site_size=[float(v) for v in d["site_size"]], tgt_z_above_obj0_cm=100 * float(tgt[0, 2] - z0))
    if len(lifted):
        s0 = lifted[0]; r["lift"] = int(s0)
        sp = np.r_[0, np.linalg.norm(np.diff(obj[:, :2], axis=0), axis=1)]
        cmd = np.r_[0, np.linalg.norm(a[1:, :2], axis=1)]
        far = xy > 0.05
        st = (np.arange(n) >= s0) & (cmd > 0.5) & (sp < 0.0015) & far
        r["stall_steps"] = int(st.sum()); r["stall_first"] = int(np.argmax(st)) if st.any() else None
        k = int(np.argmax(st)) if st.any() else n - 1
        r["at_stall"] = dict(step=k, rel_cm=[100 * float(v) for v in rel[k]], xy_cm=100 * float(xy[k]), c_cm=100 * float(obj[k, 2] - tgt[k, 2]), obj_rise_cm=100 * float(obj[k, 2] - z0),
                             a_xy=[float(v) for v in a[k, :2]], eef_minus_obj_cm=[100 * float(v) for v in (eef[k] - obj[k])], contact_flag=float(con[k]), dsite_cm=[100 * float(v) for v in ds[k]])
        r["min_xy_cm"] = 100 * float(xy[s0:].min()); r["xy_when_min"] = int(s0 + np.argmin(xy[s0:]))
        r["c_at_min_xy_cm"] = 100 * float((obj[:, 2] - tgt[:, 2])[s0:][np.argmin(xy[s0:])])
        r["max_c_cm"] = 100 * float((obj[:, 2] - tgt[:, 2]).max())
    r["end"] = dict(rel_cm=[100 * float(v) for v in rel[-1]], dsite_cm=[100 * float(v) for v in ds[-1]], under=float(under[-1]), contact=float(con[-1]), obj_z_cm=100 * float(obj[-1, 2] - z0))
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    res = {}
    dm = demos()
    res["demos_rack"] = dict(n=len(dm), per=dm,
                             rel_at_release_dx=q([e["rel_at_release"][0] for e in dm]), rel_at_release_dy=q([e["rel_at_release"][1] for e in dm]), rel_at_release_dz=q([e["rel_at_release"][2] for e in dm]),
                             xy_at_release=q([e["xy_at_release"] for e in dm]), start_xy=q([e["start_xy"] for e in dm]), start_dz=q([e["start_dz"] for e in dm]), n_held=q([e["n_held"] for e in dm]),
                             c_max=q([e["c_max"] for e in dm]), c_min=q([e["c_min"] for e in dm]), closed_after_first_open=q([e["closed_after_first_open"] for e in dm]))
    for arm, dd in (("W3", "raasw3K"), ("SL", "raaslK")):
        fs = sorted(glob.glob(os.path.join(D, dd, "instr_raac_8_t*.npz")))
        E = [ep_feats(f) for f in fs]
        res[arm] = dict(n=len(E), succ=sum(e["succ"] for e in E), per=E)
        for nm, key in (("stall_xy_cm", lambda e: e["at_stall"]["xy_cm"] if "at_stall" in e else None), ("stall_c_cm", lambda e: e["at_stall"]["c_cm"] if "at_stall" in e else None),
                        ("min_xy_cm", lambda e: e.get("min_xy_cm")), ("max_c_cm", lambda e: e.get("max_c_cm"))):
            res[arm][nm] = q([key(e) for e in E])
    # channel error on target-role queries, W3 vs SL
    for arm, dd in (("W3", "raasw3K"), ("SL", "raaslK")):
        fs = sorted(glob.glob(os.path.join(D, dd, "chan_*.jsonl")))
        L = [json.loads(l) for f in fs for l in open(f, encoding="utf-8")]
        t = [x for x in L if x["role"] == "target"]
        res[arm]["chan_target_q"] = len(t)
        res[arm]["chan_target_err_cm"] = q([x["err_cm"] for x in t])
        res[arm]["chan_target_axis_err_cm"] = {ax: q([x["axis_err_cm"][i] for x in t if "axis_err_cm" in x]) for i, ax in enumerate("xyz")}
        res[arm]["chan_target_true_cm"] = {ax: q([x["true_cm"][i] for x in t]) for i, ax in enumerate("xyz")}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    for k in ("W3", "SL"):
        print(k, {x: y for x, y in res[k].items() if x != "per"})
    print("demos", {x: y for x, y in res["demos_rack"].items() if x != "per"})


if __name__ == "__main__":
    main()

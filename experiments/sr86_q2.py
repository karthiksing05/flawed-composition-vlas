"""SR 86 diagnostic (Q2, exploratory): per-cell paired W3 vs RAA-sL on Phase B stored rollouts (wasw3 vs wasl; full arm), flips and per-episode mechanism features.
Reads only existing results; writes a new json (refuses to overwrite).   python experiments/sr86_q2.py --out results/flawed/sr86_q2.json"""
import argparse, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed")
KEYS = {0: "F7_relL", 1: "F7_relR", 2: "F7_pos", 3: "F7_neg", 4: "F7_uniL", 5: "F7_uniR", 6: "F7_uniP", 7: "K2t", 8: "F6_b", 9: "F6_a", 10: "K1t"}


def jl(f):
    return [json.loads(l) for l in open(f, encoding="utf-8")]


def eps(arm_dir, tid):
    j = json.load(open(os.path.join(D, arm_dir, "full_%d.json" % tid), encoding="utf-8"))["tasks"][str(tid)]["episodes"]
    return {int(e["trial"]): e for e in j}


def load_logs(arm_dir):
    """-> {(tid, trial): list of (chan, where)} aligned by dropping the first 2 where lines of each job file"""
    out = {}
    stat = [0, 0]
    for cf in sorted(glob.glob(os.path.join(D, arm_dir, "chan_*.jsonl"))):
        wf = cf.replace("chan_", "where_")
        ch = jl(cf)
        wh = jl(wf) if os.path.exists(wf) else [dict(m=(c["role"] == "target"), own=False, c=-c["rel_cm"][2] / 100.0, rel_xy=float(np.hypot(c["rel_cm"][0], c["rel_cm"][1])) / 100.0, g_old=float("nan"), exp_xy=[0, 0], w_xy=[0, 0]) for c in ch]
        wh = wh[len(wh) - len(ch):]                      # the extra lines are at the start (warm-up calls)
        for c, w in zip(ch, wh):
            if c.get("arm") != "full":
                continue
            stat[0] += 1; stat[1] += int((c["role"] == "target") == bool(w["m"]))
            out.setdefault((c["tid"], c["ep"]), []).append((c, w))
    return out, stat


def feats(tid, trial, q, trk, ep):
    names = [str(x) for x in trk["names"]]
    i = [k for k, e in enumerate(ep_list(trk))]
    return None


def ep_list(trk):
    return [k for k in trk.files if k.startswith("e") and k.endswith("_pos")]


def episode_feats(q, trk, k, roles):
    names = [str(x) for x in trk["names"]]
    pos, ee, a6 = trk["e%d_pos" % k], trk["e%d_ee" % k], trk["e%d_a6" % k]
    r = {}
    io, it = (names.index(roles[0]) if roles[0] in names else None), (names.index(roles[1]) if roles[1] in names else None)
    held = [(c, w) for c, w in q if c["role"] == "target"]
    r["n_q"] = len(q); r["n_held_q"] = len(held)
    r["own_frac"] = float(np.mean([w["own"] for c, w in held])) if held else None
    r["held_err_cm_med"] = float(np.median([c["err_cm"] for c, w in held])) if held else None
    r["held_err_cm_max"] = float(np.max([c["err_cm"] for c, w in held])) if held else None
    r["held_t0"] = int(held[0][0]["t"]) if held else None
    r["held_t1"] = int(held[-1][0]["t"]) if held else None
    if io is not None and it is not None:
        T = len(pos); d = np.linalg.norm(pos[:, io, :2] - pos[:, it, :2], axis=1)
        r["end_xy_cm"] = 100 * float(d[-1]); r["min_xy_cm"] = 100 * float(d.min()); r["end_dz_cm"] = 100 * float(pos[-1, io, 2] - pos[-1, it, 2])
        r["obj_rise_cm"] = 100 * float(pos[:, io, 2].max() - pos[0, io, 2])
        lift = np.where(pos[:, io, 2] > pos[0, io, 2] + 0.02)[0]
        r["lift_step"] = int(lift[0]) if len(lift) else None
        r["T"] = T
        g = a6 > 0                                                       # command closed
        r["grip_closed_end"] = bool(g[-1])
        # last step the gripper command was closed while the object is > 5 cm above start (a carry)
        r["ee_obj_end_cm"] = 100 * float(np.linalg.norm(ee[-1] - pos[-1, io]))
    own_q = [(c, w) for c, w in held if w["own"]]
    if own_q:
        dif = [np.linalg.norm(np.array(w["w_xy"]) - np.array(w["exp_xy"])) for c, w in own_q]
        cosv = []
        for c, w in own_q:
            a, b = np.array(w["w_xy"]), np.array(w["exp_xy"])
            if np.linalg.norm(a) > 1e-3 and np.linalg.norm(b) > 1e-3: cosv.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        r["own_q"] = len(own_q); r["own_xy_diff_med"] = float(np.median(dif)); r["own_cos_med"] = float(np.median(cosv)) if cosv else None
        r["own_relxy_cm_med"] = 100 * float(np.median([w["rel_xy"] for c, w in own_q]))
        r["own_c_cm_med"] = 100 * float(np.median([w["c"] for c, w in own_q]))
        r["own_min_relxy_cm"] = 100 * float(np.min([w["rel_xy"] for c, w in own_q]))
        # gripper opening: expert's g_old vs sent
    gs = [w["g_old"] for c, w in held]
    r["g_old_last3_mean"] = float(np.mean(gs[-3:])) if gs else None
    r["rel_xy_last_cm"] = 100 * float(held[-1][1]["rel_xy"]) if held else None
    r["c_last_cm"] = 100 * float(held[-1][1]["c"]) if held else None
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit("refusing to overwrite")
    LG = {k: load_logs(k) for k in ("wasw3", "wasl")}
    print("where/chan role agreement", {k: v[1] for k, v in LG.items()})
    res = dict(cells={}, flips=[])
    for tid in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10):
        E3, EL = eps("wasw3", tid), eps("wasl", tid)
        T3 = np.load(os.path.join(D, "wasw3", "full_%d_track.npz" % tid), allow_pickle=True)
        TL = np.load(os.path.join(D, "wasl", "full_%d_track.npz" % tid), allow_pickle=True)
        js3 = json.load(open(os.path.join(D, "wasw3", "full_%d_wa.json" % tid), encoding="utf-8"))
        tr3 = [int(e["trial"]) for e in js3["eps"]]
        jsl = json.load(open(os.path.join(D, "wasl", "full_%d_wa.json" % tid), encoding="utf-8"))
        trl = [int(e["trial"]) for e in jsl["eps"]]
        roles = js3["roles"]
        tr = sorted(set(E3) & set(EL))
        s3 = np.array([E3[t]["success"] for t in tr], bool); sl = np.array([EL[t]["success"] for t in tr], bool)
        res["cells"][KEYS[tid]] = dict(n=len(tr), w3=float(s3.mean()), sl=float(sl.mean()), ss=int((s3 & sl).sum()), lose=int((~s3 & sl).sum()), gain=int((s3 & ~sl).sum()), ff=int((~s3 & ~sl).sum()))
        for t, x3, xl in zip(tr, s3, sl):
            f3 = episode_feats(LG["wasw3"][0].get((tid, t), []), T3, tr3.index(t), roles)
            fl = episode_feats(LG["wasl"][0].get((tid, t), []), TL, trl.index(t), roles)
            f3["steps"] = E3[t]["steps"]; fl["steps"] = EL[t]["steps"]
            res["flips"].append(dict(cell=KEYS[tid], tid=tid, trial=t, w3=bool(x3), sl=bool(xl), kind=("lose" if (xl and not x3) else "gain" if (x3 and not xl) else "ss" if x3 else "ff"), f3=f3, fl=fl))
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in res["cells"].items(): print(k, v)


if __name__ == "__main__":
    main()

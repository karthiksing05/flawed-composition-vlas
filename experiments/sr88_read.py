"""SR 88 reader: per-episode diagnosis of the instrumented re-runs (results/flawed/sr88/<arm><A|B>/), EXPLORATORY, no labels.
Joins, per episode: the simulator track (instr_*.npz: executed action, eef, object, target, contact terms, contacts), the server's per-call where log
(m, rel_xy, e_xy, own, gate, c, cls, g_old, g_new; episodes split by the ep_start markers the client wrote into the same file) and the channel log
(rel_cm vs true_cm).  Call j covers policy steps 5j..5j+4 (env step n = 10 + s, the first 10 are settle steps).
Writes results/flawed/sr88/read_eps.json, timelines_<cell>.txt and prints the tables."""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

R = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
S = R + "/sr88"
MU = json.load(open("/coc/flash10/ksingara3/flawed-composition-vlas/lifelong_ood/policy/raasw5_mu.json"))["mu_m"]
CELL = {("a", 22): "N4", ("a", 26): "N6", ("a", 10): "K6", ("a", 0): "K1", ("b", 8): "F6_b", ("b", 4): "uniL"}
ARMS = ["w5", "w4", "w5c1", "raasl"]
SETTLE = 10
R_HAND = 0.03
CLEAR = 0.03


def split(path):
    eps, cur = {}, None
    if not path or not os.path.exists(path):
        return eps
    for ln in open(path, encoding="utf-8"):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "marker" in d:
            key = (d["tid"], d["i"])
            if key != cur:
                cur = key
                eps[key] = []
        elif cur is not None:
            eps[cur].append(d)
    return eps


def short(c):
    return c.replace("robot0_", "").replace("gripper0_", "g_")


def parse_cons(s):
    o, r = (str(s).split("||") + [""])[:2]
    return [x for x in o.split(";") if x], [x for x in r.split(";") if x]


def analyse(arm, ph, tid, trial, d, W, C):
    n = d["n"][:, 0].astype(int)
    a, eef, obj, tgt = d["a"], d["eef"], d["obj"], d["tgt"]
    xy, succ, contact, zok = d["xy"][:, 0], d["succ"][:, 0], d["contact"][:, 0], d["z_ok"][:, 0]
    cons = [str(x) for x in d["cons"]]
    T = len(n)
    z0 = obj[0, 2]
    lifted = obj[:, 2] > z0 + 0.02
    a6 = a[:, 6]
    rows = [i for i in range(T) if n[i] >= SETTLE]
    carry = next((i for i in rows if a6[i] > 0 and lifted[i]), None)
    rel = None
    if carry is not None:
        rel = next((i for i in range(carry + 1, T) if a6[i] < 0), None)
    ok = bool(succ.max() > 0.5) if T else False
    ok_end = bool(succ[-1] > 0.5)
    res = dict(arm=arm, ph=ph, tid=tid, trial=trial, success=ok_end, success_any=ok, steps=int(n[-1] + 1 - SETTLE), oe=str(d["oe"]), te=str(d["te"]),
               lifted=bool(lifted.any()), carry_n=None if carry is None else int(n[carry]))
    if not W and C:                                                      # RAA-sL (where_own 0) has no where log: synthesise the channel view from the chan log (no ownership, no gate)
        W = []
        for c_ in sorted(C, key=lambda q: q["t"]):
            rc = c_["rel_cm"]
            W.append(dict(m=(c_.get("role") == "target"), rel_xy=float(np.hypot(rc[0], rc[1])) / 100, e_xy=float(np.hypot(rc[0], rc[1])) / 100, c=-rc[2] / 100.0, own=False, gate=False, g_old=0.0, g_new=0.0, cls=None))
    ncalls = int(np.ceil(res["steps"] / 5.0))
    res["ncalls_expected"], res["ncalls_where"] = ncalls, len(W)
    res["aligned"] = (len(W) == ncalls)
    # class and mu
    cls = next((w["cls"] for w in W if w.get("cls")), None)
    if cls is None:
        cl = [c.get("prompt") for c in C if c.get("role") == "target" and c.get("prompt")]
        cls = cl[0].split(". rel:")[0].strip() if cl else None
    res["cls"] = cls
    mu = np.array(MU.get(cls, [0.0, 0.0])) * 100 if cls else np.zeros(2)       # cm
    res["mu_cm"] = [round(float(v), 2) for v in mu]
    # ownership / gate summary over calls where a target is held
    held = [j for j, w in enumerate(W) if w.get("m")]
    res["held_calls"] = len(held)
    own = [bool(W[j].get("own")) for j in range(len(W))]
    gate = [bool(W[j].get("gate")) for j in range(len(W))]
    res["own_calls"] = int(sum(own))
    res["gate_calls"] = int(sum(gate))
    if any(own):
        res["first_own_s"] = 5 * own.index(True)
        res["last_own_s"] = 5 * (len(own) - 1 - own[::-1].index(True))
    # hand-off: the first call after the module has owned where the expert owns with m, c >= CLEAR (inside R)
    ho = None
    seen_own = False
    for j in range(len(W)):
        if own[j]:
            seen_own = True
        elif seen_own and W[j].get("m") and W[j].get("c", 0) >= CLEAR and W[j].get("e_xy", 9) <= R_HAND:
            ho = j
            break
    res["handoff_call"] = ho
    # channel info by call: true rel (channel frame, cm) minus mu
    byt = {c["t"] // 5: c for c in C if "t" in c}

    def etrue(j):
        c = byt.get(j)
        if c is None or c.get("role") != "target":
            return None
        return float(np.hypot(c["true_cm"][0] - mu[0], c["true_cm"][1] - mu[1]))

    def echan(j):
        c = byt.get(j)
        if c is None or c.get("role") != "target":
            return None
        return float(np.hypot(c["rel_cm"][0] - mu[0], c["rel_cm"][1] - mu[1]))
    if ho is not None:
        res["ho_e_where_cm"] = round(100 * W[ho]["e_xy"], 2)
        res["ho_e_chan_cm"] = echan(ho)
        res["ho_e_true_cm"] = etrue(ho)
        lim = (rel - 0 if rel is not None else T)
        jend = (max(0, (int(n[rel]) - SETTLE) // 5) if rel is not None else len(W) - 1)
        post = [etrue(j) for j in range(ho, jend + 1) if etrue(j) is not None]
        res["post_ho_e_true_max_cm"] = max(post) if post else None
        res["post_ho_e_true_end_cm"] = post[-1] if post else None
        res["reentry"] = int(sum(1 for j in range(ho + 1, len(own)) if own[j] and not own[j - 1]))
    res["end_xy_cm"] = round(100 * float(xy[-1]), 2)
    # release
    if rel is not None:
        jr = max(0, min((int(n[rel]) - SETTLE) // 5, len(W) - 1)) if W else None
        res.update(rel_s=int(n[rel]) - SETTLE, rel_xy_cm=round(100 * float(xy[rel]), 2), rel_dz_cm=round(100 * float(obj[rel, 2] - tgt[rel, 2]), 2),
                   rel_contact=float(contact[rel]), rel_zok=float(zok[rel]), rel_own=(own[jr] if jr is not None and jr < len(own) else None),
                   rel_gate_sofar=int(sum(gate[:jr + 1])) if jr is not None else None)
        res["rel_true_minus_mu_cm"] = etrue(jr) if jr is not None else None
        if jr is not None and jr < len(W):
            res["rel_chan_rel_cm"] = round(100 * float(W[jr].get("rel_xy", 0)), 2)
            res["rel_chan_e_cm"] = round(100 * float(W[jr].get("e_xy", 0)), 2)
            res["rel_c_cm"] = round(100 * float(W[jr].get("c", 0)), 2)
        res["rel_eef_obj_z_cm"] = round(100 * float(eef[rel, 2] - obj[rel, 2]), 2)
        # all releases (open commands after closed carry), final offset
        evs = [i for i in range(carry + 1, T) if a6[i] < 0 and a6[i - 1] >= 0]
        res["n_release_events"] = len(evs)
        res["end_xy_cm"] = round(100 * float(xy[-1]), 2)
        res["end_obj_dz_cm"] = round(100 * float(obj[-1, 2] - tgt[-1, 2]), 2)
        res["drift_after_release_cm"] = round(res["end_xy_cm"] - res["rel_xy_cm"], 2)
    # held-target channel error (cm, vs simulator) and module / expert disagreement on owned calls near the target
    ee = [c_["err_cm"] for c_ in C if c_.get("role") == "target" and "err_cm" in c_]
    res["held_err_med_cm"] = None if not ee else round(float(np.median(ee)), 2)
    dis = []
    for j in range(len(W)):
        w_ = W[j]
        if w_.get("own") and w_.get("e_xy", 9) <= 0.06 and "exp_xy" in w_ and "w_xy" in w_:
            ex, wx = np.array(w_["exp_xy"]), np.array(w_["w_xy"])
            if np.linalg.norm(ex) > 1e-6 and np.linalg.norm(wx) > 1e-6:
                dis.append(float(ex @ wx / (np.linalg.norm(ex) * np.linalg.norm(wx))))
    res["dis_n"] = len(dis)
    res["dis_frac"] = None if not dis else round(float(np.mean([v < 0 for v in dis])), 2)
    res["dis_last"] = None if not dis else round(dis[-1], 2)
    bad = [c_ for c_ in sorted(C, key=lambda q: q["t"]) if c_.get("role") == "target" and c_.get("err_cm", 0) > 5.0]
    res["err_gt5_calls"] = len(bad)
    res["err_gt5_first_j"] = None if not bad else int(bad[0]["t"] // 5)
    res["err_gt5_ho_gap"] = None if (not bad or ho is None) else int(bad[0]["t"] // 5) - ho
    # crossing into R (all arms): first held call with c >= CLEAR and channel |rel - mu| <= R; chunk speed in, overshoot along the approach direction after
    jx = next((j for j in held if W[j].get("c", 0) >= CLEAR and echan(j) is not None and echan(j) <= 100 * R_HAND), None)
    res["cross_call"] = jx
    if jx is not None:
        ix = next((i for i in range(T) if n[i] == SETTLE + 5 * jx), None)
        if ix is not None and ix >= 5:
            dv = (eef[ix, :2] - eef[ix - 5, :2]) * 100
            res["cross_v_cm"] = round(float(np.hypot(*dv)), 2)
            res["cross_e_chan_cm"] = round(float(echan(jx)), 2)
            res["cross_c_cm"] = round(100 * float(W[jx].get("c", 0)), 2)
            if np.hypot(*dv) > 0.3:
                u = dv / np.hypot(*dv)
                prj = []
                for j2 in range(jx, min(len(W), jx + 9)):
                    c2 = byt.get(j2)
                    if c2 is not None and c2.get("role") == "target":
                        prj.append(float((c2["true_cm"][0] - mu[0]) * u[0] + (c2["true_cm"][1] - mu[1]) * u[1]))
                res["cross_overshoot_cm"] = round(max(0.0, -min(prj)), 2) if prj else None
                res["cross_ahead0_cm"] = round(prj[0], 2) if prj else None
    if jx is not None:                                                    # obstacle contacts (robot body touching something other than the target / object) from the crossing to the release / end, all arms
        i_x2 = next((i for i in range(T) if n[i] == SETTLE + 5 * jx), None)
        i_e2 = rel if rel is not None else T - 1
        pc = Counter()
        for i in range(i_x2 if i_x2 is not None else 0, i_e2 + 1):
            _, r_ = parse_cons(cons[i])
            pc.update(set(short(x) for x in r_ if "finger" not in x))
        res["post_x_obst"] = int(sum(v for k_, v in pc.items() if not k_.split(">")[-1].startswith(("plate", "akita", "flat_stove", "wine_rack", "wine_bottle"))))
        res["post_x_obst_names"] = [k_ for k_, v in pc.most_common(3)]
    # frame offset delta = (target - object body origin, sim) - (channel true rel): the channel's object / target reference vs the body origins
    dl = []
    for j in held:
        c_ = byt.get(j)
        i0 = next((i for i in range(T) if n[i] == SETTLE + 5 * j), None)
        if c_ is None or c_.get("role") != "target" or i0 is None:
            continue
        dl.append([100 * float(tgt[i0, 0] - obj[i0, 0]) - c_["true_cm"][0], 100 * float(tgt[i0, 1] - obj[i0, 1]) - c_["true_cm"][1]])
    res["ref_delta_cm"] = None if not dl else [round(float(v), 2) for v in np.median(np.array(dl), axis=0)]
    # channel true rel (cm) at the last held call (success) or at the release call: where the episode ended in the channel frame
    jl = None
    if res["success"] and len(W):
        jl = len(W) - 1
    elif rel is not None and len(W):
        jl = max(0, min((int(n[rel]) - SETTLE) // 5, len(W) - 1))
    c_ = byt.get(jl) if jl is not None else None
    res["end_chan_true_cm"] = None if c_ is None or c_.get("role") != "target" else list(c_["true_cm"][:2])
    if ho is not None:
        res["ho_c_cm"] = round(100 * float(W[ho].get("c", 0)), 2)
        i_h = next((i for i in range(T) if n[i] == SETTLE + 5 * ho), None)
        res["ho_body_xy_cm"] = None if i_h is None else round(100 * float(xy[i_h]), 2)
        i_e = (rel if rel is not None else T - 1)
        post_c = Counter()
        for i in range(i_h if i_h is not None else 0, i_e + 1):
            _, r_ = parse_cons(cons[i])
            post_c.update(set(short(x) for x in r_ if "finger" not in x))
        res["post_ho_contacts"] = post_c.most_common(3)
        res["post_ho_obst"] = int(sum(v for k_, v in post_c.items() if not k_.split(">")[-1].startswith(("plate", "akita", "flat_stove", "wine_rack"))))
    # category
    if res["success"]:
        cat = "success"
    elif not res["lifted"] or carry is None:
        cat = "no_carry"
    elif rel is None:
        cat = "no_release"
    elif res["rel_xy_cm"] >= 3.0:
        cat = "release_far"
    else:
        cat = "release_near_fail"
    res["cat"] = cat
    # contacts: robot (non-finger) contacts during module-owned calls and the last 100 steps
    cown, clast = Counter(), Counter()
    for i in rows:
        j = (int(n[i]) - SETTLE) // 5
        o, r = parse_cons(cons[i])
        rr = [short(x) for x in r if not x.startswith("gripper0_finger") and "finger" not in x]
        if 0 <= j < len(own) and own[j]:
            cown.update(rr)
        if i >= T - 100:
            clast.update(rr)
    res["contacts_own_top"] = cown.most_common(3)
    res["contacts_last100_top"] = clast.most_common(3)
    # timeline
    lines = []
    for j, w in enumerate(W):
        if not w.get("m") and not (rel is not None and j <= (int(n[rel]) - SETTLE) // 5 + 3 and carry is not None and j >= (int(n[carry]) - SETTLE) // 5 - 2):
            continue
        s0 = 5 * j
        i0 = next((i for i in range(T) if n[i] == SETTLE + s0), None)
        if i0 is None:
            continue
        i1 = min(T - 1, i0 + 4)
        c = byt.get(j, {})
        o, r = parse_cons(cons[i1])
        lines.append("j%02d s%03d c=%5.1f rel=%5.1f e=%5.1f own=%d gate=%d go=%+.2f gn=%+.2f | chan_rel=%s true_rel=%s err=%.1f | obj-tgt xy=%5.1f dz=%5.1f eef-obj dz=%5.1f a6=%+.1f | %s || %s" % (
            j, s0, 100 * w.get("c", 0), 100 * w.get("rel_xy", 0), 100 * w.get("e_xy", 0), int(bool(w.get("own"))), int(bool(w.get("gate"))), w.get("g_old", 0), w.get("g_new", 0),
            c.get("rel_cm"), c.get("true_cm"), c.get("err_cm", -1), 100 * xy[i1], 100 * (obj[i1, 2] - tgt[i1, 2]), 100 * (eef[i1, 2] - obj[i1, 2]), a6[i1],
            ",".join(sorted(short(x) for x in o))[:40], ",".join(sorted(short(x) for x in r if "finger" not in x))[:60]))
    res["_timeline"] = lines
    return res


def main():
    out = {}
    P = json.load(open(S + "/pick.json"))
    for arm in ARMS:
        for K in "AB":
            d0 = "%s/%s%s" % (S, arm, K)
            if not os.path.isdir(d0):
                continue
            wl = sorted(glob.glob(d0 + "/where_*.jsonl"))
            cl = sorted(glob.glob(d0 + "/chan_*.jsonl"))
            # take the newest (a requeue restarts the file with markers appended; episodes keyed by (tid, i) keep the last occurrence)
            WE = {}
            CE = {}
            for p in wl:
                WE.update(split(p))
            for p in cl:
                CE.update(split(p))
            for f in sorted(glob.glob(d0 + "/instr_*.npz")):
                z = np.load(f, allow_pickle=True)
                tid, trial = int(z["tid"]), int(z["trial"])
                ph = "a" if K == "A" else "b"
                brp = "%s/br/br_%s_%d.json" % (S, ph, tid)
                br = [b["trial"] for b in json.load(open(brp))]
                i = br.index(trial) if trial in br else None
                W = WE.get((tid, i), [])
                C = CE.get((tid, i), [])
                try:
                    r = analyse(arm, ph, tid, trial, z, W, C)
                except Exception as ex:
                    r = dict(arm=arm, ph=ph, tid=tid, trial=trial, err=repr(ex), cat="ERR")
                r["cell"] = CELL.get((ph, tid), str((ph, tid)))
                out[(arm, tid, trial)] = r
    json.dump([{k: v for k, v in r.items() if k != "_timeline"} for r in out.values()], open(S + "/read_eps.json", "w"), indent=0, default=str)
    cells = ["N4", "N6", "K6", "K1", "F6_b", "uniL"]
    print("episodes read:", len(out), "errors:", sum(1 for r in out.values() if r["cat"] == "ERR"), "misaligned:", Counter(r["arm"] for r in out.values() if r.get("aligned") is False))
    for r in out.values():
        if r.get("aligned") is False and r["arm"] != "raasl":
            print("MISALIGNED", r["arm"], r["cell"], r["trial"], "where", r["ncalls_where"], "expected", r["ncalls_expected"], "steps", r["steps"])
    for r in out.values():
        if r["cat"] == "ERR":
            print("ERR", r["arm"], r["tid"], r["trial"], r["err"])
    # timelines
    for cell in cells:
        with open("%s/timelines_%s.txt" % (S, cell), "w") as fh:
            trials = sorted({k[2] for k, r in out.items() if r["cell"] == cell})
            for t in trials:
                fh.write("##### %s trial %d\n" % (cell, t))
                for arm in ARMS:
                    r = out.get((arm, [k[1] for k, rr in out.items() if rr["cell"] == cell][0], t))
                    if r is None:
                        continue
                    fh.write("== %s success=%s cat=%s steps=%s rel_xy=%s ho=%s e_ho(w/chan/true)=%s/%s/%s own_calls=%s gate_calls=%s mu=%s\n" % (
                        arm, r.get("success"), r.get("cat"), r.get("steps"), r.get("rel_xy_cm"), r.get("handoff_call"), r.get("ho_e_where_cm"), r.get("ho_e_chan_cm"), r.get("ho_e_true_cm"),
                        r.get("own_calls"), r.get("gate_calls"), r.get("mu_cm")))
                    for ln in r.get("_timeline", []):
                        fh.write("   " + ln + "\n")
    # tables
    def med(v):
        v = [x for x in v if x is not None]
        return None if not v else round(float(np.median(v)), 2)
    for cell in cells:
        print("\n=== %s" % cell)
        print("%-6s %3s %5s | %s | rel_xy med  frac>=3 | gate_calls med | own_calls med | handoff n  e_true_ho med  post_max med | dz@rel med" % ("arm", "n", "succ", "categories"))
        for arm in ARMS:
            rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm]
            if not rs:
                continue
            cats = Counter(r["cat"] for r in rs)
            rx = [r["rel_xy_cm"] for r in rs if r.get("rel_xy_cm") is not None]
            ho = [r for r in rs if r.get("handoff_call") is not None]
            print("%-6s %3d %5.2f | %s | %s %s | %s | %s | %d %s %s | %s" % (
                arm, len(rs), np.mean([r["success"] for r in rs]), dict(cats), med(rx), None if not rx else round(float(np.mean([x >= 3 for x in rx])), 2),
                med([r.get("gate_calls") for r in rs]), med([r.get("own_calls") for r in rs]), len(ho), med([r.get("ho_e_true_cm") for r in ho]),
                med([r.get("post_ho_e_true_max_cm") for r in ho]), med([r.get("rel_dz_cm") for r in rs])))
    def medv(vs):
        vs = [v for v in vs if v is not None]
        return None if not vs else [round(float(x), 1) for x in np.median(np.array(vs), axis=0)]
    print("\n=== features by cell / arm / outcome (medians): ho_c = height of the target ref above the object at hand-off (cm), ref_delta = (tgt-obj sim body) - channel rel (cm),")
    print("    end_chan_true = channel true rel at success / release (cm), mu, post-handoff obstacle-contact steps, e_true at hand-off, e_true max after")
    for cell in cells:
        for arm in ARMS:
            for okk in (True, False):
                rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and bool(r["success"]) == okk]
                if not rs:
                    continue
                hs = [r for r in rs if r.get("handoff_call") is not None]
                print("%-5s %-6s %-5s n=%2d ho_n=%2d ho_c=%s ref_delta=%s end_chan_true=%s mu=%s obst=%s e_ho=%s post_max=%s ho_body_xy=%s cls=%s oe=%s te=%s" % (
                    cell, arm, "ok" if okk else "FAIL", len(rs), len(hs), med([r.get("ho_c_cm") for r in hs]), medv([r.get("ref_delta_cm") for r in rs]),
                    medv([r.get("end_chan_true_cm") for r in rs]), rs[0].get("mu_cm"), med([r.get("post_ho_obst") for r in hs]), med([r.get("ho_e_true_cm") for r in hs]),
                    med([r.get("post_ho_e_true_max_cm") for r in hs]), med([r.get("ho_body_xy_cm") for r in hs]), rs[0].get("cls"), rs[0].get("oe"), rs[0].get("te")))
    print("\n=== mechanism flags, W5 episodes (fractions of the group); ho_hi = hand-off at c >= 5 cm; leave_R = true |rel - mu| > 3 cm after hand-off; obst = robot touches a non-target / non-object body after hand-off;")
    print("    reent = module re-owns xy after hand-off; w4band = first release with channel |rel_xy| > 2 cm (W4's gate would have blocked it); rel_hi = release at c >= 3 cm above the target ref; far = first-release body xy >= 3 cm; gate = any gate call")
    for cell in cells:
        for okk in (False, True):
            rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == "w5" and bool(r["success"]) == okk]
            if not rs:
                continue
            hs = [r for r in rs if r.get("handoff_call") is not None]
            rl = [r for r in rs if r.get("rel_xy_cm") is not None]
            fr = lambda L, f: None if not L else round(sum(1 for r in L if f(r)) / len(L), 2)
            print("%-5s %-4s n=%2d ho_n=%2d | ho_hi=%s leave_R=%s obst=%s reent=%s | rel_n=%2d w4band=%s rel_hi=%s far=%s | gate=%s own_calls_med=%s" % (
                cell, "ok" if okk else "FAIL", len(rs), len(hs), fr(hs, lambda r: r.get("ho_c_cm", 0) >= 5), fr(hs, lambda r: (r.get("post_ho_e_true_max_cm") or 0) > 3),
                fr(hs, lambda r: r.get("post_ho_obst", 0) > 0), fr(hs, lambda r: r.get("reentry", 0) > 0), len(rl), fr(rl, lambda r: r.get("rel_chan_rel_cm", 0) > 2),
                fr(rl, lambda r: r.get("rel_c_cm", 0) >= 3), fr(rl, lambda r: r["rel_xy_cm"] >= 3), fr(rs, lambda r: r.get("gate_calls", 0) > 0), med([r.get("own_calls") for r in rs])))
    print("\n=== same flags for W4 / W5c1 / RAA-sL releases (w4band, rel_hi, far) by outcome")
    for cell in cells:
        for arm in ("w4", "w5c1", "raasl"):
            for okk in (False, True):
                rl = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and bool(r["success"]) == okk and r.get("rel_xy_cm") is not None]
                if not rl:
                    continue
                fr = lambda L, f: None if not L else round(sum(1 for r in L if f(r)) / len(L), 2)
                print("%-5s %-6s %-4s rel_n=%2d w4band=%s rel_hi=%s far=%s rel_xy_med=%s rel_c_med=%s" % (cell, arm, "ok" if okk else "FAIL", len(rl), fr(rl, lambda r: r.get("rel_chan_rel_cm", 0) > 2),
                      fr(rl, lambda r: r.get("rel_c_cm", 0) >= 3), fr(rl, lambda r: r["rel_xy_cm"] >= 3), med([r["rel_xy_cm"] for r in rl]), med([r.get("rel_c_cm") for r in rl])))
    print("\n=== crossing into R = 3 cm (first held call, c >= 3 cm, channel |rel - mu| <= 3): chunk speed in (cm per 5 steps), height c, overshoot past the target along the approach direction in the next 8 calls (cm), by cell / arm / outcome (medians, n)")
    for cell in cells:
        for arm in ARMS:
            for okk in (True, False):
                rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and bool(r["success"]) == okk and r.get("cross_v_cm") is not None]
                if not rs:
                    continue
                print("%-5s %-6s %-5s n=%2d v_in=%s c=%s e=%s ahead0=%s overshoot=%s frac_over>1.5=%s" % (cell, arm, "ok" if okk else "FAIL", len(rs), med([r["cross_v_cm"] for r in rs]),
                      med([r.get("cross_c_cm") for r in rs]), med([r.get("cross_e_chan_cm") for r in rs]), med([r.get("cross_ahead0_cm") for r in rs]),
                      med([r.get("cross_overshoot_cm") for r in rs]), round(sum(1 for r in rs if (r.get("cross_overshoot_cm") or 0) > 1.5) / len(rs), 2)))
    print("\n=== gate radius separation: success of the episode by the channel |rel_xy| at its FIRST release (band), per arm, pooled over the six cells and per cell (n, success)")
    for arm in ARMS:
        row = []
        for lo, hi in ((0, 2), (2, 3), (3, 99)):
            rs = [r for r in out.values() if r["arm"] == arm and r.get("rel_chan_rel_cm") is not None and lo <= r["rel_chan_rel_cm"] < hi]
            row.append("%g-%g: n=%d succ=%s far=%s" % (lo, hi, len(rs), None if not rs else round(float(np.mean([r["success"] for r in rs])), 2), None if not rs else round(float(np.mean([r["rel_xy_cm"] >= 3 for r in rs])), 2)))
        print("%-6s " % arm + " | ".join(row))
    for cell in cells:
        for arm in ARMS:
            row = []
            for lo, hi in ((0, 2), (2, 3), (3, 99)):
                rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and r.get("rel_chan_rel_cm") is not None and lo <= r["rel_chan_rel_cm"] < hi]
                row.append("%g-%g: n=%d succ=%s" % (lo, hi, len(rs), None if not rs else round(float(np.mean([r["success"] for r in rs])), 2)))
            print("%-5s %-6s " % (cell, arm) + " | ".join(row))
    print("\n=== gate calls (calls with an open command replaced by closed) and release delay: W4 / W5c1 / W5 per cell (mean calls, frac of episodes with >= 1)")
    for cell in cells:
        for arm in ("w5", "w4", "w5c1"):
            rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm]
            print("%-5s %-6s gate_calls mean=%.2f frac>=1=%.2f  | own_calls mean=%.2f" % (cell, arm, np.mean([r.get("gate_calls", 0) for r in rs]), np.mean([r.get("gate_calls", 0) > 0 for r in rs]), np.mean([r.get("own_calls", 0) for r in rs])))
    print("\n=== held-target channel error (median over queries, cm) and module/expert disagreement on owned calls within 6 cm (frac of owned calls with cos(expert xy, module xy) < 0), by cell / arm / outcome (medians)")
    for cell in cells:
        for arm in ("w5", "w4", "w5c1", "raasl"):
            for okk in (True, False):
                rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and bool(r["success"]) == okk]
                if not rs:
                    continue
                print("%-5s %-6s %-5s n=%2d held_err=%s dis_frac=%s dis_n=%s dis_last=%s" % (cell, arm, "ok" if okk else "FAIL", len(rs), med([r.get("held_err_med_cm") for r in rs]),
                      med([r.get("dis_frac") for r in rs]), med([r.get("dis_n") for r in rs]), med([r.get("dis_last") for r in rs])))
    print("\n=== success by held-target channel error bin (episodes, all cells pooled per arm; and uniL / F6_b / K6 / N6 / N4 / K1 separately for W5)")
    for arm in ARMS:
        row = []
        for lo, hi in ((0, 1), (1, 2), (2, 99)):
            rs = [r for r in out.values() if r["arm"] == arm and r.get("held_err_med_cm") is not None and lo <= r["held_err_med_cm"] < hi]
            row.append("%g-%g: n=%d succ=%s" % (lo, hi, len(rs), None if not rs else round(float(np.mean([r["success"] for r in rs])), 2)))
        print("%-6s " % arm + " | ".join(row))
    print("\n=== channel breakdown: episodes with any held-phase query whose channel error > 5 cm (frac of episodes, median first call, median gap to hand-off in calls), by cell / arm / outcome")
    for cell in cells:
        for arm in ("w5", "w4", "w5c1", "raasl"):
            for okk in (True, False):
                rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and bool(r["success"]) == okk]
                if not rs:
                    continue
                bd = [r for r in rs if r.get("err_gt5_calls", 0) > 0]
                print("%-5s %-6s %-5s n=%2d any_err>5=%d (%.2f) first_j=%s gap_to_ho=%s calls_med=%s" % (cell, arm, "ok" if okk else "FAIL", len(rs), len(bd), len(bd) / len(rs), med([r.get("err_gt5_first_j") for r in bd]),
                      med([r.get("err_gt5_ho_gap") for r in bd]), med([r.get("err_gt5_calls") for r in bd])))
    print("\n=== per-state detail for W5 vs W4 vs W5c1 (cat[:4] rel_xy_body / body dz at first release / chan c at release / hand-off c / hand-off true e / overshoot / ref_delta)")
    for cell in cells:
        print("-- " + cell)
        trials = sorted({k[2] for k, r in out.items() if r["cell"] == cell})
        for t in trials:
            parts = []
            for arm in ("w5", "w4", "w5c1"):
                r = next((rr for k, rr in out.items() if k[0] == arm and k[2] == t and rr["cell"] == cell), None)
                if r is None:
                    parts.append("%-5s -" % arm)
                    continue
                f = lambda v: "-" if v is None else ("%.1f" % v)
                parts.append("%-5s %-4s rx=%s dz=%s c=%s hoc=%s eho=%s ov=%s rd=%s" % (arm, r["cat"][:4], f(r.get("rel_xy_cm")), f(r.get("rel_dz_cm")), f(r.get("rel_c_cm")), f(r.get("ho_c_cm")),
                                                                              f(r.get("ho_e_true_cm")), f(r.get("cross_overshoot_cm")), "-" if not r.get("ref_delta_cm") else "%.1f/%.1f" % tuple(r["ref_delta_cm"])))
            print("  t%02d " % t + " || ".join(parts))
    print("\n=== W5 success vs hand-off quantities (pooled over the six cells; episodes with a hand-off)")
    hs = [r for r in out.values() if r["arm"] == "w5" and r.get("handoff_call") is not None]
    for name, key, bins in (("true e at hand-off (cm)", "ho_e_true_cm", ((0, 1.5), (1.5, 2.5), (2.5, 99))), ("hand-off height c (cm)", "ho_c_cm", ((0, 6), (6, 9), (9, 12), (12, 99))),
                            ("overshoot after crossing (cm)", "cross_overshoot_cm", ((0, 1), (1, 2.5), (2.5, 99))), ("chunk speed at crossing (cm / 5 steps)", "cross_v_cm", ((0, 4), (4, 5.5), (5.5, 99)))):
        row = []
        for lo, hi in bins:
            rs = [r for r in hs if r.get(key) is not None and lo <= r[key] < hi]
            row.append("%g-%g: n=%d succ=%s" % (lo, hi, len(rs), None if not rs else round(float(np.mean([r["success"] for r in rs])), 2)))
        print("%-38s " % name + " | ".join(row))
    print("\n=== reproduction of the SR 87 outcome in the instrumented re-run (same state, same arm): SR87 success -> re-run success counts")
    for cell, d in P.items():
        for arm in ARMS:
            a = b = c_ = e = 0
            for t, s, _ in d["rows"]:
                r = out.get((arm, d["tid"], t))
                if r is None or s.get(arm, -1) < 0:
                    continue
                if s[arm] == 1:
                    a += int(r["success"]); b += int(not r["success"])
                else:
                    c_ += int(r["success"]); e += int(not r["success"])
            print("%-5s %-6s SR87 ok -> ok %d / fail %d | SR87 fail -> ok %d / fail %d" % (cell, arm, a, b, c_, e))
    print("\n=== obstacle-contact steps from the crossing to the release / end (robot body touching a body other than target / object), by cell / arm / outcome: frac of episodes with >= 10 steps, median steps")
    for cell in cells:
        for arm in ARMS:
            for okk in (True, False):
                rs = [r for r in out.values() if r["cell"] == cell and r["arm"] == arm and bool(r["success"]) == okk and r.get("post_x_obst") is not None]
                if not rs:
                    continue
                nm = Counter(x for r in rs for x in r.get("post_x_obst_names", []))
                print("%-5s %-6s %-5s n=%2d obst>=10: %.2f median=%s top=%s" % (cell, arm, "ok" if okk else "FAIL", len(rs), np.mean([r["post_x_obst"] >= 10 for r in rs]), med([r["post_x_obst"] for r in rs]), nm.most_common(2)))
    # release offset vs success, pooled by arm
    print("\n=== success vs first-release true xy offset (cm), pooled over cells, by arm")
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 99)]
    for arm in ARMS:
        row = []
        for lo, hi in bins:
            rs = [r for r in out.values() if r["arm"] == arm and r.get("rel_xy_cm") is not None and lo <= r["rel_xy_cm"] < hi]
            rs_end = [r for r in out.values() if r["arm"] == arm and r.get("rel_xy_cm") is None and r["success"] and lo <= r["end_xy_cm"] < hi]
            row.append("%d-%d: n=%d succ=%s (+%d succ w/o release, offset at end)" % (lo, hi, len(rs), None if not rs else round(float(np.mean([r["success"] for r in rs])), 2), len(rs_end)))
        print(arm, " | ".join(row))
    # paired table per state
    print("\n=== per state outcomes (category per arm)")
    for cell in cells:
        trials = sorted({k[2] for k, r in out.items() if r["cell"] == cell})
        print("-- " + cell)
        for t in trials:
            parts = []
            for arm in ARMS:
                r = next((rr for k, rr in out.items() if k[0] == arm and k[2] == t and rr["cell"] == cell), None)
                parts.append("%s:%s%s" % (arm, "-" if r is None else r["cat"][:9], "" if r is None or r.get("rel_xy_cm") is None else "(%.1f)" % r["rel_xy_cm"]))
            print("  t%02d  " % t + "  ".join(parts))


if __name__ == "__main__":
    main()

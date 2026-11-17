"""RAA place-step instrumentation, analysis (EXPLORATORY; plan docs/notes/raa_instr_plan.md).  CPU, numpy only.

    python experiments/raa_instr_read.py [--dir results/flawed/raa_instr] [--out results/flawed/raa_instr.json]   (refuses to overwrite)

Inputs: <dir>/{raa,rgp}/instr_<arm>_<tid>_t<trial>.npz (experiments/raa_instr.py), the policies' own result jsons (raa_instr_<tid>.json,
pi05_rgp_<tid>.json) and RAA's chan_raa_*.jsonl (role per query).  Row i of an npz is the state after env step n = i-1 (row 0 = after
set_init_state); policy step k = n - 10 (libero_client's 10 settle steps); the query at policy step t sees row t + 10.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
WAIT = 10
TIDS = {"N1": 16, "N4": 22, "N6": 26, "K5": 8}
EXIST = {"raa": lambda tid: os.path.join(OUT, "raa", "raa_rx_%d.json" % tid),
         "rgp": lambda tid: os.path.join(OUT, "raa_rgp_rx", "pi05_rgp_%d.json" % tid)}
MINE = {"raa": lambda d, tid: os.path.join(d, "raa", "raa_instr_%d.json" % tid),
        "rgp": lambda d, tid: os.path.join(d, "rgp", "pi05_rgp_%d.json" % tid)}
XY_TOL, TIP_DEG, SETTLE, NEAR_XY, NEAR_REL = 0.03, 30.0, 20, 0.05, 0.04


def zaxis(q):
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["site"] = bool(d["site"])
    d["trial"], d["tid"] = int(d["trial"]), int(d["tid"])
    za = zaxis(d["objq"])
    d["tilt"] = np.degrees(np.arccos(np.clip(za @ za[0], -1, 1)))               # change of the object's body-z axis from its start
    d["abs_tilt"] = np.degrees(np.arccos(np.clip(za[:, 2], -1, 1)))
    return d


def hold_mask(d):
    closed = d["a"][:, 6] > 0
    lifted = d["obj"][:, 2] > d["obj"][0, 2] + 0.01
    near = np.linalg.norm(d["obj"] - d["eef"], axis=1) < 0.15
    h = closed & lifted & near
    h[: WAIT + 1] = False
    return h


def carries(d):
    """Closed-command runs (a6 > 0, at least 3 rows) in which the object is lifted 1 cm and within 15 cm of the eef for at least 3 rows.
    A carry ends when the gripper command opens (a release) or at the episode end."""
    closed = d["a"][:, 6] > 0
    closed[: WAIT + 1] = False
    H = hold_mask(d)
    return [(s, e) for s, e in runs(closed, 3) if H[s:e + 1].sum() >= 3]


def runs(m, minlen=3):
    out, s = [], None
    for i, v in enumerate(m):
        if v and s is None:
            s = i
        if (not v) and s is not None:
            if i - s >= minlen:
                out.append((s, i - 1))
            s = None
    if s is not None and len(m) - s >= minlen:
        out.append((s, len(m) - 1))
    return out


def cat(d, i):
    """Outcome category of the object at row i, by the terms of On(obj, tgt)."""
    if (not d["site"]) and d["tilt"][i] > TIP_DEG:
        return "TIPPED"
    if d["site"]:
        if d["under"][i] < 0.5:
            return "UNDER_FALSE"
        return "NO_CONTACT" if d["contact"][i] < 0.5 else "VALID"
    if d["xy"][i] >= XY_TOL:
        return "OFF_XY"
    if d["z_ok"][i] < 0.5:
        return "Z_FAIL"
    return "NO_CONTACT" if d["contact"][i] < 0.5 else "VALID"


def near_valid(d, i):
    """Upright and within a slack of the valid set: xy <= 4 cm (body target); inside the site box widened by 1 cm in x/y and 3 cm in z."""
    if (not d["site"]) and d["tilt"][i] > TIP_DEG:
        return False
    if d["site"]:
        s, ds = d["site_size"], d["dsite"][i]
        return bool(np.all(np.abs(ds[:2]) < s[:2] + 0.01) and s[2] - 0.035 < ds[2] < s[2] + 0.13)
    return bool(d["xy"][i] <= NEAR_REL)


def rel_obj(d, i):
    """object - target in the world frame (m) and the xy / height scalars."""
    r = d["obj"][i] - d["tgt"][i]
    return float(np.linalg.norm(r[:2])), float(r[2])


def episode(d, roles):
    H = hold_mask(d)
    R = carries(d)
    n = len(H)
    succ = bool(d["done"][-1] > 0)
    ev = []
    for q, (s, e) in enumerate(R):
        j = e + 1
        rec = dict(hold=[int(s - 1 - WAIT + 1) + 0, int(e - WAIT)], held_to_end=bool(e == n - 1))     # policy-step indices (row i -> n = i-1 -> k = n-10)
        rec["hold"] = [int(s - 1 - WAIT), int(e - 1 - WAIT)]
        if e == n - 1:
            ev.append(rec)
            continue
        p = e
        xy, dz = rel_obj(d, p)
        rec.update(release_row=int(j), cmd_open=bool((d["a"][j:j + 3, 6] <= 0).any()),
                   xy_rel=xy, dz_rel=dz, tilt_rel=float(d["tilt"][p]), eef_z=float(d["eef"][p, 2]),
                   eef_speed=float(np.linalg.norm(d["eef"][p] - d["eef"][max(p - 3, 0)]) / 3.0),
                   obj_to_eef=float(np.linalg.norm(d["obj"][p] - d["eef"][p])), dropped_closed=bool(np.linalg.norm(d["obj"][p] - d["eef"][p]) > 0.15), terms_rel=dict(z_ok=float(d["z_ok"][p]), contact=float(d["contact"][p]),
                   under=float(d["under"][p])), cat_rel=cat(d, p))
        nxt = R[q + 1][0] if q + 1 < len(R) else n
        s_i = int(min(j + SETTLE, nxt - 1, n - 1))
        xy2, dz2 = rel_obj(d, s_i)
        rec.update(settle_row=s_i, xy_settle=xy2, dz_settle=dz2, tilt_settle=float(d["tilt"][s_i]), cat_settle=cat(d, s_i),
                   valid_after=bool((d["on_terms"][j:s_i + 1] > 0.5).any() or (d["succ"][j:s_i + 1] > 0.5).any()),
                   obj_speed_settle=float(np.linalg.norm(d["obj"][s_i] - d["obj"][max(s_i - 2, 0)]) / 2.0),
                   max_tilt_after=float(d["tilt"][j:s_i + 1].max()), near_release=bool(xy <= NEAR_XY))
        # re-pick: the next close command after the release (hover/re-pick attempt), and the next hold
        c = np.flatnonzero(d["a"][j:, 6] > 0)
        c = int(j + c[0]) if len(c) else None
        rec["reclose_row"] = c
        if c is not None:
            xy3, dz3 = rel_obj(d, c)
            rec.update(reclose_delay=int(c - j), reclose_xy=xy3, reclose_dz=dz3, reclose_tilt=float(d["tilt"][c]), reclose_cat=cat(d, c),
                       reclose_near_valid=near_valid(d, c))
        ev.append(rec)
    last = n - 1
    xyf, dzf = rel_obj(d, last)
    ep = dict(trial=d["trial"], success=succ, abs_tilt0=float(d["abs_tilt"][0]), abs_tilt_final=float(d["abs_tilt"][-1]), tilt_final=float(d["tilt"][-1]), success_while_held=bool(succ and R and R[-1][1] == n - 1), steps=int(n - 1 - WAIT), n_hold_runs=len(R), n_releases=sum(1 for r in ev if "release_row" in r),
              held_to_end=bool(R and R[-1][1] == n - 1), final=dict(xy=xyf, dz=dzf, tilt=float(d["tilt"][last]), cat=cat(d, last)),
              first_lift_row=None, events=ev, snapshot_err=int(d["errs"]))
    lift = np.flatnonzero((d["obj"][:, 2] > d["obj"][0, 2] + 0.01) & (np.arange(n) > WAIT))
    ep["first_lift_row"] = int(lift[0]) if len(lift) else None
    # role flips (RAA): query t sees row t + 10
    if roles is not None:
        t = np.array([q["t"] for q in roles])
        tgt = np.array([q["role"] == "target" for q in roles])
        fl = []
        for i in np.flatnonzero(np.diff(tgt.astype(int)) < 0) + 1:                  # target -> object
            row = int(t[i]) + WAIT
            if row >= n:
                continue
            f = dict(t=int(t[i]), row=row, held=bool(H[row]), cat=cat(d, row), near_valid=near_valid(d, row), tilt=float(d["tilt"][row]),
                     xy=rel_obj(d, row)[0], on_terms_window10=bool((d["on_terms"][max(row - 10, 0):row + 1] > 0.5).any()),
                     on_terms_window20=bool((d["on_terms"][max(row - 20, 0):row + 1] > 0.5).any()))
            prev = [r for r in ev if "release_row" in r and r["release_row"] <= row]
            f["since_release"] = int(row - prev[-1]["release_row"]) if prev else None
            fl.append(f)
        ep["flips"] = fl
        ep["n_target_queries"] = int(tgt.sum())
        ep["n_queries"] = int(len(tgt))
    return ep


def load_roles(rdir, tid_lang):
    runs_, out = {}, {}
    for f in sorted(glob.glob(os.path.join(rdir, "chan_raa_*.jsonl"))):
        cur = {}
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            k = (r["text"], int(r["ep"]))
            if r["t"] == 0:
                cur[k] = []
            cur.setdefault(k, []).append(r)
        runs_.update(cur)
    for (text, ep), v in runs_.items():
        out[(text, ep)] = v
    return out


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def summarize(eps, roles_on):
    out = dict(n=len(eps), n_success=int(sum(e["success"] for e in eps)), n_fail=int(sum(not e["success"] for e in eps)))
    rel = [(e, r) for e in eps for r in e["events"] if "release_row" in r]
    out["releases"] = len(rel)
    out["success_while_held"] = int(sum(e["success_while_held"] for e in eps))
    out["tilt_final_success_p50_p90"] = [pct(np.array([e["tilt_final"] for e in eps if e["success"]], float), 50), pct(np.array([e["tilt_final"] for e in eps if e["success"]], float), 90)]
    out["abs_tilt0_p50"] = pct(np.array([e["abs_tilt0"] for e in eps], float), 50)
    out["eps_with_release"] = int(sum(e["n_releases"] > 0 for e in eps))
    out["releases_per_ep_mean"] = float(np.mean([e["n_releases"] for e in eps])) if eps else None
    out["fail_releases_per_ep_mean"] = float(np.mean([e["n_releases"] for e in eps if not e["success"]])) if out["n_fail"] else None
    out["fail_never_lifted"] = int(sum((not e["success"]) and e["first_lift_row"] is None for e in eps))
    out["fail_held_to_end"] = int(sum((not e["success"]) and e["held_to_end"] for e in eps))
    out["fail_final_cat"] = _count([e["final"]["cat"] for e in eps if not e["success"]])
    for tag, sel in (("all_releases", rel), ("fail_ep_releases", [x for x in rel if not x[0]["success"]]),
                     ("succ_ep_releases", [x for x in rel if x[0]["success"]]),
                     ("near_releases", [x for x in rel if x[1]["near_release"]]),
                     ("near_releases_fail_ep", [x for x in rel if x[1]["near_release"] and not x[0]["success"]])):
        R = [x[1] for x in sel]
        o = dict(n=len(R))
        if R:
            for k in ("xy_rel", "dz_rel", "tilt_rel", "eef_z", "eef_speed", "xy_settle", "tilt_settle", "obj_to_eef"):
                v = np.array([r[k] for r in R], float)
                o[k] = dict(p10=pct(v, 10), p50=pct(v, 50), p90=pct(v, 90))
            o["cat_at_release"] = _count([r["cat_rel"] for r in R])
            o["cat_settle"] = _count([r["cat_settle"] for r in R])
            o["valid_after_release"] = int(sum(r["valid_after"] for r in R))
            o["dropped_while_closed"] = int(sum(r["dropped_closed"] for r in R))
            o["tilt_change_settle_gt30"] = int(sum(r["tilt_settle"] > TIP_DEG for r in R))
            o["settled_obj_speed_lt_2mm"] = int(sum(r["obj_speed_settle"] < 0.002 for r in R))
            rc = [r for r in R if r.get("reclose_row") is not None]
            o["reclose"] = dict(n=len(rc), cat=_count([r["reclose_cat"] for r in rc]), near_valid=int(sum(r["reclose_near_valid"] for r in rc)),
                                delay_p50=pct(np.array([r["reclose_delay"] for r in rc], float), 50), xy_p50=pct(np.array([r["reclose_xy"] for r in rc], float), 50))
            o["no_reclose"] = int(len(R) - len(rc))
        out[tag] = o
    if roles_on:
        fl = [(e, f) for e in eps for f in e.get("flips", [])]
        out["flips"] = dict(n=len(fl), eps_with_flip=int(sum(bool(e.get("flips")) for e in eps)),
                            fail_eps_with_flip=int(sum(bool(e.get("flips")) and not e["success"] for e in eps)),
                            flips_while_held=int(sum(f["held"] for _, f in fl)),
                            flips_with_on_terms_window10=int(sum(f["on_terms_window10"] for _, f in fl)),
                            flips_with_on_terms_window20=int(sum(f["on_terms_window20"] for _, f in fl)),
                            flips_near_valid=int(sum(f["near_valid"] for _, f in fl)),
                            cat=_count([f["cat"] for _, f in fl]),
                            since_release=dict(p10=pct(np.array([f["since_release"] for _, f in fl if f["since_release"] is not None], float), 10),
                                               p50=pct(np.array([f["since_release"] for _, f in fl if f["since_release"] is not None], float), 50),
                                               p90=pct(np.array([f["since_release"] for _, f in fl if f["since_release"] is not None], float), 90)),
                            flips_before_first_release=int(sum(f["since_release"] is None for _, f in fl)))
    return out


def _count(xs):
    o = {}
    for x in xs:
        o[x] = o.get(x, 0) + 1
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(OUT, "raa_instr"))
    ap.add_argument("--out", default=os.path.join(OUT, "raa_instr.json"))
    ap.add_argument("--tasks", default="N1,N4,N6,K5")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite " + a.out)
    meta = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["tid"]: t["language"] for t in meta["tasks"]}
    res = dict(note="EXPLORATORY; docs/notes/raa_instr_plan.md; per control step logging by experiments/raa_instr.py", dir=a.dir,
               constants=dict(XY_TOL=XY_TOL, TIP_DEG=TIP_DEG, SETTLE=SETTLE, NEAR_XY=NEAR_XY, NEAR_REL=NEAR_REL),
               identity={}, per={}, episodes={})
    for pol in ("raa", "rgp"):
        roles = load_roles(os.path.join(a.dir, "raa"), None) if pol == "raa" else None
        for name in a.tasks.split(","):
            tid = TIDS[name]
            files = sorted(glob.glob(os.path.join(a.dir, pol, "instr_%s_%d_t*.npz" % (pol, tid))))
            eps = []
            for f in files:
                d = load(f)
                q = roles.get((lang[tid], d["trial"])) if roles is not None else None
                eps.append(episode(d, q))
            key = "%s|%s" % (name, pol)
            if not eps:
                continue
            res["episodes"][key] = eps
            res["per"][key] = summarize(eps, pol == "raa")
            # identity: the logged rollout against the existing unlogged record of the same trial
            ex = {int(e["trial"]): e for e in json.load(open(EXIST[pol](tid), encoding="utf-8"))["tasks"][str(tid)]["episodes"]}
            mine = MINE[pol](a.dir, tid)
            mine = {int(e["trial"]): e for e in json.load(open(mine, encoding="utf-8"))["tasks"][str(tid)]["episodes"]} if os.path.exists(mine) else {}
            rows = []
            for e in eps:
                t = e["trial"]
                if t in ex and t in mine:
                    rows.append(dict(trial=t, steps_old=ex[t]["steps"], steps_new=mine[t]["steps"], succ_old=bool(ex[t]["success"]),
                                     succ_new=bool(mine[t]["success"]), npz_steps=e["steps"], npz_succ=e["success"]))
            same = [r for r in rows if r["steps_old"] == r["steps_new"] and r["succ_old"] == r["succ_new"]]
            res["identity"][key] = dict(n=len(rows), same_steps_and_success=len(same), diverged=[r for r in rows if r not in same],
                                        npz_matches_result=int(sum(r["npz_steps"] == r["steps_new"] and r["npz_succ"] == r["succ_new"] for r in rows)),
                                        success_old=int(sum(r["succ_old"] for r in rows)), success_new=int(sum(r["succ_new"] for r in rows)))
    # the on-terms recomputation against the simulator's own success
    chk = {}
    for pol in ("raa", "rgp"):
        for f in sorted(glob.glob(os.path.join(a.dir, pol, "instr_*.npz"))):
            d = load(f)
            k = os.path.basename(f)[6:9]
            m = (d["on_full"] > 0.5) == (d["on_terms"] > 0.5)
            s = (d["succ"] > 0.5) == (d["done"] > 0.5)
            chk[os.path.basename(f)] = dict(on_full_eq_terms=bool(m.all()), succ_eq_done_prefix=bool(s[:-1].all()), errs=int(d["errs"]))
    res["checks"] = chk
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(dict(identity=res["identity"], per=res["per"]), indent=1)[:12000])


if __name__ == "__main__":
    main()

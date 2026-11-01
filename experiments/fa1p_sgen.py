"""FLAWED_COMPOSITION prototype A, stage 1p (section 5.30), DATA: generated carries STRATIFIED by the carried object's clearance to
the nearest fixture.  5.29: the layout sampler puts the carried object within the free-space input's radius (OCC_R = 0.08 m) of a
fixture in few attempts, and the planner grasps there about half as often, so the generated data holds few carries where the
object term's free-space input is non-zero, and the policy fails there.  Here each known task gets attempts in three clearance bins
inside OCC_R, by rejection over layout seeds: [0.02, 0.04), [0.04, 0.06), [0.06, 0.08) m (0.02 m is the sampler's own gap;
clearance = layout-frame centre distance minus fa1g_plan's MOVR (object) and FIXR (fixture)).  Everything else is fa1l_cgen's:
the same sampler, the same grasp OPTIONS tried in order, the same recorded episode and ROLE-WISE acceptance; no demo action is
executed or replayed; every action comes from the proportional controller on the live simulator state.
    gen    (LIBERO venv, CPU): --task K6 --bin 0 --n 20 --chunk 0 -> DATA/gen_lp/<task>_b<bin>_c<chunk>.npz,
           results/flawed/a1p_sgen/<task>_b<bin>_c<chunk>.json
    merge  (openpi venv, CPU): DATA/gen_lo.npz + every chunk file -> DATA/gen_lp.npz (demo ids 800 + j per task, rows kept as
           executed, no no-op filter, held / offset / lift recomputed as fa1o_merge) -> results/flawed/a1p_sgen.json

    python experiments/fa1p_sgen.py gen|merge [--task K6 --bin 0 --n 20 --chunk 0]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402
import fa1g_plan as FG                                                      # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
BINS = ((0.02, 0.04), (0.04, 0.06), (0.06, 0.08))
SEED0 = 2_000_000
TASKS = ["K1", "K2", "K3", "K4", "K5", "K6"]


def first_ok(n2, seed):
    """The first position set fa1g_plan.layout would try for `seed` (same rng sequence, same constraints), or None."""
    c = n2["constraints"]
    lo, hi, gap = np.array(c["box"][0]), np.array(c["box"][1]), c["gap"]
    fixp, sf = np.array(c["fixture_xy"]), c["stove_front_world"]
    mov, fixn = n2["movable"], n2["fixtures"]
    rng = np.random.default_rng(seed)

    def ok(p):
        for i, nm in enumerate(mov):
            r = FG.MOVR[nm]
            if any(np.linalg.norm(p[i] - p[j]) < r + FG.MOVR[mov[j]] + gap for j in range(i)):
                return False
            if any(np.linalg.norm(p[i] - fixp[f]) < r + FG.FIXR[fn] + gap for f, fn in enumerate(fixn)):
                return False
            cx, cy = np.clip(p[i][0], sf[0], sf[2]), np.clip(p[i][1], sf[1], sf[3])
            if np.hypot(p[i][0] - cx, p[i][1] - cy) < r + gap:
                return False
        return True

    for _ in range(2000):
        p = rng.uniform(lo, hi, size=(len(mov), 2))
        if ok(p):
            return p
    return None


def clearance(n2, p, obj):
    i = n2["movable"].index(obj)
    fixp, fixn = n2["constraints"]["fixture_xy"], n2["fixtures"]
    return float(min(np.linalg.norm(np.asarray(p[i]) - np.asarray(fixp[f])) - FG.FIXR[fn] - FG.MOVR[obj] for f, fn in enumerate(fixn)))


def gen(task, b, n, chunk):
    import fa1l_cgen as CG
    n2, rl = FG.lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][task]
    t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = rl[task]
    tix = list(FC.KNOWN).index(task)
    lo_c, hi_c = BINS[b]
    rows = {x: [] for x in ("ep", "seg", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    rep, kept, t0, q, att = [], 0, time.time(), 0, 0
    base = SEED0 + 100000 * tix + 20000 * b + 2000 * chunk
    lay_env = FC.make_env(bddl)
    while att < n and q < 2000:
        seed = base + q
        q += 1
        p0 = first_ok(n2, seed)
        if p0 is None or not (lo_c <= clearance(n2, p0, oe) < hi_c):
            continue
        s0, p = FG.layout(lay_env, n2, np.random.default_rng(seed), seed)
        if s0 is None or not (lo_c <= clearance(n2, p, oe) < hi_c):
            continue
        att += 1
        chosen, tried = None, []
        for opt in CG.OPTIONS:
            r = CG.episode(bddl, s0, oe, te, kp, opt, seed, record=False, full=False)
            tried.append(dict(option=list(opt), held_lift=r["held_lift"]))
            if r["held_lift"]:
                chosen = opt
                break
        res = dict(seed=seed, layout=True, clearance=clearance(n2, p, oe), positions=np.round(p, 4).tolist(), tried=tried,
                   option=list(chosen) if chosen else None)
        if chosen is not None:
            r = CG.episode(bddl, s0, oe, te, kp, chosen, seed, record=True, full=True)
            res.update(success=r["success"], held_lift=r["held_lift"], steps=r["steps"])
            keep = None
            if r["success"]:
                keep, seg = len(r["ep"]["step"]), 0
            elif r["held_lift"] and r["lift_end"]:
                keep, seg = int(r["lift_end"]), 1
            res["kept"] = None if keep is None else ("episode" if seg == 0 else "grasp_segment")
            if keep:
                for x in r["ep"]:
                    rows[x] += r["ep"][x][:keep]
                rows["ep"] += [kept] * keep
                rows["seg"] += [seg] * keep
                kept += 1
        rep.append(res)
        print(f"[fa1p] sgen {task} b{b} c{chunk} {att}/{n} seed {seed} clr {res['clearance']:.3f}: option={res['option']} "
              f"kept={res.get('kept')} ({time.time() - t0:.0f}s, {q} seeds scanned)", flush=True)
    lay_env.close()
    os.makedirs(os.path.join(FC.DATA, "gen_lp"), exist_ok=True)
    np.savez_compressed(os.path.join(FC.DATA, "gen_lp", f"{task}_b{b}_c{chunk}.npz"), task_index=tix, **{x: np.asarray(v) for x, v in rows.items()})
    os.makedirs(os.path.join(OUT, "a1p_sgen"), exist_ok=True)
    with open(os.path.join(OUT, "a1p_sgen", f"{task}_b{b}_c{chunk}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, bin=list(BINS[b]), chunk=chunk, attempts=att, seeds_scanned=q, kept=kept, seed_base=base,
                       options=CG.OPTIONS, episodes=rep, generated=True,
                       note="clearance-stratified layouts; otherwise fa1l_cgen's planner and acceptance; no demo action executed"),
                  fh, indent=1)


def merge():
    import fa1b_common as FB
    G0 = dict(np.load(os.path.join(FC.DATA, "gen_lo.npz")))
    keys = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off", "touch", "lift")
    cols = {x: list(G0[x]) for x in keys}
    seg = list(G0["seg"])
    rep = {}
    for k in TASKS:
        j = 0
        r = dict(episodes=0, segments=0, rows=0, files=0)
        for f in sorted(glob.glob(os.path.join(FC.DATA, "gen_lp", f"{k}_b*_c*.npz"))):
            G = dict(np.load(f))
            r["files"] += 1
            tix = int(G["task_index"])
            for e in (np.unique(G["ep"]) if len(G["ep"]) else []):
                m = np.nonzero(G["ep"] == e)[0]
                hm, prev = FB.Held(), -1.0
                sg = int(G["seg"][m[0]])
                for s, i in enumerate(m):
                    lt, rt = bool(G["touch"][i][0]), bool(G["touch"][i][1])
                    h, off = hm.step(lt, rt, G["objp"][i], G["ee_pos"][i])
                    for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                        cols[x].append(G[x][i])
                    cols["task"].append(tix); cols["demo"].append(800 + j); cols["step"].append(s); cols["prev_grip"].append(prev)
                    cols["held"].append(float(h)); cols["off"].append(off); cols["touch"].append((lt, rt)); cols["lift"].append(G["objp"][i][2] - hm.z0)
                    seg.append(sg)
                    prev = float(G["action"][i][6])
                j += 1
                r["segments" if sg else "episodes"] += 1
                r["rows"] += len(m)
        assert j < 200, ("too many new carries for one task", k, j)
        for f in sorted(glob.glob(os.path.join(OUT, "a1p_sgen", f"{k}_b*_c*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            bb = r.setdefault("per_bin", {}).setdefault(f"{d['bin'][0]:.2f}-{d['bin'][1]:.2f}", dict(attempts=0, kept=0, kept_episode=0))
            bb["attempts"] += d["attempts"]
            bb["kept"] += sum(1 for e in d["episodes"] if e.get("kept"))
            bb["kept_episode"] += sum(1 for e in d["episodes"] if e.get("kept") == "episode")
        rep[k] = r
        print(f"[fa1p] merge {k}: {json.dumps(r)}", flush=True)
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    assert out["demo"].max() < 1000 and out["demo"].min() >= 100
    np.savez_compressed(os.path.join(FC.DATA, "gen_lp.npz"), gen=np.ones(len(out["task"]), np.int8), seg=np.asarray(seg, np.int8), **out)
    with open(os.path.join(OUT, "a1p_sgen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rep, rows=int(len(out["task"])), base_rows=int(len(G0["task"])), bins=BINS, generated=True), fh, indent=1)
    print(f"[fa1p] gen_lp rows {len(out['task'])} (gen_lo {len(G0['task'])})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "merge"))
    ap.add_argument("--task", default="K6")
    ap.add_argument("--bin", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "gen":
        gen(a.task, a.bin, a.n, a.chunk)
    else:
        merge()

"""Coverage arm (docs/COVERAGE_RESULTS.md section 1; PLAN_PARTS 10.11): GENERATED demos under the widening rule W.  Labelled ablation.

W (fixed in COVERAGE_RESULTS 1.3): for each known pick-and-place task K1-K6, initial scenes from stage 1g's layout sampler
(fa1g_plan.layout, unchanged: the four movable objects uniform over N2's box with N2's clearance rules and settle check, no 0.10 m
exclusion from the training spot); seeds BASE + 1000 j + k (j = the task's index in K1-K6); the v2 servoing planner
(pp_n4d.plan_episode(servo=True), PARTS_RESULTS 19) with stage 1g's fitted keypoints (a1g_keypoints.json, unchanged); kept iff
LIBERO's success predicate fires; rows recorded as pp_n4d2.genfam records them.  No recorded human action is executed and no
evaluation state of any read is used.

shard  (LIBERO venv, CPU) --task K1 --lo 0 --hi 40: attempts k in [lo, hi) -> DATA/gen/pp_cov_parts/<task>_<lo>_<hi>.npz + .json
       (skips if the shard exists)
merge  (LIBERO venv, CPU): per task, shards contiguous from k = 0; the first N_KEEP kept in seed order (all tasks cut to the
       smallest kept count if any task keeps fewer after CAP attempts) -> DATA/gen/pp_cov_<task>.npz (gen = 1, demo ids 100 + j),
       results/flawed/pp_cov_gen_<task>.json, results/flawed/pp_cov_gen.json
stats  (openpi venv, CPU): the rule checks and the support table, human against generated -> results/flawed/pp_cov_gen_stats.json

    python experiments/pp_cov_gen.py shard --task K5 --lo 0 --hi 40
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
GEN = os.path.join(FC.DATA, "gen")
PARTS = os.path.join(GEN, "pp_cov_parts")
BASE, N_KEEP, CAP = 8_000_000, 50, 600
PP = ["K1", "K2", "K3", "K4", "K5", "K6"]
# seed ranges W must avoid: N2 dev (familiar, new, branch noise), confirmatory bases (pp_conf_states.py), earlier generators
USED = [(600000, 600024), (700000, 700024), (13000000, 13000000 + 1000 * 24), (9100000, 9100050), (9200000, 9200050),
        (23000000, 23000000 + 1000 * 50), (900000, 906000), (950000, 950300), (960000, 963000), (970000, 973000)]
COLS = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off", "seed")


def seed_of(task, k):
    s = BASE + 1000 * PP.index(task) + k
    assert 0 <= k < CAP and not any(lo <= s < hi for lo, hi in USED), ("seed outside W or in a used range", s)
    return s


def shard(task, lo, hi):
    import fa1b_common as FB
    import pp_n4d as N
    from fa1_data import noop_mask
    from fa1g_plan import layout
    os.makedirs(PARTS, exist_ok=True)
    fn = os.path.join(PARTS, f"{task}_{lo}_{hi}")
    if os.path.exists(fn + ".json"):
        print(f"[pp_cov gen] {fn} exists", flush=True)
        return
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = FC.roles(t["language"])
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][task]
    assert (kp["object"], kp["target"]) == (oe, te), (kp["object"], kp["target"], oe, te)
    lay_env = FC.make_env(bddl)
    cols = {x: [] for x in COLS}
    rep, t0 = [], time.time()
    for k in range(lo, hi):
        seed = seed_of(task, k)
        rng = np.random.default_rng(seed)
        s0, p = layout(lay_env, n2, rng, seed)
        if s0 is None:
            rep.append(dict(k=k, seed=seed, layout=False))
            continue
        ep, info = N.plan_episode(bddl, s0, kp, oe, te, rng, servo=True)
        rep.append(dict(k=k, seed=seed, layout=True, positions=np.round(p, 5).tolist(), success=info["success"], steps=info["steps"],
                        phase=info["phase"], held_step=info["held_step"]))
        print(f"[pp_cov gen] {task} k {k} seed {seed}: success={info['success']} phase={info['phase']} ({time.time() - t0:.0f}s)", flush=True)
        if not info["success"]:
            continue
        A = np.asarray(ep["action"])
        keep = np.nonzero(noop_mask(A))[0]
        hm, prev = FB.Held(), -1.0
        for s, i in enumerate(keep):                                         # pp_n4d2.genfam's recording, unchanged
            h, off = hm.step(bool(ep["touch"][i][0]), bool(ep["touch"][i][1]), ep["objp"][i], ep["ee_pos"][i])
            for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                cols[x].append(ep[x][i])
            cols["task"].append(list(FC.KNOWN).index(task)); cols["demo"].append(k); cols["step"].append(s); cols["prev_grip"].append(prev)
            cols["held"].append(float(h)); cols["off"].append(off); cols["seed"].append(seed)
            prev = float(A[i][6])
    lay_env.close()
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step", "seed") else np.int64) for x, v in cols.items()}
    if not len(out["task"]):
        out = {x: np.zeros((0,), np.float32) for x in COLS}
    np.savez_compressed(fn + ".npz", **out)
    with open(fn + ".json", "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, lo=lo, hi=hi, base=BASE, attempts=rep, kept=int(sum(bool(r.get("success")) for r in rep)),
                       secs=time.time() - t0, generated=True, generator="pp_n4d.plan_episode(servo=True) (instrument v2, PARTS_RESULTS 19)",
                       note="widening rule W (COVERAGE_RESULTS 1.3); no human action executed; no evaluation state used"), fh, indent=1)
    print(f"[pp_cov gen] {task} [{lo}, {hi}): kept {sum(bool(r.get('success')) for r in rep)} ({time.time() - t0:.0f}s)", flush=True)


def _shards(task):
    out = []
    for f in glob.glob(os.path.join(PARTS, f"{task}_*_*.json")):
        j = json.load(open(f, encoding="utf-8"))
        out.append((j["lo"], j["hi"], f[:-5], j))
    out.sort()
    return out


def merge():
    per = {}
    for task in PP:
        sh = _shards(task)
        att, kept_k, upto = [], [], 0
        for lo, hi, base, j in sh:
            if lo != upto:
                break                                                        # contiguous from k = 0 only
            att += j["attempts"]
            kept_k += [r["k"] for r in j["attempts"] if r.get("success")]
            upto = hi
        per[task] = dict(shards=[(lo, hi) for lo, hi, _, _ in sh], contiguous_to=upto, attempts=att, kept_k=kept_k)
        done = len(kept_k) >= N_KEEP or upto >= CAP
        assert done, (task, "needs more shards", upto, len(kept_k))
    n = min(N_KEEP, min(len(v["kept_k"]) for v in per.values()))
    assert n >= 10, ("W keeps too few demos to train on (COVERAGE_RESULTS 1.9 stop)", n)
    summ = dict(n_per_task=n, base=BASE, cap=CAP, tasks={})
    for task in PP:
        take = per[task]["kept_k"][:n]
        last = take[-1]
        G = {x: [] for x in COLS}
        for lo, hi, base, _ in _shards(task):
            if lo > last:
                break
            Z = dict(np.load(base + ".npz"))
            if not len(Z["task"]):
                continue
            for x in COLS:
                G[x].append(Z[x])
        G = {x: np.concatenate(v) for x, v in G.items()}
        sel = np.isin(G["demo"], take)
        G = {x: v[sel] for x, v in G.items()}
        order = {k: j for j, k in enumerate(take)}
        G["demo"] = np.array([100 + order[int(k)] for k in G["demo"]], np.int32)   # demo ids 100 + j (j in seed order)
        G["task"] = G["task"].astype(np.int32)
        G["step"] = G["step"].astype(np.int32)
        G["gen"] = np.ones(len(G["task"]), np.int8)
        fn = os.path.join(GEN, f"pp_cov_{task}.npz")
        assert not os.path.exists(fn), f"{fn} exists (never overwritten)"
        np.savez_compressed(fn, **G)
        att = [r for r in per[task]["attempts"] if r["k"] <= last]
        lay = [r for r in att if r.get("layout")]
        rep = dict(task=task, kept=n, attempts=len(att), layouts_accepted=len(lay), planner_success=float(np.mean([r["success"] for r in lay])),
                   success_given_layout=f"{sum(r['success'] for r in lay)} of {len(lay)}", seeds=[seed_of(task, k) for k in take],
                   rows=int(len(G["task"])), steps_median=float(np.median(np.bincount(G["demo"] - 100))), attempts_detail=att,
                   sha256=hashlib.sha256(open(fn, "rb").read()).hexdigest(), generated=True,
                   rule="W (COVERAGE_RESULTS 1.3)", filter="LIBERO success predicate within 300 steps; layout clearance + settle")
        with open(os.path.join(OUT, f"pp_cov_gen_{task}.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        summ["tasks"][task] = {x: rep[x] for x in ("kept", "attempts", "layouts_accepted", "planner_success", "rows", "steps_median", "sha256")}
        print(f"[pp_cov merge] {task}: {json.dumps(summ['tasks'][task])}", flush=True)
    with open(os.path.join(OUT, "pp_cov_gen.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)


# ── stats (rule checks and support table) ──────────────────────────────────────────────────────────────────────────────────────
def pct(v, q=(10, 50, 90, 100)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return np.percentile(v, q).round(4).tolist() if len(v) else None


def support(b, sel):
    """The support table of COVERAGE_RESULTS 1.2 on the rows `sel` (bool over b's rows): object start offsets per object atom and task;
    per target atom the carries' onset d0, onset height h0, bearing to the target's yaw, release d and h (last held row of each demo)."""
    import torch
    from pp_rack_diag import foot_rel, wrap
    key = b["dkey"].numpy()
    oi, ti = b["oi"].numpy(), b["ti"].numpy()
    held = (b["held"].numpy() > 0.5) & sel
    idx = np.arange(len(key))
    d_ho = np.linalg.norm((b["ee_pos"].numpy() - b["pos"].numpy()[idx, oi])[:, :2], axis=1)
    d, h, _ = foot_rel(b)
    tyaw = b["yaw"][torch.arange(len(ti)), b["ti"]].numpy()
    cy = b["cyaw"].numpy()
    first = np.r_[True, key[1:] != key[:-1]] & sel
    obj = {}
    for o in np.unique(oi[first]):
        m = first & (oi == o)
        obj[FC.ENT[o]] = dict(all=pct(d_ho[m]), by_task={f"K{t + 1}": pct(d_ho[m & (key // 1000 == t)]) for t in np.unique(key[m] // 1000)})
    seg_on = held & np.r_[True, (~held[:-1]) | (key[1:] != key[:-1])]
    last = np.zeros(len(key), bool)
    seen = set()
    for i in np.nonzero(held)[0][::-1]:
        if int(key[i]) not in seen:
            seen.add(int(key[i]))
            last[i] = True
    tgt = {}
    for t in np.unique(ti[held]):
        s = np.nonzero(seg_on & (ti == t))[0]
        e = np.nonzero(last & (ti == t))[0]
        tgt[FC.ENT[t]] = dict(n_carries=int(len(s)), d0=pct(d[s]), h0=pct(h[s], (10, 50, 90)),
                              bearing=pct([math.degrees(wrap(cy[i] - tyaw[i])) for i in s], (0, 10, 50, 90, 100)),
                              release_d=pct(d[e], (10, 50, 90)), release_h=pct(h[e], (10, 50, 90)))
    return dict(objects=obj, targets=tgt)


def stats():
    import torch
    import pp_cov_train as CT
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    c = n2["constraints"]
    lo, hi = np.array(c["box"][0]), np.array(c["box"][1])
    centre = np.array(n2["training"]["centre"])
    mov = n2["movable"]
    disc = math.pi * c["own_min"] ** 2 / float(np.prod(hi - lo))
    out = dict(rule_checks={}, placements={}, support={})
    summ = json.load(open(os.path.join(OUT, "pp_cov_gen.json"), encoding="utf-8"))
    for task in PP:
        r = json.load(open(os.path.join(OUT, f"pp_cov_gen_{task}.json"), encoding="utf-8"))
        att = r["attempts_detail"]
        lay = [a for a in att if a.get("layout")]
        kept = [a for a in lay if a["success"]]
        for a in att:
            assert a["seed"] == seed_of(task, a["k"])
        P_att, P_kept = np.array([a["positions"] for a in lay]), np.array([a["positions"] for a in kept])
        inbox = bool(((P_att >= lo - 1e-9) & (P_att <= hi + 1e-9)).all())
        out["rule_checks"][task] = dict(n_kept=len(kept), n_attempts=len(att), n_layouts=len(lay), all_in_box=inbox,
                                        seeds_in_W=True, seeds_disjoint=True, kept_equals_n=len(kept) == summ["n_per_task"])
        out["placements"][task] = {mov[i]: dict(near_own_attempted=float(np.mean(np.linalg.norm(P_att[:, i] - centre[i], axis=1) < c["own_min"])),
                                                near_own_kept=float(np.mean(np.linalg.norm(P_kept[:, i] - centre[i], axis=1) < c["own_min"])),
                                                disc_area_share=round(disc, 4),
                                                x_pct_kept=pct(P_kept[:, i, 0], (10, 50, 90)), y_pct_kept=pct(P_kept[:, i, 1], (10, 50, 90)),
                                                x_pct_attempted=pct(P_att[:, i, 0], (10, 50, 90)), y_pct_attempted=pct(P_att[:, i, 1], (10, 50, 90)))
                                   for i in range(len(mov))}
    out["rule_checks"]["equal_count"] = len({v["n_kept"] for k, v in out["rule_checks"].items() if k in PP}) == 1
    b, X, _, _, n, _ = CT.rows_cov(torch.device("cpu"), list(FC.KNOWN), gen=PP)
    g = (b["dkey"].numpy() % 1000) >= 100
    out["rows"] = dict(total=int(n), generated=int(g.sum()), human=int((~g).sum()), generated_share=float(g.mean()))
    out["support"]["human"] = support(b, ~g)
    out["support"]["generated"] = support(b, g)
    json.dump(out, open(os.path.join(OUT, "pp_cov_gen_stats.json"), "w", encoding="utf-8"), indent=1)
    print("[pp_cov stats] rows", json.dumps(out["rows"]), flush=True)
    for k, v in out["rule_checks"].items():
        print("[pp_cov stats] check", k, json.dumps(v), flush=True)
    for task, v in out["placements"].items():
        for o, w in v.items():
            print("[pp_cov stats] place", task, o, json.dumps(w), flush=True)
    for src in ("human", "generated"):
        for o, w in out["support"][src]["objects"].items():
            print("[pp_cov stats] start", src, o, json.dumps(w), flush=True)
        for t, w in out["support"][src]["targets"].items():
            print("[pp_cov stats] carry", src, t, json.dumps(w), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("shard", "merge", "stats"))
    ap.add_argument("--task", default="K1")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=40)
    a = ap.parse_args()
    if a.mode == "shard":
        assert a.task in PP and 0 <= a.lo < a.hi <= CAP
        shard(a.task, a.lo, a.hi)
    elif a.mode == "merge":
        merge()
    else:
        stats()

"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): the generated unit set.  Chunks env, CPU.  Reads the rendered layouts
(results/bind/b1_3/frames/{train,test}/*.npz, b1_3_render.py) and the frozen plan (plan.json, b1_3_plan.py).

Per frame, b1_3_gen.units_of_frame lists every UNAMBIGUOUS referring unit from the settled simulator positions (LIBERO's mirrored
convention).  A unit is VALID when its target is visible and resolvable under S1's hit rule (some token where the target's
coverage is > 0 and >= every competitor's).
  TRAINING POOL  every valid unit of a training layout whose tuple is a training tuple (no held-out pair, not a test tuple) and
                 that is NOT a held-out LIBERO task's scene + phrase combination (same family, the task scene's types all present,
                 same role phrase).  Its REFERENT for the unsupervised binder is SCRIPTED: the projected world point where a
                 replay-free grasp / release would put the gripper (object: body + 2 cm; goal on a body: body + 5 cm; a region
                 goal / a drawer: the region site's centre) -> its agentview token, the token form of b1_demos.py's contact-event
                 referents (no demo exists for a generated layout).  Sim labels (targets) feed the holistic comparator (B1b-style,
                 trained on labels as S1 did) and scoring only.
  TEST           per test tuple, the first TEST_LAYOUTS of its TEST_ATTEMPTS layouts on which the tuple's unit is valid.
Then: atom divergence D_A(train || test) (alpha 0.5, unit atom counts); if > AD_MAX the training pool is thinned (units whose
atoms are over-represented relative to the test set are dropped, 1% at a time, deterministic) until it is not; every test atom
must occur >= 2 times in the pool.  Each test unit's unseen-compound fraction u is recomputed against the pairs the FINAL pool
realises, and its bin is the realised one (c_split's grading); a bin's CD = D_C(train || bin) (alpha 0.1).

    python experiments/b1_3_units.py  -> results/bind/b1_3/units.json
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
B13 = os.path.join(ROOT, "results", "bind", "b1_3")
TABLE_TYPES = {"kitchen_table", "living_room_table", "study_table", "table", "main_table"}


def main():
    import b1_3_gen as G
    import s1_read as SR
    from lifelong_ood.policy import where as W
    plan = json.load(open(os.path.join(B13, "plan.json"), encoding="utf-8"))
    sp = plan["split"]
    held = {tuple(p) for p in sp["held_pairs"]}
    test_t = {tuple(t["tuple"]): t for t in sp["test"]}
    train_t = {tuple(t) for t in sp["train_tuples"]}
    ho = plan["heldout_libero"]

    def valid(names, cov, bodies, tg):
        if tg not in names:
            return False
        t = cov[names.index(tg)]
        if t.sum() < 0.5:
            return False
        comp, _ = SR.competitors(tg, names, bodies)
        cc = [cov[names.index(c)] for c in comp]
        return any(SR.hit(int(k), t, cc) for k in np.flatnonzero(t > 0))

    train, test_by, n_frames, purity = [], collections.defaultdict(list), collections.Counter(), []
    overlap, phrase_overlap = collections.Counter(), collections.Counter()
    ho_phr = {(r, p) for h in ho for r in ("object", "goal") for p in h["phrases"][r]}
    dropped = collections.Counter()
    for set_name in ("train", "test"):
        for f in sorted(glob.glob(os.path.join(B13, "frames", set_name, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
            names, cov = SR.masks(z)
            bodies = [str(b) for b in z["body_names"]]
            n_frames[set_name] += 1
            rel = os.path.relpath(f, ROOT)
            K, E = z["K"], z["E"]
            us = G.units_of_frame(meta, names, meta["sites"])
            types = {o["type"] for o in meta["objects"]} | {x["type"] for x in meta["fixtures"]}
            for u in us:
                tp = tuple(u["tuple"])
                if not valid(names, cov, bodies, u["target"]):
                    dropped["invalid_" + set_name] += 1
                    continue
                rc = W.project(np.asarray(u["ref"], np.float64), K, E)
                tok = int(np.clip(rc[0] // 16, 0, 15)) * 16 + int(np.clip(rc[1] // 16, 0, 15))
                row = dict(frame=rel, layout=meta["id"], family=meta["family"], role=tp[0], phrase=u["phrase"], targets=[u["target"]],
                           token=tok, tuple=list(tp), depth=G.depth(tp), atoms=G.atoms(tp))
                if set_name == "train":
                    if tp not in train_t:
                        dropped["not_training_tuple"] += 1
                        continue
                    hit_ho = [h["file"] for h in ho if h["family"] == meta["family"] and set(h["types"]) - TABLE_TYPES <= types
                              and u["phrase"] in h["phrases"][tp[0]]]
                    if hit_ho:
                        overlap[u["phrase"]] += 1
                        dropped["heldout_libero_scene_phrase"] += 1
                        continue
                    if (tp[0], u["phrase"]) in ho_phr:
                        phrase_overlap[u["phrase"]] += 1
                    purity.append(float(cov[names.index(u["target"])][tok] > 0))
                    train.append(row)
                elif tp == tuple(meta["tuple"]):
                    row.update(bin_plan=meta["bin"], attempt=meta["attempt"], compound="|".join(str(x) for x in tp))
                    test_by[row["compound"]].append(row)
    test = []
    for c, rows in sorted(test_by.items()):
        rows = sorted(rows, key=lambda r: r["attempt"])[:G.TEST_LAYOUTS]
        test.extend(rows)
    missing = sorted({"|".join(str(x) for x in t) for t in test_t} - set(test_by))

    # atom divergence and thinning
    def adist(rows):
        c = collections.Counter(a for r in rows for a in r["atoms"])
        n = sum(c.values())
        return {k: v / n for k, v in c.items()}
    q = adist(test)
    da0 = 1 - G.chernoff(adist(train), q, G.ALPHA_A)
    pool = list(train)
    thin = 0
    while 1 - G.chernoff(adist(pool), q, G.ALPHA_A) > G.AD_MAX * 0.9 and len(pool) > 0.4 * len(train):
        p = adist(pool)
        ratio = {a: np.log(p[a] / q.get(a, 1e-9)) for a in p}
        sc = np.array([np.mean([ratio[a] for a in r["atoms"]]) for r in pool])
        k = max(1, len(pool) // 100)
        drop = set(np.argsort(-sc, kind="stable")[:k].tolist())
        pool = [r for i, r in enumerate(pool) if i not in drop]
        thin += k
    da = 1 - G.chernoff(adist(pool), q, G.ALPHA_A)
    acount = collections.Counter(a for r in pool for a in r["atoms"])
    test_atoms = sorted({a for r in test for a in r["atoms"]})
    low = {a: acount[a] for a in test_atoms if acount[a] < 2}
    trc = {tuple(sorted(c)) for r in pool for c in G.compounds(tuple(r["tuple"]))}
    for r in test:
        r["u"] = G.u_of(tuple(r["tuple"]), trc)
        r["bin"] = G.grade(r["u"])
    rebinned = sorted({r["compound"] for r in test if r["bin"] != r["bin_plan"]})

    def cdist(rows):
        c = collections.Counter(tuple(sorted(x)) for r in rows for x in G.compounds(tuple(r["tuple"])))
        n = sum(c.values())
        return {k: v / n for k, v in c.items()}
    ptr = cdist(pool)
    bin_cd = {b: 1 - G.chernoff(ptr, cdist([r for r in test if r["bin"] == b]), G.ALPHA_C) for b in G.BINS if any(r["bin"] == b for r in test)}
    for r in test:
        r["cd_unit"] = 1 - G.chernoff(ptr, cdist([r]), G.ALPHA_C)
    cells = collections.Counter((r["depth"], r["bin"]) for r in test)
    ccells = {f"d{d}_{b}": len({r["compound"] for r in test if r["depth"] == d and r["bin"] == b}) for d in (0, 1, 2) for b in G.BINS}
    croles = {f"d{d}_{b}_{ro}": len({r["compound"] for r in test if r["depth"] == d and r["bin"] == b and r["role"] == ro})
              for d in (0, 1, 2) for b in G.BINS for ro in ("object", "goal")}
    stats = dict(frames=dict(n_frames), n_train_units_before_thin=len(train), n_train_units=len(pool), thinned=thin,
                 n_train_layouts=len({r["layout"] for r in pool}), n_test_units=len(test), n_test_compounds=len({r["compound"] for r in test}),
                 d_atom_before_thin=da0, d_atom=da, test_atoms_below_2=low, min_test_atom_count=min(acount[a] for a in test_atoms),
                 bin_cd=bin_cd, cells_units={f"d{d}_{b}": n for (d, b), n in sorted(cells.items())}, cells_compounds=ccells,
                 cells_compounds_by_role=croles, rebinned_compounds=rebinned, missing_test_compounds=missing, dropped=dict(dropped),
                 heldout_libero_scene_phrase_excluded=dict(overlap), heldout_libero_scene_phrase_in_pool=0,
                 heldout_phrase_only_in_pool=dict(phrase_overlap), referent_purity=float(np.mean(purity)) if purity else None,
                 train_depth=dict(collections.Counter(r["depth"] for r in pool)), train_role=dict(collections.Counter(r["role"] for r in pool)),
                 n_train_phrases=len({(r["role"], r["phrase"]) for r in pool}))
    assert not low, low
    out = dict(train=pool, test=test, stats=stats, held_pairs=sp["held_pairs"])
    json.dump(out, open(os.path.join(B13, "units.json"), "w", encoding="utf-8"), indent=1)
    print("[b1.3 units] " + json.dumps(stats, indent=1, default=str)[:6000], flush=True)


if __name__ == "__main__":
    main()

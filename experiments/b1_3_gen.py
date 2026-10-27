"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): the procedural LIBERO layout generator and its MCD split over the binder's
own atoms.  Pure python + numpy (py3.8-compatible: imported by the LIBERO-venv renderer b1_3_render.py and by the chunks-env
scripts b1_3_units.py / b1_3_bind.py / b1_3_read.py).

ASSETS.  LIBERO-90's object / fixture types (BDDL), each with its LIBERO phrase (labels.json type_phrases, s1_labels.GOAL_T) and
LIBERO's own yaw / placement conventions (fixtures at the slots LIBERO-90 uses, facing the robot).  Families = the four tabletop
problem classes (kitchen, living room, study, plain tabletop): the table / wall appearance is the "scene" atom.

ATOMS of a referring unit (role, phrase):  r:<role>  s:<family>  t:<head type phrase>  q:<qualifier>  p:<part noun>
  depth 0  r s t          "ketchup" / "on the plate"
  depth 1  r s t q        "left black bowl" / "black bowl at the front" / "on the plate on the right"  (q in left right front back middle)
  depth 2  r s t q p      "in the top drawer of the white cabinet" / "on top of the shelf" (q top, p side) /
                          "on the bottom level of the shelf" / "in the left compartment of the caddy" / "top drawer of the wooden cabinet"
COMPOUNDS = every unordered pair of a unit's atoms (3 / 6 / 10 pairs).
CONVENTION (LIBERO's, docs/PLAN_BINDER.md 6.1, MIRRORED w.r.t. the policy image): "left" instance = smallest world y, "right" =
largest y, "front" = largest x, "back" = smallest x, "middle" = the median in x (depth) of three; drawers / levels / caddy
compartments = LIBERO's own site names (top_region, left_contain_region, ...).

MCD SPLIT (plan(); seeded).  A set H of held-out compounds (pairs) and a set of TEST tuples are chosen so that, with TRAINING
tuples = feasible tuples - test tuples - tuples containing a pair of H, each test tuple's unseen-compound fraction u (w.r.t. the
pairs the training tuples realise) bins it as in experiments/c_split.py (u == 0 -> "0.1", 0 < u < 0.4 -> "0.3", u >= 0.4 -> "0.5"),
every test atom is in >= 3 training tuples, and the depth-matched atom divergence D_A = 1 - C_0.5 <= AD_MAX.  Every test tuple
(even bin 0.1) is a held-out combination at the tuple level; pairs with r and s together are never held out.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import random

import numpy as np

SEED = 20261027
AD_MAX = 0.02
ALPHA_A, ALPHA_C = 0.5, 0.1
BINS = ("0.1", "0.3", "0.5")
CELL_TARGET = {0: 10, 1: 10, 2: 8}       # depth 2: the pre-agreed fallback minimum (9 failed the search: plan jobs 4289735-4290381)
ORDER = (("0.5", 2), ("0.1", 2), ("0.3", 2), ("0.1", 1), ("0.1", 0), ("0.3", 1), ("0.3", 0), ("0.5", 1), ("0.5", 0))
TEST_LAYOUTS, TEST_ATTEMPTS, N_TRAIN_LAYOUTS = 3, 5, 720
FAMILIES = {"kitchen": ("LIBERO_Kitchen_Tabletop_Manipulation", "kitchen_table", "kitchen_table"),
            "living": ("LIBERO_Living_Room_Tabletop_Manipulation", "living_room_table", "living_room_table"),
            "study": ("LIBERO_Study_Tabletop_Manipulation", "study_table", "study_table"),
            "tabletop": ("LIBERO_Tabletop_Manipulation", "main_table", "table")}
# movable type -> (phrase, footprint radius m, may have several instances, goal preposition or None, goal target: body / site suffix)
MOV = {"akita_black_bowl": ("black bowl", 0.06, True, "on", None),
       "white_bowl": ("white bowl", 0.06, True, "on", None),
       "plate": ("plate", 0.10, True, "on", None),
       "ketchup": ("ketchup", 0.04, True, None, None),
       "tomato_sauce": ("tomato sauce", 0.04, True, None, None),
       "alphabet_soup": ("alphabet soup", 0.04, True, None, None),
       "cream_cheese": ("cream cheese box", 0.045, False, None, None),
       "butter": ("butter", 0.045, True, None, None),
       "milk": ("milk", 0.04, False, None, None),
       "orange_juice": ("orange juice", 0.04, False, None, None),
       "chocolate_pudding": ("chocolate pudding", 0.045, False, None, None),
       "new_salad_dressing": ("salad dressing", 0.04, False, None, None),
       "wine_bottle": ("wine bottle", 0.045, False, None, None),
       "moka_pot": ("moka pot", 0.06, True, None, None),
       "chefmate_8_frypan": ("frying pan", 0.12, False, None, None),
       "red_coffee_mug": ("red mug", 0.05, True, None, None),
       "white_yellow_mug": ("yellow and white mug", 0.05, False, None, None),
       "porcelain_mug": ("white mug", 0.05, True, None, None),
       "black_book": ("book", 0.07, True, None, None),
       "basket": ("basket", 0.12, False, "in", "contain_region"),
       "wooden_tray": ("tray", 0.13, False, "in", "contain_region")}
NOT_OBJECT_ROLE = ("basket", "wooden_tray")
YAW = {"black_book": (-1.5707963267948966, -0.7853981633974483)}
# fixture type -> (phrase, footprint radius, slots [(x, y, yaw)], parts [(q, p, site suffix, roles)], d0 goal (prep, site) or None)
PI = math.pi
DRAWERS = [("top", "drawer", "top_region", ("goal", "object")), ("middle", "drawer", "middle_region", ("goal", "object")),
           ("bottom", "drawer", "bottom_region", ("goal", "object")), ("top", "side", "top_side", ("goal",))]
FIX = {"wooden_cabinet": ("wooden cabinet", 0.13, [(0.0, -0.30, PI), (0.0, 0.30, 0.0)], DRAWERS, None),
       "white_cabinet": ("white cabinet", 0.13, [(0.0, 0.30, 0.0), (0.0, -0.30, PI)], DRAWERS, None),
       "wooden_two_layer_shelf": ("shelf", 0.13, [(0.0, -0.26, PI), (0.0, 0.28, 0.0)],
                                  [("top", "level", "top_region", ("goal",)), ("bottom", "level", "bottom_region", ("goal",)),
                                   ("top", "side", "top_side", ("goal",))], None),
       "microwave": ("microwave", 0.13, [(0.0, -0.26, PI), (0.0, 0.35, 0.0)], [("top", "side", "top_side", ("goal",))], None),
       "desk_caddy": ("caddy", 0.10, [(-0.20, -0.14, PI), (-0.20, 0.10, PI), (-0.20, -0.02, PI)],
                      [("left", "compartment", "left_contain_region", ("goal",)), ("right", "compartment", "right_contain_region", ("goal",)),
                       ("front", "compartment", "front_contain_region", ("goal",)), ("back", "compartment", "back_contain_region", ("goal",))], None),
       "flat_stove": ("stove", 0.11, [(-0.20, 0.20, 0.0), (-0.20, -0.20, 0.0), (-0.20, 0.30, 0.0)], [], ("on", "cook_region")),
       "wine_rack": ("wine rack", 0.10, [(-0.10, -0.30, PI), (-0.10, 0.30, 0.0)], [], ("on", "top_region"))}
SIDE_FIX = ("wooden_cabinet", "white_cabinet", "wooden_two_layer_shelf", "microwave")
QUALS = ("left", "right", "front", "back", "middle")
PART_TEMPLATE = {"drawer": "in the {q} drawer of the {t}", "side": "on top of the {t}", "level": "on the {q} level of the {t}",
                 "compartment": "in the {q} compartment of the {t}"}
X_RANGE, Y_RANGE = (-0.18, 0.17), (-0.24, 0.24)
MID_MAX_R = 0.07                  # "middle" (three in a depth column) only for types this small
LAT_GAP, DEP_GAP, MID_GAP = 0.10, 0.10, 0.08


def phrase_of_type(t):
    return MOV[t][0] if t in MOV else FIX[t][0]


def receptacles():
    """d0 goal head types -> (prep, site suffix or None)."""
    out = {t: (v[3], v[4]) for t, v in MOV.items() if v[3]}
    out.update({t: v[4] for t, v in FIX.items() if v[4]})
    return out


def multi_receptacles():
    return [t for t, v in MOV.items() if v[3] and v[2]]


# ── tuples, atoms, compounds ─────────────────────────────────────────────────────────────────────────────────────────
def feasible_tuples():
    """every (role, family, head type, q, p) the generator can realise; t is the ASSET type (atoms use its phrase)."""
    out = []
    for f in FAMILIES:
        for t, v in MOV.items():
            if t not in NOT_OBJECT_ROLE:
                out.append(("object", f, t, None, None))
                if v[2]:
                    for q in QUALS:
                        if q != "middle" or v[1] <= MID_MAX_R:
                            out.append(("object", f, t, q, None))
        for t in receptacles():
            out.append(("goal", f, t, None, None))
        for t in multi_receptacles():
            for q in QUALS:
                if q != "middle" or MOV[t][1] <= MID_MAX_R:
                    out.append(("goal", f, t, q, None))
        for t, v in FIX.items():
            for q, p, _, roles in v[3]:
                for r in roles:
                    out.append((r, f, t, q, p))
    return out


def depth(tp):
    return 2 if tp[4] else (1 if tp[3] else 0)


def atoms(tp):
    r, f, t, q, p = tp
    a = ["r:" + r, "s:" + f, "t:" + phrase_of_type(t)]
    if q:
        a.append("q:" + q)
    if p:
        a.append("p:" + p)
    return a


def compounds(tp):
    return [tuple(sorted(c)) for c in itertools.combinations(atoms(tp), 2)]


def holdable(pair):
    k = {x.split(":")[0] for x in pair}
    return k != {"r", "s"}


def dist(counters):
    tot = collections.Counter()
    for c in counters:
        tot.update(c)
    n = sum(tot.values())
    return {k: v / n for k, v in tot.items()} if n else {}


def chernoff(p, q, a):
    return sum(p[k] ** a * q[k] ** (1 - a) for k in p.keys() & q.keys())


def weighted_dist(tuples, weights):
    tot = collections.Counter()
    for tp, w in zip(tuples, weights):
        for a in atoms(tp):
            tot[a] += w
    n = sum(tot.values())
    return {k: v / n for k, v in tot.items()}


def depth_weights(tuples, shares=None):
    """weights that give each depth the share it has in `shares` (default: equal), uniform within a depth."""
    by = collections.Counter(depth(t) for t in tuples)
    shares = shares or {d: 1.0 / len(by) for d in by}
    return [shares.get(depth(t), 0.0) / by[depth(t)] for t in tuples]


def grade(u):
    return BINS[0] if u <= 0 else (BINS[1] if u < 0.4 else BINS[2])


def split_state(H, test, feas):
    tr = [t for t in feas if t not in test and not any(c in H for c in compounds(t))]
    trc = set(c for t in tr for c in compounds(t))
    acount = collections.Counter(a for t in tr for a in atoms(t))
    return tr, trc, acount


def u_of(tp, trc):
    c = compounds(tp)
    return sum(1 for x in c if x not in trc) / len(c)


def plan_split(seed=SEED, tries=150):
    """Seeded randomized greedy construction with restarts; keeps the restart with the lowest D_A that fills every cell."""
    feas = feasible_tuples()
    pcount = collections.Counter(c for t in feas for c in compounds(t))
    fails = collections.Counter()

    def pick(rng, comp, k):
        """k distinct pairs, preferring SPECIFIC ones (few feasible tuples share them: fewer collateral removals)."""
        comp, out = list(comp), []
        for _ in range(k):
            w = [1.0 / pcount[c] ** 2 for c in comp]
            c = rng.choices(comp, weights=w)[0]
            out.append(c)
            comp.remove(c)
        return out
    best = None
    for rs in range(tries):
        rng = random.Random(seed * 1000 + rs)
        H, test = set(), {}
        ok = True
        # cell order: depth 2 (the smallest tuple space) bin 0.5 first, then its u == 0 and 0.3 cells; then depths 1 and 0 with
        # their u == 0 cells first (later held-out pairs may not break an accepted test tuple's bin: checked below)
        for b, d in ORDER:
            if True:
                pool = [t for t in feas if depth(t) == d]
                need = CELL_TARGET[d]
                got = sum(1 for t, bb in test.items() if bb == b and depth(t) == d)
                roles = collections.Counter(t[0] for t, bb in test.items() if bb == b and depth(t) == d)
                attempts = 0
                while got < need and attempts < 600:
                    attempts += 1
                    cand = [t for t in pool if t not in test]
                    # role balance: prefer the role this cell has fewer of (when that role is feasible at this depth)
                    rr = [r for r in ("object", "goal") if any(t[0] == r for t in cand)]
                    want = min(rr, key=lambda r: roles[r]) if rr else None
                    cw = [t for t in cand if t[0] == want] if want and rng.random() < 0.8 else cand
                    tp = rng.choice(cw)
                    comp = [c for c in compounds(tp) if holdable(c)]
                    if b == "0.1":
                        add = []
                    elif b == "0.3":
                        add = pick(rng, comp, 1)
                    else:                                   # add specific pairs one at a time until u >= 0.4 (removals also unrealise pairs)
                        k = int(math.ceil(0.4 * len(compounds(tp))))
                        add, left_ = [], list(comp)
                        t1 = dict(test)
                        t1[tp] = b
                        while left_ and len(add) < k:
                            add += pick(rng, left_, 1)
                            left_ = [c for c in left_ if c not in add]
                            if u_of(tp, split_state(H | set(add), t1, feas)[1]) >= 0.4:
                                break
                    H2 = H | set(add)
                    t2 = dict(test)
                    t2[tp] = b
                    tr, trc, acount = split_state(H2, t2, feas)
                    if any(acount[a] < 3 for t in t2 for a in atoms(t)):
                        continue
                    if any(grade(u_of(t, trc)) != bb for t, bb in t2.items()):
                        continue
                    H, test = H2, t2
                    got += 1
                    roles[tp[0]] += 1
                if got < need:
                    fails[(d, b)] += 1
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        tr, trc, acount = split_state(H, test, feas)
        tl = list(test)
        da = 1.0 - chernoff(weighted_dist(tr, depth_weights(tr)), weighted_dist(tl, depth_weights(tl)), ALPHA_A)
        if best is None or da < best[0]:
            best = (da, H, test, rs)
        if da <= AD_MAX * 0.6:
            break
    assert best is not None, f"no restart filled every cell: failing cells {dict(fails)}"
    print(f"[b1.3 gen] plan restarts that failed a cell: {dict(fails)}", flush=True)
    da, H, test, rs = best
    tr, trc, acount = split_state(H, test, feas)
    return dict(seed=seed, restart=rs, d_atom_plan=da, held_pairs=sorted([list(c) for c in H]),
                test=[dict(tuple=list(t), bin=b, depth=depth(t), u=u_of(t, trc)) for t, b in sorted(test.items(), key=lambda x: (depth(x[0]), x[1], str(x[0])))],
                train_tuples=[list(t) for t in tr], n_feasible=len(feas))


# ── layouts ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _free(x, y, r, placed):
    if not (X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]):
        return False
    return all((x - px) ** 2 + (y - py) ** 2 >= (r + pr + 0.015) ** 2 for px, py, pr in placed)


def _pattern(q, n, r, rng):
    """offsets for n instances realising qualifier q (LIBERO convention: left = small y, front = large x, middle = depth median)."""
    s = 2 * r + 0.03
    if n == 3:
        sx = max(s, 0.11)
        return [(-sx, rng.uniform(-0.02, 0.02)), (0.0, rng.uniform(-0.02, 0.02)), (sx, rng.uniform(-0.02, 0.02))]
    if q in ("left", "right"):
        sy = max(s, 0.16)
        return [(rng.uniform(-0.03, 0.03), -sy / 2), (rng.uniform(-0.03, 0.03), sy / 2)]
    sx = max(s, 0.16)
    return [(-sx / 2, rng.uniform(-0.03, 0.03)), (sx / 2, rng.uniform(-0.03, 0.03))]


def make_layout(rng, family, need=None, n_extra=None):
    """-> layout dict or None.  need: a tuple the layout must realise (test), else a random training layout."""
    fixtures, objects, placed = [], [], []
    want_fix = []
    if need is not None and need[2] in FIX:
        want_fix.append(need[2])
    nfix = len(want_fix) + (rng.choice([0, 1, 1, 2]) if need is None else rng.choice([0, 0, 1]))
    pool_fix = [f for f in FIX if f not in want_fix]
    rng.shuffle(pool_fix)
    used_slots = []
    for ft in want_fix + pool_fix:
        if len(fixtures) >= nfix:
            break
        if ft in SIDE_FIX and sum(1 for f in fixtures if f["type"] in SIDE_FIX) >= 2:
            continue
        ok = False
        for (x, y, yaw) in rng.sample(FIX[ft][2], len(FIX[ft][2])):
            x2, y2 = x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01)
            if all((x2 - a) ** 2 + (y2 - b) ** 2 >= (FIX[ft][1] + rr + 0.02) ** 2 for a, b, rr in used_slots):
                fixtures.append(dict(name=ft + "_1", type=ft, x=x2, y=y2, yaw=yaw))
                used_slots.append((x2, y2, FIX[ft][1]))
                ok = True
                break
        if not ok and ft in want_fix:
            return None
    placed = list(used_slots)

    def put(t, n, q=None):
        r = MOV[t][1]
        k0 = sum(1 for o in objects if o["type"] == t)
        for _ in range(60):
            cx, cy = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)
            offs = [(0.0, 0.0)] if n == 1 else _pattern(q, n, r, rng)
            pts = [(cx + a, cy + b) for a, b in offs]
            if all(_free(x, y, r, placed) for x, y in pts) and all(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2 >= (2 * r + 0.015) ** 2 for (x1, y1), (x2, y2) in itertools.combinations(pts, 2)):
                for j, (x, y) in enumerate(pts):
                    objects.append(dict(name=f"{t}_{k0 + j + 1}", type=t, x=x, y=y))
                    placed.append((x, y, r))
                return True
        return False
    if need is not None and need[2] in MOV:
        t, q = need[2], need[3]
        n = 1 if q is None else (3 if q == "middle" else (2 if q in ("left", "right") or MOV[t][1] > MID_MAX_R else rng.choice([2, 2, 3])))
        if not put(t, n, q):
            return None
    if need is not None and need[0] == "goal" and need[2] in MOV:
        movers = [t for t in MOV if t not in NOT_OBJECT_ROLE and t != need[2] and MOV[t][1] <= 0.07]
        put(rng.choice(movers), 1)
    n_extra = rng.choice([2, 3, 3, 4]) if n_extra is None else n_extra
    head = need[2] if need is not None else None
    tries = 0
    while sum(1 for _ in objects) < n_extra + (0 if need is None else 1) and tries < 30:
        tries += 1
        t = rng.choice([x for x in MOV if x != head])
        if any(o["type"] == t for o in objects):
            continue
        multi = need is None and MOV[t][2] and rng.random() < 0.45
        n = (3 if rng.random() < 0.3 and MOV[t][1] <= MID_MAX_R else 2) if multi else 1
        put(t, n, rng.choice(("left", "front")) if multi else None)
    if len(objects) == 0:
        return None
    return dict(family=family, fixtures=fixtures, objects=objects)


def bddl_text(L, language="pick up the object"):
    prob, table, ttype = FAMILIES[L["family"]]
    regs, init = [], []
    for o in L["objects"] + L["fixtures"]:
        h = 0.005
        lo, hi = YAW.get(o["type"], (o.get("yaw", 0.0), o.get("yaw", 0.0)))
        regs.append(f"      (r_{o['name']}_init_region\n          (:target {table})\n          (:ranges (\n"
                    f"              ({o['x'] - h:.4f} {o['y'] - h:.4f} {o['x'] + h:.4f} {o['y'] + h:.4f})\n            )\n          )\n"
                    f"          (:yaw_rotation (\n              ({lo} {hi})\n            )\n          )\n      )")
        init.append(f"    (On {o['name']} {table}_r_{o['name']}_init_region)")
    for f in L["fixtures"]:
        for _, _, suf, _ in FIX[f["type"]][3]:
            regs.append(f"      ({suf}\n          (:target {f['name']})\n      )")
        if FIX[f["type"]][4]:
            regs.append(f"      ({FIX[f['type']][4][1]}\n          (:target {f['name']})\n      )")
    for o in L["objects"]:                              # LIBERO keys regions by (target, name): duplicates across targets are fine
        if MOV[o["type"]][4]:
            regs.append(f"      ({MOV[o['type']][4]}\n          (:target {o['name']})\n      )")
    regs2 = regs
    by = collections.OrderedDict()
    for o in L["objects"]:
        by.setdefault(o["type"], []).append(o["name"])
    fx = "\n".join(f"    {f['name']} - {f['type']}" for f in L["fixtures"])
    ob = "\n".join(f"    {' '.join(v)} - {t}" for t, v in by.items())
    first = L["objects"][0]["name"]
    return (f"(define (problem {prob})\n  (:domain robosuite)\n  (:language {language})\n    (:regions\n" + "\n".join(regs2) + "\n    )\n\n"
            f"  (:fixtures\n    {table} - {ttype}\n" + (fx + "\n" if fx else "") + "  )\n\n"
            f"  (:objects\n{ob}\n  )\n\n  (:obj_of_interest\n    {first}\n  )\n\n  (:init\n" + "\n".join(init) + "\n  )\n\n"
            f"  (:goal\n    (And (On {first} {table}_r_{first}_init_region))\n  )\n\n)\n")


def plan_layouts(split, seed=SEED):
    rng = random.Random(seed + 7)
    out = []
    fams = list(FAMILIES)
    k = 0
    while sum(1 for x in out if x["set"] == "train") < N_TRAIN_LAYOUTS:
        L = make_layout(rng, fams[k % len(fams)])
        k += 1
        if L:
            L.update(set="train", id=f"tr{len(out):04d}")
            out.append(L)
    for i, te in enumerate(split["test"]):
        tp = tuple(te["tuple"])
        for j in range(TEST_ATTEMPTS):
            for _ in range(50):
                L = make_layout(rng, tp[1], need=tp)
                if L:
                    break
            assert L, tp
            L.update(set="test", id=f"te{i:03d}_{j}", tuple=list(tp), bin=te["bin"], attempt=j)
            out.append(L)
    return out


# ── units from a rendered frame (positions after the settle; sim labels; scoring + scripted referents) ──────────────────
def _qualifiers(pos):
    """pos: [(name, x, y)] of one type's instances -> {name: [q]} under the LIBERO convention with margins."""
    out = {n: [] for n, _, _ in pos}
    if len(pos) < 2:
        return out
    ys = sorted(pos, key=lambda p: p[2])
    xs = sorted(pos, key=lambda p: p[1])
    if ys[1][2] - ys[0][2] >= LAT_GAP:
        out[ys[0][0]].append("left")
    if ys[-1][2] - ys[-2][2] >= LAT_GAP:
        out[ys[-1][0]].append("right")
    if xs[-1][1] - xs[-2][1] >= DEP_GAP:
        out[xs[-1][0]].append("front")
    if xs[1][1] - xs[0][1] >= DEP_GAP:
        out[xs[0][0]].append("back")
    if len(pos) == 3 and xs[1][1] - xs[0][1] >= MID_GAP and xs[2][1] - xs[1][1] >= MID_GAP:
        out[xs[1][0]].append("middle")
    return out


def surface(tp, key):
    """the unit's phrase (normalised, no leading 'the'); key picks the d1 template (pre- or post-nominal) deterministically."""
    r, f, t, q, p = tp
    T = phrase_of_type(t)
    if p:
        s = PART_TEMPLATE[p].format(q=q, t=T)
        return s if r == "goal" else s[len("in the "):]
    post = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 1
    if q:
        np_ = (f"{T} " + ({"left": "on the left", "right": "on the right", "front": "at the front", "back": "at the back",
                            "middle": "in the middle"}[q])) if post else f"{q} {T}"
    else:
        np_ = T
    if r == "object":
        return np_
    prep = MOV[t][3] if t in MOV else FIX[t][4][0]
    return f"{prep} the {np_}"


def units_of_frame(meta, names, sites):
    """meta: the rendered frame's layout + positions; -> [dict(tuple, phrase, target, referent_xyz)] of every UNAMBIGUOUS unit."""
    pos = meta["pos"]
    out = []
    by = collections.defaultdict(list)
    for o in meta["objects"]:
        if o["name"] in pos:
            by[o["type"]].append((o["name"], pos[o["name"]][0], pos[o["name"]][1]))
    fam = meta["family"]
    for t, inst in by.items():
        quals = _qualifiers(inst)
        recept = MOV[t][3]
        for n, x, y in inst:
            p3 = pos[n]
            gsite = None
            if MOV[t][4]:
                gsite = n + "_" + MOV[t][4]
            gtarget = gsite if gsite else n
            gref = sites.get(gsite) if gsite else [p3[0], p3[1], p3[2] + 0.05]
            qs = [None] if len(inst) == 1 else quals[n]
            for q in qs:
                if t not in NOT_OBJECT_ROLE:
                    out.append(dict(tuple=("object", fam, t, q, None), target=n, ref=[p3[0], p3[1], p3[2] + 0.02]))
                if recept and gref is not None and (q is None or MOV[t][2]):
                    out.append(dict(tuple=("goal", fam, t, q, None), target=gtarget, ref=list(gref)))
    for f in meta["fixtures"]:
        fn = f["name"]
        for q, p, suf, roles in FIX[f["type"]][3]:
            sn = fn + "_" + suf
            if sn in sites:
                for r in roles:
                    out.append(dict(tuple=(r, fam, f["type"], q, p), target=sn, ref=list(sites[sn])))
        if FIX[f["type"]][4]:
            sn = fn + "_" + FIX[f["type"]][4][1]
            if sn in sites:
                out.append(dict(tuple=("goal", fam, f["type"], None, None), target=sn, ref=list(sites[sn])))
    for u in out:
        u["phrase"] = surface(u["tuple"], meta["id"] + "|" + "|".join(str(x) for x in u["tuple"]))
    return out


if __name__ == "__main__":
    sp = plan_split()
    print(json.dumps({k: v for k, v in sp.items() if k not in ("train_tuples", "test")}, indent=1)[:2000])
    print(collections.Counter((t["depth"], t["bin"]) for t in sp["test"]))

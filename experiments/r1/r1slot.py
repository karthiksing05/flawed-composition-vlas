"""Gate 0 of PLAN_PARTWISE_POLICY: slot attribution + t=0 nearest-neighbour routing on stored episodes. No training, no rollouts.
Preregistration: results/r1/stream/PREDICTIONS_slot.md.   run: r1slot.py gate0"""
import json, os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1s
from r1s import *  # noqa (np, torch, r1, b1, load_cell, TASKS, GROUPS, newvals, all_cells, dropped, sources, draw, S0)

SLOTS = {"object": (0, 7), "obstacle": (14, 21), "goal": (28, 35), "proprio": (45, 77)}
OWN = {"HB": "object", "OW": "obstacle", "TC": "goal"}   # labels: scoring only
B, ALPHA, NULL_SEED = 1000, 0.01, 20261124


def t0(cell):
    o, _ = load_cell(cell)
    return o[:, 0, :].astype(np.float64)


def feat(x, s):
    lo, hi = SLOTS[s]
    return x[:, lo:hi]


def edist_sum(D):  # sum of a distance matrix block
    return float(D.sum())


def energy(X, Y, cross, dxx_mean, dyy_mean):
    return 2 * cross - dxx_mean - dyy_mean


def pd(a, b):
    return torch.cdist(torch.as_tensor(a), torch.as_tensor(b)).numpy()


def attribution(bank, cell_id, drawn, rng):
    """bank: dict slot -> standardised (N,d); cell_id (N,) cell index; drawn: dict task -> dict slot -> standardised (16,d).
    Returns per slot: null array, per task observed stat and p."""
    out = {}
    ncell = int(cell_id.max()) + 1
    for s, Xb in bank.items():
        D = pd(Xb, Xb).astype(np.float64); N = len(Xb); S = D.sum(); rows = D.sum(1)
        cell_rows = [np.where(cell_id == c)[0] for c in range(ncell)]
        null = np.empty(B)
        for b in range(B):
            cs = rng.choice(ncell, 3, replace=False)
            per = [6, 5, 5]
            idx = np.concatenate([rng.choice(cell_rows[c], k, replace=False) for c, k in zip(cs, per)])
            left = np.concatenate([cell_rows[c] for c in cs])
            nr = N - len(left)   # exact sums over rest = bank minus the three left-out cells, without gathering rest x rest
            cross = (rows[idx].sum() - D[np.ix_(idx, left)].sum()) / (len(idx) * nr)
            dxx = D[np.ix_(idx, idx)].sum() / (len(idx) * (len(idx) - 1))
            sub = (S - 2 * rows[left].sum() + D[np.ix_(left, left)].sum()) / (nr * nr)
            null[b] = 2 * cross - dxx - sub
        res = {}
        for k, Xs in drawn.items():
            Xt = Xs[s]
            cross = pd(Xt, Xb).mean()
            dxx = pd(Xt, Xt).sum() / (len(Xt) * (len(Xt) - 1))
            stat = 2 * cross - dxx - S / (N * N)
            p = (1 + (null >= stat).sum()) / (B + 1)
            res[k] = dict(stat=float(stat), p=float(p), clear=bool(p < ALPHA), null95=float(np.quantile(null, .95)), null99=float(np.quantile(null, .99)))
        out[s] = dict(null_mean=float(null.mean()), null_max=float(null.max()), tasks=res)
    return out


def cmd_gate0(a):
    sp = r1.make_split(0); dr = dropped()
    cells = [c for c in sp["train"] if c not in dr]
    X0, cid = [], []
    for i, c in enumerate(cells):
        x = t0(c); X0.append(x); cid.append(np.full(len(x), i))
        obst0 = np.abs(x[:, 14:21]).max()
        if (obst0 == 0) != (c[2] == "None"):
            print("LAYOUT obstacle-zero check failed", c, "absmax per dim 14-28:", np.round(np.abs(x[:, 14:28]).max(0), 4).tolist(), flush=True)
    X0 = np.concatenate(X0); cid = np.concatenate(cid)
    print("bank", X0.shape, "cells", len(cells), flush=True)
    mu = {s: feat(X0, s).mean(0) for s in SLOTS}; sd = {s: np.maximum(feat(X0, s).std(0), 1e-3) for s in SLOTS}
    z = lambda x, s: (feat(x, s) - mu[s]) / sd[s]
    bank = {s: z(X0, s) for s in SLOTS}
    layout = {}
    for c in r1s.all_cells():
        if c in dr: continue
        x = t0(c); layout[b1.cell_name(c)] = bool(np.abs(x[:, 14:21]).max() == 0)
    # also check the layout on cells by label
    lay_ok = all(layout[b1.cell_name(c)] == (c[2] == "None") for c in r1s.all_cells() if c not in dr)
    print("layout obstacle-zero iff None, all cells:", lay_ok, flush=True)
    for nm, sel in (("None", [c for c in cells if c[2] == "None"][:1]), ("GoalWall", [c for c in cells if c[2] == "GoalWall"][:1]),
                    ("ObjectDoor", [c for c in cells if c[2] == "ObjectDoor"][:1])):
        x = t0(sel[0]); print("obstacle dims 14-28 mean/std", nm, np.round(x[:, 14:28].mean(0), 3).tolist(), np.round(x[:, 14:28].std(0), 4).tolist(), flush=True)
    rng = np.random.RandomState(NULL_SEED)
    result = dict(layout_ok=lay_ok, seeds={}, params=dict(B=B, alpha=ALPHA, null_seed=NULL_SEED, slots=SLOTS, bank_n=int(len(X0))))
    nullcache = None
    correct = 0; tot = 0
    for seed in (0, 1, 2):
        sp_, src = sources(seed)
        drw = {}; keyidx = {}
        rs = np.random.RandomState(seed)
        for k in TASKS:   # same call order as r1s.draw
            per = [16 // 3 + (1 if i < 16 % 3 else 0) for i in range(3)]
            xs, ids = [], []
            for c, kk in zip(src[k], per):
                o, _ = load_cell(c)
                sel = rs.choice(len(o), kk, replace=False)
                xs.append(o[sel][:, 0, :].astype(np.float64)); ids += [(c, int(j)) for j in sel]
            drw[k] = np.concatenate(xs); keyidx[k] = ids
        # equivalence with r1s.draw
        chk = np.random.RandomState(seed)
        for k in TASKS:
            ref = np.concatenate([p[1][:, 0, :] for p in draw(src[k], chk)]).astype(np.float64)
            assert np.allclose(ref, drw[k]), "draw mismatch"
        drawn = {k: {s: z(drw[k], s) for s in SLOTS} for k in TASKS}
        att = attribution(bank, cid, drawn, np.random.RandomState(NULL_SEED))  # same null for every seed (same rng seed)
        owned = {k: [s for s in SLOTS if att[s]["tasks"][k]["clear"]] for k in TASKS}
        verdict = {k: owned[k] == [OWN[k]] for k in TASKS}
        correct += sum(verdict.values()); tot += 3
        print(f"seed {seed} owned {owned} verdict {verdict}", flush=True)
        for s in SLOTS:
            print("  ", s, {k: (round(att[s]['tasks'][k]['stat'], 3), att[s]['tasks'][k]['p']) for k in TASKS}, "null99", round(att[s]['tasks']['HB']['null99'], 3), flush=True)
        # routing
        keys = {k: {s: z(drw[k], s) for s in owned[k]} for k in TASKS}
        slot_bank = {}
        for s in SLOTS:
            F = [bank[s]]; lab = [np.full(len(bank[s]), -1)]
            for i, k in enumerate(TASKS):
                if s in keys[k]:
                    F.append(keys[k][s]); lab.append(np.full(len(keys[k][s]), i))
            slot_bank[s] = (torch.as_tensor(np.concatenate(F)), np.concatenate(lab))
        excl = {}
        for k in TASKS:
            for c, j in keyidx[k]: excl.setdefault(c, set()).add(j)
        cellrep, tot_states, tot_ok = {}, 0, 0
        pp = {k: [0, 0] for k in TASKS}
        for g, cs in list(GROUPS.items()) + [("panel", sp["panel"])]:
            for c in cs:
                if c in dr: continue
                x = t0(c)
                keep = np.array([j not in excl.get(c, ()) for j in range(len(x))])
                x = x[keep]
                on = np.zeros((len(x), 3), bool)
                for i, k in enumerate(TASKS):
                    if not owned[k]: continue
                    ok = np.ones(len(x), bool)
                    for s in owned[k]:
                        Fb, lab = slot_bank[s]
                        nn = torch.cdist(torch.as_tensor(z(x, s)), Fb).argmin(1).numpy()
                        ok &= lab[nn] == i
                    on[:, i] = ok
                want = np.array([t in newvals(c) for t in TASKS])
                match = (on == want).all(1)
                for i, k in enumerate(TASKS):
                    pp[k][0] += int((on[:, i] == want[i]).sum()); pp[k][1] += len(x)
                tot_states += len(x); tot_ok += int(match.sum())
                cellrep[b1.cell_name(c)] = dict(group=g, n=len(x), exact=float(match.mean()), want=newvals(c),
                                                on_rate={k: float(on[:, i].mean()) for i, k in enumerate(TASKS)})
        result["seeds"][str(seed)] = dict(owned=owned, verdict=verdict, att=att, routing_exact=tot_ok / tot_states, n_states=tot_states,
                                          per_part_agree={k: pp[k][0] / pp[k][1] for k in TASKS}, cells=cellrep)
        print(f"seed {seed} routing exact {tot_ok / tot_states:.4f} over {tot_states}; per part {({k: round(pp[k][0] / pp[k][1], 4) for k in TASKS})}", flush=True)
        for n, r in cellrep.items():
            if r["exact"] < 1: print("   mismatch", n, r["group"], "want", r["want"], "exact", round(r["exact"], 3), r["on_rate"], flush=True)
    sd_ = result["seeds"]
    pooled = sum(v["routing_exact"] * v["n_states"] for v in sd_.values()) / sum(v["n_states"] for v in sd_.values())
    result["summary"] = dict(draws_correct=correct, draws=tot, routing_pooled=pooled,
                             routing_per_seed={k: v["routing_exact"] for k, v in sd_.items()},
                             per_part_pooled={t: sum(v["per_part_agree"][t] * v["n_states"] for v in sd_.values()) / sum(v["n_states"] for v in sd_.values()) for t in TASKS})
    p0 = "PASS" if correct == 9 and pooled >= .95 else ("FALSIFIED" if correct < 8 or pooled < .90 else "MARGINAL")
    result["summary"]["P0"] = p0
    print("SUMMARY", json.dumps(result["summary"]), flush=True)
    json.dump(result, open(S0 + "/slot_gate0.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    {"gate0": cmd_gate0}[sys.argv[1]](None)

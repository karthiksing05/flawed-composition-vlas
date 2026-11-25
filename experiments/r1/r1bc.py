"""Idea B gate 1 (PLAN_PARTWISE_POLICY.md section 3 item 2 + P4): explaining away by behaviour cloning on the R1 stream.
For each task's 16 demos, fit one new slot-owning part per hypothesis f in {object, obstacle, goal} (zero-init adapter MLP on that slot's
normalised state slice, added to the state token; base and all else frozen), score held-out flow loss (2-fold over the 16 demos), and
pick the lowest.  Labels only score the pick.  Null = same procedure on 16 demos from held-out panel cells (known values).
  run --seed S      -> results/r1/stream/bc/run_seedS.json
  report            -> results/r1/stream/bc/report_bc.json (verdicts BC-P0, BC-P1, BC-P4, BC-P2)
Registered in results/r1/stream/PREDICTIONS_bc.md."""
import argparse, itertools, json, os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1, r1s
from r1s import *  # noqa  (np, torch, load_cell, Data, load_model, RES, b1, b1_data, TASKS)

SLOTS = {"object": (0, 14), "obstacle": (14, 28), "goal": (28, 45)}
TRUTH = {"HB": "object", "OW": "obstacle", "TC": "goal"}
OUT = RES + "/stream/bc"


class SlotState(torch.nn.Module):
    """state token = orig(state) + adapter(state[slot]); adapter output layer zero-initialised so the start equals the base"""
    def __init__(self, orig, lo, hi, d, hid=64):
        super().__init__()
        self.orig, self.lo, self.hi = orig, lo, hi
        self.a = torch.nn.Sequential(torch.nn.Linear(hi - lo, hid), torch.nn.GELU(), torch.nn.Linear(hid, d))
        torch.nn.init.zeros_(self.a[2].weight); torch.nn.init.zeros_(self.a[2].bias)

    def forward(self, x):
        return self.orig(x) + self.a(x[..., self.lo:self.hi])


def fold(parts, parity):
    """episode-level split of a list of (cell, obs, act, wt): returns (kept parity, other)"""
    tr, va, n = [], [], 0
    for c, o, a, w in parts:
        ids = np.arange(len(o)) + n; n += len(o)
        k = ids % 2 == parity
        if k.any(): tr.append((c, o[k], a[k], float(k.sum())))
        if (~k).any(): va.append((c, o[~k], a[~k], float((~k).sum())))
    return tr, va


def score_draw(parts, norm, dev, steps, lr, seed, log):
    base_l, fit_l = [], {s: [] for s in SLOTS}
    for parity in (0, 1):
        tr, va = fold(parts, parity)
        Dtr, Dva = Data(tr, norm, "free", dev), Data(va, norm, "free", dev)
        m0, _, _ = load_model(RES + "/free/base.pt", dev)
        for p in m0.parameters(): p.requires_grad_(False)
        base_l.append(r1s.val_loss(m0, Dva))
        for s, (lo, hi) in SLOTS.items():
            torch.manual_seed(seed * 1000 + parity)
            m, _, _ = load_model(RES + "/free/base.pt", dev)
            for p in m.parameters(): p.requires_grad_(False)
            m.state_in = SlotState(m.state_in, lo, hi, m.cfg["d"]).to(dev)
            params = list(m.state_in.a.parameters())
            r1.train(m, Dtr, params, steps, lr, 256, log, f"bc_{s}")
            fit_l[s].append(r1s.val_loss(m, Dva))
    base = float(np.mean(base_l)); fit = {s: float(np.mean(v)) for s, v in fit_l.items()}
    order = sorted(fit, key=fit.get)
    return dict(base=base, fit=fit, gain={s: base - v for s, v in fit.items()}, pick=order[0],
                margin=fit[order[1]] - fit[order[0]], second=order[1])


def cmd_run(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    def log(m): print(m, flush=True)
    steps = 20 if a.smoke else a.steps
    ck = torch.load(RES + "/free/base.pt", map_location="cpu", weights_only=False)
    norm = b1_data.Norm.from_dict(ck["norm"])
    sp, src = r1s.sources(a.seed)
    rng = np.random.RandomState(a.seed)
    demos = {k: r1s.draw(src[k], rng) for k in TASKS}   # identical draws to r1s.py / r1fd.py
    res = []
    for k in [t for t in TASKS if a.part in ("all", t)]:
        r = score_draw(demos[k], norm, dev, steps, a.lr, a.seed, log)
        r.update(kind="task", task=k, seed=a.seed, truth=TRUTH[k]); res.append(r)
        log(f"seed {a.seed} task {k} truth {TRUTH[k]} pick {r['pick']} margin {r['margin']:.5f} base {r['base']:.5f} fit {json.dumps(r['fit'])}")
        if a.smoke: break
    dr = r1s.dropped(); panel = [c for c in sp["panel"] if c not in dr]
    nrng = np.random.RandomState(100 + a.seed)
    for i, combo in enumerate(itertools.combinations(panel, 3)):
        dd = r1s.draw(list(combo), nrng)   # draw first so the rng stream is identical across parts
        if a.part not in ("all", f"N{i}"): continue
        r = score_draw(dd, norm, dev, steps, a.lr, a.seed, log)
        r.update(kind="null", task="PANEL", seed=a.seed, cells=[b1.cell_name(c) for c in combo], truth=None); res.append(r)
        log(f"seed {a.seed} null {r['cells']} pick {r['pick']} margin {r['margin']:.5f} base {r['base']:.5f} fit {json.dumps(r['fit'])}")
        if a.smoke: break
    json.dump(res, open(f"{OUT}/run_seed{a.seed}_{a.part}{'_smoke' if a.smoke else ''}.json", "w", encoding="utf-8"), indent=1)
    log("run done")


def cmd_report(a):
    rows = []
    for s in (0, 1, 2):
        for part in ("HB", "OW", "TC", "N0", "N1", "N2", "N3"):
            rows += json.load(open(f"{OUT}/run_seed{s}_{part}.json", encoding="utf-8"))
    T = [r for r in rows if r["kind"] == "task"]; N = [r for r in rows if r["kind"] == "null"]
    nm = np.array([r["margin"] for r in N]); nb = np.array([r["base"] for r in N]); ng = np.array([max(r["gain"].values()) for r in N])
    q = lambda v: float(np.percentile(v, 95))
    th_m, th_b, th_g = q(nm), q(nb), q(ng)
    out = dict(null=dict(n=len(N), margin_p95=th_m, margin_mean=float(nm.mean()), margin_std=float(nm.std()), base_p95=th_b,
                         base_mean=float(nb.mean()), maxgain_p95=th_g, maxgain_mean=float(ng.mean()),
                         null_picks=[r["pick"] for r in N]), tasks=[])
    for r in T:
        out["tasks"].append(dict(seed=r["seed"], task=r["task"], truth=r["truth"], pick=r["pick"], hit=r["pick"] == r["truth"],
                                 margin=r["margin"], margin_gt_null=r["margin"] > th_m, base=r["base"], base_gt_null=r["base"] > th_b,
                                 maxgain=max(r["gain"].values()), create=max(r["gain"].values()) > th_g, fit=r["fit"], gain=r["gain"]))
    t = out["tasks"]
    out["BC_P0"] = dict(hits=sum(x["hit"] for x in t), of=len(t), verdict="PASS" if sum(x["hit"] for x in t) >= 8 else ("FALSIFIED" if sum(x["hit"] for x in t) <= 6 else "INCONCLUSIVE"))
    out["BC_P1"] = dict(n=sum(x["margin_gt_null"] for x in t), of=len(t), verdict="PASS" if sum(x["margin_gt_null"] for x in t) >= 8 else "FAIL")
    out["BC_P4"] = dict(n=sum(x["base_gt_null"] for x in t), of=len(t), verdict="PASS" if sum(x["base_gt_null"] for x in t) >= 8 else "FAIL")
    out["BC_P2"] = dict(n=sum(x["create"] for x in t), of=len(t), verdict="PASS" if sum(x["create"] for x in t) == len(t) else "FAIL")
    for x in t: print(json.dumps(x))
    print("null", json.dumps(out["null"]))
    for k in ("BC_P0", "BC_P1", "BC_P4", "BC_P2"): print(k, json.dumps(out[k]))
    json.dump(out, open(f"{OUT}/report_bc.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--smoke", action="store_true"); p.add_argument("--part", default="all")
    a = p.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)

"""Slot-token R1 policy: the state enters as 4 slot tokens (object 0-14, obstacle 14-28, goal 28-45, proprio 45-77), each its own MLP embed
plus a learned slot embedding, attended jointly with task, time and action tokens.  Otherwise the r1s recipe (same data, steps, lr, batch,
demo draws).  Registered in results/r1/stream/PREDICTIONS_tok.md.
  base                       -> results/r1/tok/base.pt          (60k steps lr 3e-4 bs 256 on the r1 base cells; norm reused from free/base.pt)
  parts --seed S             -> results/r1/tok/sS/parts/{HB,OW,TC}.pt   (rank-1 LoRA, 3000 steps lr 2e-3, r1s demo draws)
  eval --seed S --group G    -> results/r1/tok/sS/eval/routed_smax__G.json  (labelled routing, smax, r1s episode counts)
  report                     -> results/r1/tok/report_tok.json  (TK-P1)
  bc --seed S                -> results/r1/tok/bc/run_seedS.json (idea B: slot-owned adapters inside the slot token, 2-fold held-out flow loss)
  bcreport                   -> results/r1/tok/bc/report_bc.json (BC-P0 BC-P1 BC-P4 BC-P2)"""
import argparse, itertools, json, os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1, r1s
from r1s import *  # noqa  (np, torch, load_cell, Data, RES, b1, b1_data, TASKS, H)
from lifelong_ood.b1_policy import FlowPolicy, time_embed

V2 = "--v2" in sys.argv; CTL = "--ctl" in sys.argv
TOK = RES + ("/tok2" if V2 else "/tokctl" if CTL else "/tok")
BASEPT = (RES + "/free/base.pt") if CTL else TOK + ("/base_smoke.pt" if "--smoke" in sys.argv else "/base.pt")
SLICES = [(0, 14), (14, 28), (28, 45), (45, 77)]   # object obstacle goal proprio
SLOTIDX = {"object": 0, "obstacle": 1, "goal": 2}
TRUTH = {"HB": "object", "OW": "obstacle", "TC": "goal"}
nn = torch.nn


class TokPolicy(FlowPolicy):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        d = cfg.get("d", 256)
        del self.state_in
        self.slot_in = nn.ModuleList([nn.Sequential(nn.Linear(hi - lo, d), nn.GELU(), nn.Linear(d, d)) for lo, hi in SLICES])
        self.slot_emb = nn.Parameter(torch.randn(len(SLICES), d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, H + 2 + len(SLICES), d) * 0.02)

    def velocity(self, x_t, t, state, task=None, z=None):
        toks = [f(state[..., lo:hi]) + self.slot_emb[i] for i, (f, (lo, hi)) in enumerate(zip(self.slot_in, SLICES))]
        c = torch.stack(toks + [self.task_vec(task, z), self.time_in(time_embed(t, self.pos.shape[-1]))], 1)
        h = torch.cat([c, self.act_in(x_t)], 1) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.out(self.out_n(h[:, c.shape[1]:]))


class TokPolicyD(TokPolicy):
    """TokPolicy + a learned null per slot; during training each slot token is replaced by its null with prob slot_drop (set only in base training)"""
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.null = nn.Parameter(torch.randn(len(SLICES), cfg.get("d", 256)) * 0.02)
        self.slot_drop = 0.0

    def velocity(self, x_t, t, state, task=None, z=None):
        toks = torch.stack([f(state[..., lo:hi]) + self.slot_emb[i] for i, (f, (lo, hi)) in enumerate(zip(self.slot_in, SLICES))], 1)
        if self.training and self.slot_drop > 0:
            keep = torch.rand(toks.shape[0], toks.shape[1], 1, device=toks.device) >= self.slot_drop
            toks = torch.where(keep, toks, self.null[None].expand_as(toks))
        c = torch.cat([toks, self.task_vec(task, z)[:, None], self.time_in(time_embed(t, self.pos.shape[-1]))[:, None]], 1)
        h = torch.cat([c, self.act_in(x_t)], 1) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.out(self.out_n(h[:, c.shape[1]:]))


def load_tok(path, dev="cpu"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = (TokPolicyD if V2 else FlowPolicy if CTL else TokPolicy)(**ck["cfg"]).to(dev)
    if ck["lora_r"]:
        m.add_lora(ck["lora_r"])
    m.load_state_dict(ck["sd"])
    return m, b1_data.Norm.from_dict(ck["norm"]), ck


def sdir(seed):
    return f"{TOK}/s{seed}"


def mklog(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a", encoding="utf-8")
    def log(m):
        print(m, flush=True); f.write(m + "\n"); f.flush()
    return log


def cmd_base(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    log = mklog(TOK + "/train.log")
    ck = torch.load(RES + "/free/base.pt", map_location="cpu", weights_only=False)
    norm = b1_data.Norm.from_dict(ck["norm"])
    sp = r1.make_split(0)
    bp = []
    for c in sp["train"]:
        o, ac = load_cell(c); bp.append((c, o, ac, float(len(o))))
    steps = 20 if a.smoke else a.base_steps
    m = (TokPolicyD if V2 else TokPolicy)(**ck["cfg"]).to(dev)
    if V2: m.slot_drop = a.slot_drop
    log(f"device {dev} tok base steps {steps} params {sum(p.numel() for p in m.parameters())} episodes {sum(len(p[1]) for p in bp)}")
    r1.train(m, Data(bp, norm, "free", dev), list(m.parameters()), steps, 3e-4, 256, log, "tokbase")
    r1.save(BASEPT, m, norm, 0)
    log("base done")


def cmd_parts(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    d = sdir(a.seed); log = mklog(d + "/train.log")
    sp, src = r1s.sources(a.seed)
    rng = np.random.RandomState(a.seed)
    demos = {k: r1s.draw(src[k], rng) for k in TASKS}   # identical draws to r1s.py
    _, norm, _ = load_tok(BASEPT, dev)
    steps = 20 if a.smoke else a.adapt_steps
    for k in TASKS:
        torch.manual_seed(a.seed)
        m, _, _ = load_tok(BASEPT, dev)
        params = m.add_lora(a.rank)
        r1.train(m, Data(demos[k], norm, "free", dev), params, steps, a.lora_lr, 256, log, f"part_{k}")
        os.makedirs(d + "/parts", exist_ok=True)
        r1.save(f"{d}/parts/{k}{'_smoke' if a.smoke else ''}.pt", m, norm, a.rank)
    log("parts done")


class TokRouted(r1s.Routed):
    def __init__(self, mode, d):
        self.mode = mode
        self.base, self.norm, _ = load_tok(BASEPT)
        self.parts = {k: load_tok(f"{d}/parts/{k}.pt")[0] for k in TASKS}
        self.on = []


def cmd_eval(a):
    torch.set_num_threads(1)
    d = sdir(a.seed); g = a.group
    cells = r1.make_split(0)["panel"] if g == "panel" else r1s.GROUPS[g]
    dr = r1s.dropped(); cells = [c for c in cells if c not in dr]
    if CTL:
        r1s.S = d
        model = r1s.Routed("smax")
    else:
        model = TokRouted("smax", d)
    res = dict(model="tok_routed_smax", group=g, cells={})
    for c in cells:
        model.set_cell(c)
        r = r1.rollout(model, model.norm, c, dict(task=torch.zeros(1, dtype=torch.long)), r1s.EPS[g], seed=0)
        res["cells"][b1.cell_name(c)] = dict(success=r["success"], steps=r["steps"])
        print(b1.cell_name(c), r["success"], f"{r['seconds']:.0f}s", flush=True)
    os.makedirs(d + "/eval", exist_ok=True)
    json.dump(res, open(f"{d}/eval/routed_smax__{g}.json", "w", encoding="utf-8"))


def cmd_report(a):
    S3 = ["single_HB", "single_OW", "single_TC"]; P3 = ["pair_HBOW", "pair_HBTC", "pair_OWTC"]
    def vals(s, gs):
        v = []
        for g in gs:
            f = f"{sdir(s)}/eval/routed_smax__{g}.json"
            if not os.path.exists(f): return None
            v += [x for c in json.load(open(f, encoding="utf-8"))["cells"].values() for x in c["success"]]
        return round(sum(v) / len(v), 3)
    out = {}
    for s in (0, 1, 2):
        out[f"seed{s}"] = dict(pairs=vals(s, P3), singles=vals(s, S3), panel=vals(s, ["panel"]), triple=vals(s, ["triple"]),
                               by_pair={g: vals(s, [g]) for g in P3}, by_single={g: vals(s, [g]) for g in S3})
    mean = lambda k: round(sum(out[f"seed{s}"][k] for s in range(3)) / 3, 3)
    out["mean"] = {k: mean(k) for k in ("pairs", "singles", "panel", "triple")}
    m = out["mean"]
    out["TK_D1" if V2 else "CTL_R8" if CTL else "TK_P1"] = dict(pairs=m["pairs"], panel=m["panel"],
                        verdict="PASS" if m["pairs"] >= .45 and m["panel"] >= .80 else ("FALSIFIED" if (m["panel"] < .75 if V2 else m["pairs"] < .40) else "INCONCLUSIVE"))
    for k, v in out.items(): print(k, json.dumps(v))
    json.dump(out, open(TOK + "/report_tok.json", "w", encoding="utf-8"), indent=1)


# ---- idea B on the slot-token base ----
class TokAdapt(nn.Module):
    """slot token embed = orig(x_slot) + adapter(x_slot); adapter output layer zero-init so the start equals the base"""
    def __init__(self, orig, n, d, hid=64):
        super().__init__()
        self.orig = orig
        self.a = nn.Sequential(nn.Linear(n, hid), nn.GELU(), nn.Linear(hid, d))
        nn.init.zeros_(self.a[2].weight); nn.init.zeros_(self.a[2].bias)

    def forward(self, x):
        return self.orig(x) + self.a(x)


def fold(parts, parity):
    tr, va, n = [], [], 0
    for c, o, ac, w in parts:
        ids = np.arange(len(o)) + n; n += len(o)
        k = ids % 2 == parity
        if k.any(): tr.append((c, o[k], ac[k], float(k.sum())))
        if (~k).any(): va.append((c, o[~k], ac[~k], float((~k).sum())))
    return tr, va


def score_draw(parts, norm, dev, steps, lr, seed, log):
    base_l, fit_l = [], {s: [] for s in SLOTIDX}
    for parity in (0, 1):
        tr, va = fold(parts, parity)
        Dtr, Dva = Data(tr, norm, "free", dev), Data(va, norm, "free", dev)
        m0, _, _ = load_tok(BASEPT, dev)
        for p in m0.parameters(): p.requires_grad_(False)
        base_l.append(r1s.val_loss(m0, Dva))
        for s, i in SLOTIDX.items():
            torch.manual_seed(seed * 1000 + parity)
            m, _, _ = load_tok(BASEPT, dev)
            for p in m.parameters(): p.requires_grad_(False)
            lo, hi = SLICES[i]
            m.slot_in[i] = TokAdapt(m.slot_in[i], hi - lo, m.cfg["d"]).to(dev)
            r1.train(m, Dtr, list(m.slot_in[i].a.parameters()), steps, lr, 256, log, f"bc_{s}")
            fit_l[s].append(r1s.val_loss(m, Dva))
    base = float(np.mean(base_l)); fit = {s: float(np.mean(v)) for s, v in fit_l.items()}
    order = sorted(fit, key=fit.get)
    return dict(base=base, fit=fit, gain={s: base - v for s, v in fit.items()}, pick=order[0],
                margin=fit[order[1]] - fit[order[0]], second=order[1])


def cmd_bc(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = mklog(f"{TOK}/bc/bc_seed{a.seed}.log")
    steps = 20 if a.smoke else a.steps
    _, norm, _ = load_tok(BASEPT, dev)
    sp, src = r1s.sources(a.seed)
    rng = np.random.RandomState(a.seed)
    demos = {k: r1s.draw(src[k], rng) for k in TASKS}
    res = []
    for k in TASKS:
        r = score_draw(demos[k], norm, dev, steps, a.bc_lr, a.seed, log)
        r.update(kind="task", task=k, seed=a.seed, truth=TRUTH[k]); res.append(r)
        log(f"seed {a.seed} task {k} truth {TRUTH[k]} pick {r['pick']} margin {r['margin']:.5f} base {r['base']:.5f} fit {json.dumps(r['fit'])}")
        if a.smoke: break
    dr = r1s.dropped(); panel = [c for c in sp["panel"] if c not in dr]
    nrng = np.random.RandomState(100 + a.seed)
    for combo in itertools.combinations(panel, 3):
        r = score_draw(r1s.draw(list(combo), nrng), norm, dev, steps, a.bc_lr, a.seed, log)
        r.update(kind="null", task="PANEL", seed=a.seed, cells=[b1.cell_name(c) for c in combo], truth=None); res.append(r)
        log(f"seed {a.seed} null {r['cells']} pick {r['pick']} margin {r['margin']:.5f} base {r['base']:.5f} fit {json.dumps(r['fit'])}")
        if a.smoke: break
    json.dump(res, open(f"{TOK}/bc/run_seed{a.seed}{'_smoke' if a.smoke else ''}.json", "w", encoding="utf-8"), indent=1)
    log("bc done")


def cmd_bcreport(a):
    rows = []
    for s in (0, 1, 2):
        rows += json.load(open(f"{TOK}/bc/run_seed{s}.json", encoding="utf-8"))
    T = [r for r in rows if r["kind"] == "task"]; N = [r for r in rows if r["kind"] == "null"]
    nm = np.array([r["margin"] for r in N]); nb = np.array([r["base"] for r in N]); ng = np.array([max(r["gain"].values()) for r in N])
    q = lambda v: float(np.percentile(v, 95))
    th_m, th_b, th_g = q(nm), q(nb), q(ng)
    out = dict(null=dict(n=len(N), margin_p95=th_m, margin_mean=float(nm.mean()), base_p95=th_b, base_mean=float(nb.mean()),
                         maxgain_p95=th_g, null_picks=[r["pick"] for r in N]), tasks=[])
    for r in T:
        out["tasks"].append(dict(seed=r["seed"], task=r["task"], truth=r["truth"], pick=r["pick"], hit=r["pick"] == r["truth"],
                                 margin=r["margin"], margin_gt_null=r["margin"] > th_m, base=r["base"], base_gt_null=r["base"] > th_b,
                                 maxgain=max(r["gain"].values()), create=max(r["gain"].values()) > th_g, fit=r["fit"], gain=r["gain"]))
    t = out["tasks"]; h = sum(x["hit"] for x in t)
    out["BC_P0"] = dict(hits=h, of=len(t), verdict="PASS" if h >= 8 else ("FALSIFIED" if h <= 6 else "INCONCLUSIVE"))
    for k, key in (("BC_P1", "margin_gt_null"), ("BC_P4", "base_gt_null"), ("BC_P2", "create")):
        n = sum(x[key] for x in t); out[k] = dict(n=n, of=len(t), verdict="PASS" if n >= 8 else "FAIL")
    for x in t: print(json.dumps(x))
    print("null", json.dumps(out["null"]))
    for k in ("BC_P0", "BC_P1", "BC_P4", "BC_P2"): print(k, json.dumps(out[k]))
    json.dump(out, open(f"{TOK}/bc/report_bc.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--group"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--base_steps", type=int, default=60000); p.add_argument("--adapt_steps", type=int, default=3000)
    p.add_argument("--lora_lr", type=float, default=2e-3); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bc_lr", type=float, default=1e-3); p.add_argument("--rank", type=int, default=1); p.add_argument("--v2", action="store_true"); p.add_argument("--ctl", action="store_true"); p.add_argument("--slot_drop", type=float, default=0.0)
    a = p.parse_args()
    {"base": cmd_base, "parts": cmd_parts, "eval": cmd_eval, "report": cmd_report, "bc": cmd_bc, "bcreport": cmd_bcreport}[a.cmd](a)

"""TAIL baseline (Liu et al. ICLR 2024): one rank-8 LoRA adapter per stage on the R1 stream.
  train --seeds 0 1 2        adapters -> <seeddir>/tail/{HB,OW,TC}.pt (same sources/demos/steps/lr as r1s.py)
  eval --seed S --group G    variants: own (adapter of the cell's single new value) | sel_X | merge | none
  report                     -> results/r1/stream/report_tail.json"""
import argparse, json, os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1s
from lifelong_ood.b1_policy import LoRALinear
from r1s import *  # noqa

def sdir(seed):
    return r1s.S0 if seed == 0 else f"{r1s.S0}/seed{seed}"

def cmd_train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(RES + "/free/base.pt", map_location=dev, weights_only=False)
    norm = b1_data.Norm.from_dict(ck["norm"])
    for seed in a.seeds:
        out = sdir(seed) + "/tail"; os.makedirs(out, exist_ok=True)
        logf = open(out + "/train.log", "a", encoding="utf-8")
        def log(m):
            print(m, flush=True); logf.write(m + "\n"); logf.flush()
        sp, src = r1s.sources(seed)
        rng = np.random.RandomState(seed)
        demos = {k: r1s.draw(src[k], rng) for k in r1s.TASKS}
        log(f"seed {seed} sources {json.dumps({k: [b1.cell_name(c) for c in v] for k, v in src.items()})}")
        for k in r1s.TASKS:
            D = Data(demos[k], norm, "free", dev)
            m = FlowPolicy(**ck["cfg"]).to(dev); m.load_state_dict(ck["sd"])
            params = m.add_lora(a.rank)
            log(f"seed {seed} adapter {k} rank {a.rank} params {sum(p.numel() for p in params)} lr {a.lora_lr}")
            r1.train(m, D, params, a.adapt_steps, a.lora_lr, 256, log, f"tail_{k}")
            r1.save(f"{out}/{k}.pt", m, norm, a.rank)
        log("train done")

def merged(d, ks, sub="tail"):
    m, norm, _ = load_model(f"{d}/{sub}/{ks[0]}.pt")
    others = [load_model(f"{d}/{sub}/{k}.pt")[0] for k in ks[1:]]
    lms = [x for x in m.modules() if isinstance(x, LoRALinear)]
    for i, lm in enumerate(lms):
        A, B = [lm.A.data], [lm.B.data]
        for o in others:
            ol = [x for x in o.modules() if type(x) is type(lm)][i]
            A.append(ol.A.data); B.append(ol.B.data)
        lm.A = torch.nn.Parameter(torch.cat(A, 0)); lm.B = torch.nn.Parameter(torch.cat(B, 1)); lm.r = lm.A.shape[0]; lm.scale = 1.0
    return m, norm

def cmd_eval(a):
    torch.set_num_threads(1)
    d = sdir(a.seed); g = a.group
    cells = r1.make_split(0)["panel"] if g == "panel" else r1s.GROUPS[g]
    dr = r1s.dropped(); cells = [c for c in cells if c not in dr]
    eps = r1s.EPS[g]
    nv = {tuple(c): r1s.newvals(c) for c in cells}
    variants = {}   # name -> (adapter list)
    if g == "panel":
        variants["none"] = []
    elif g.startswith("single"):
        variants["own"] = None
    else:
        ks = sorted({k for c in cells for k in nv[tuple(c)]}, key=r1s.TASKS.index)
        for k in ks: variants[f"sel_{k}"] = [k]
        variants["merge"] = ks
        if g.startswith("pair"): variants["wmerge"] = ks
    os.makedirs(d + "/eval", exist_ok=True)
    for name, ks in variants.items():
        res = dict(model="tail_" + name, group=g, cells={})
        cache = {}
        for c in cells:
            use = nv[tuple(c)] if ks is None else ks
            key = tuple(use)
            if key not in cache:
                if not use: cache[key] = load_model(RES + "/free/base.pt")[:2]
                elif len(use) == 1: cache[key] = load_model(f"{d}/tail/{use[0]}.pt")[:2]
                else: cache[key] = merged(d, list(use), "parts" if name == "wmerge" else "tail")
            model, norm = cache[key]
            r = r1.rollout(model, norm, c, dict(task=torch.zeros(1, dtype=torch.long)), eps, seed=0)
            res["cells"][b1.cell_name(c)] = dict(success=r["success"], steps=r["steps"], adapters=list(use))
            print(name, b1.cell_name(c), r["success"], f"{r['seconds']:.0f}s", flush=True)
        json.dump(res, open(f"{d}/eval/tail_{name}__{g}.json", "w", encoding="utf-8"))

def cmd_report(a):
    def cellrates(d, name, g):
        f = f"{d}/eval/tail_{name}__{g}.json"
        if not os.path.exists(f): return None
        return {k: v["success"] for k, v in json.load(open(f, encoding="utf-8"))["cells"].items()}
    def pool(rs):
        v = [x for r in rs for xs in r.values() for x in xs]
        return sum(v) / len(v) if v else None
    S3 = ["single_HB", "single_OW", "single_TC"]; P3 = ["pair_HBOW", "pair_HBTC", "pair_OWTC"]
    def routed(d, g):
        f = f"{d}/eval/routed_smax__{g}.json"
        return json.load(open(f, encoding="utf-8"))["cells"] if os.path.exists(f) else None
    out = {}
    for seed in (0, 1, 2):
        d = sdir(seed); r = {}
        r["panel"] = pool([cellrates(d, "none", "panel")])
        r["singles"] = pool([cellrates(d, "own", g) for g in S3])
        sel, orc, mer = [], [], []; per = {}
        for g in P3 + ["triple"]:
            ks = {"pair_HBOW": ["HB", "OW"], "pair_HBTC": ["HB", "TC"], "pair_OWTC": ["OW", "TC"], "triple": ["HB", "OW", "TC"]}[g]
            ss = [cellrates(d, f"sel_{k}", g) for k in ks]; m = cellrates(d, "merge", g)
            cells = list(ss[0])
            sm = sum(pool([s]) for s in ss) / len(ss)
            ob = sum(max(sum(s[c]) / len(s[c]) for s in ss) for c in cells) / len(cells)
            per[g] = dict(select_mean=round(sm, 3), select_oracle=round(ob, 3), merge=round(pool([m]), 3),
                          routed_smax=round(pool([{c: routed(d, g)[c]["success"] for c in cells}]), 3))
            if g != "triple": sel.append(sm); orc.append(ob); mer.append(pool([m])); per[g]["parts_wmerge"] = round(pool([cellrates(d, "wmerge", g)]), 3)
        r["pairs_select_mean"] = sum(sel) / 3; r["pairs_select_oracle"] = sum(orc) / 3; r["pairs_merge"] = sum(mer) / 3; r["pairs_parts_wmerge"] = sum(per[g]["parts_wmerge"] for g in P3) / 3
        r["per_group"] = per
        r["routed_smax_pairs"] = sum(per[g]["routed_smax"] for g in P3) / 3
        r["routed_smax_singles"] = pool([{c: v["success"] for c, v in routed(d, g).items()} for g in S3])
        r["routed_smax_panel"] = pool([{c: v["success"] for c, v in routed(d, "panel").items()}])
        out[f"seed{seed}"] = r
    mean = {}
    for k in ("panel", "singles", "pairs_select_mean", "pairs_select_oracle", "pairs_merge", "pairs_parts_wmerge", "routed_smax_pairs", "routed_smax_singles", "routed_smax_panel"):
        mean[k] = sum(out[f"seed{s}"][k] for s in range(3)) / 3
    out["mean"] = mean
    json.dump(out, open(r1s.S0 + "/report_tail.json", "w", encoding="utf-8"), indent=1)
    for k, v in out.items(): print(k, json.dumps({x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "per_group"}))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group"); p.add_argument("--rank", type=int, default=8)
    p.add_argument("--adapt_steps", type=int, default=3000); p.add_argument("--lora_lr", type=float, default=2e-3)
    a = p.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)

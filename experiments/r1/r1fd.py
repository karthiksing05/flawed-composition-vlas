"""Factored-Diffusion-Policies-style baseline (arXiv 2605.22596): ONE monolith trained jointly on all stream data
(base demos + 16 demos per new value, same weighting as r1s.py joint) with FACTOR DROPOUT, composed at test time by
v = v_null + sum_f (v_f - v_null) ("fd-sum"), or used plainly fully conditioned ("fd-full").
  train --seeds S        -> <seeddir>/fd/fd.pt
  eval --seed S --group G -> <seeddir>/eval/fd_full__G.json (+ fd_sum__G.json on pair/triple groups)
  report                 -> results/r1/stream/report_fd.json
Factor identity: the r1s 'free' policy never sees the 16-dim CompoSuite indicator (stripped from the state, one shared task
embedding), so there is nothing to mask in the state.  The monolith therefore gets the 16-dim factor multi-hot (b1.multi_hot:
4 blocks of 4 = robot, object, obstacle, objective) through task_in (FlowPolicy cond='oracle'), plus a 4-dim keep mask.
Dropping factor f zeroes its 4 multi-hot entries and adds a learned null vector null[f] (d=256) to the conditioning token.
Per training sample each factor is dropped independently with p=0.25.  Trained from scratch (base recipe 60k steps lr 3e-4
plus 12k = 72k steps, bs 256) on the r1s joint data, like the joint_scratch arm."""
import argparse, json, os, sys
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1s
from r1s import *  # noqa
from lifelong_ood.b1_policy import FlowPolicy

FNAMES = ["robot", "object", "obstacle", "objective"]

class FDPolicy(FlowPolicy):
    def __init__(self, **cfg):
        cfg["cond"] = "oracle"
        super().__init__(**cfg)
        self.null = torch.nn.Parameter(torch.randn(4, cfg.get("d", 256)) * 0.02)
    def task_vec(self, task=None, z=None):
        mh, keep = task[:, :16], task[:, 16:20]
        return self.task_in(mh * keep.repeat_interleave(4, 1)) + ((1 - keep) @ self.null)

def sdir(seed):
    return r1s.S0 if seed == 0 else f"{r1s.S0}/seed{seed}"

def cmd_train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(RES + "/free/base.pt", map_location=dev, weights_only=False)
    norm = b1_data.Norm.from_dict(ck["norm"])
    cfg = {k: v for k, v in ck["cfg"].items() if k != "cond"}
    for seed in a.seeds:
        out = sdir(seed) + "/fd"; os.makedirs(out, exist_ok=True)
        logf = open(out + "/train.log", "a", encoding="utf-8")
        def log(m):
            print(m, flush=True); logf.write(m + "\n"); logf.flush()
        torch.manual_seed(seed)
        sp, src = r1s.sources(seed)
        rng = np.random.RandomState(seed)
        demos = {k: r1s.draw(src[k], rng) for k in r1s.TASKS}   # identical draws to r1s.py / r1tail.py
        dall = [p for k in r1s.TASKS for p in demos[k]]
        bp = []
        for c in sp["train"]:
            o, ac = load_cell(c); bp.append((c, o, ac, float(len(o))))
        tot = sum(p[3] for p in bp); dw = sum(p[3] for p in dall)
        D = Data(bp + [(c, o, ac, tot * wt / dw) for c, o, ac, wt in dall], norm, "oracle", dev)
        m = FDPolicy(**cfg).to(dev)
        log(f"seed {seed} fd monolith, demos {sum(len(p[1]) for p in dall)} p_drop {a.pdrop} steps {a.steps} lr {a.lr}")
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        m.train(); run = 0.0
        for i in range(a.steps):
            s, ac_, task = D.batch(256)
            keep = (torch.rand(task.shape[0], 4, device=task.device) >= a.pdrop).float()
            loss = m.loss(s, ac_, task=torch.cat([task, keep], 1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item() if i % 50 == 0 else 0
            if (i + 1) % (a.steps // 10) == 0:
                log(f"[fd] step {i+1}/{a.steps} loss~{run / (a.steps // 10 // 50):.4f}"); run = 0.0
        torch.save(dict(cfg=m.cfg, sd=m.state_dict(), norm=norm.to_dict()), out + "/fd.pt")
        log("train done")

class FDEval:
    def __init__(self, path, mode):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.m = FDPolicy(**{k: v for k, v in ck["cfg"].items() if k != "cond"}); self.m.load_state_dict(ck["sd"]); self.m.eval()
        self.norm = b1_data.Norm.from_dict(ck["norm"]); self.mode = mode
    def set_cell(self, c):
        self.mh = torch.tensor([b1.multi_hot(c)], dtype=torch.float32)
    def eval(self):
        pass
    @torch.no_grad()
    def sample(self, state, task=None, steps=10, gen=None, z=None):
        x = torch.randn(1, H, 8, generator=gen)
        eye = torch.eye(4)
        keeps = torch.cat([torch.zeros(1, 4), eye], 0)   # row 0 = all null, rows 1-4 = only factor f on
        for i in range(steps):
            t = torch.full((1,), i / steps)
            if self.mode == "full":
                v = self.m.velocity(x, t, state, torch.cat([self.mh, torch.ones(1, 4)], 1))
            else:
                vs = self.m.velocity(x.expand(5, -1, -1), t.expand(5), state.expand(5, -1), torch.cat([self.mh.expand(5, -1), keeps], 1))
                v = (vs[0] + (vs[1:] - vs[0]).sum(0))[None]
            x = x + v / steps
        return x

def cmd_eval(a):
    torch.set_num_threads(1)
    d = sdir(a.seed); g = a.group
    cells = r1.make_split(0)["panel"] if g == "panel" else r1s.GROUPS[g]
    dr = r1s.dropped(); cells = [c for c in cells if c not in dr]
    eps = r1s.EPS[g]
    modes = ["full", "sum"] if g.startswith("pair") or g == "triple" else ["full"]
    for mode in modes:
        model = FDEval(d + "/fd/fd.pt", mode)
        res = dict(model="fd_" + mode, group=g, cells={})
        for c in cells:
            model.set_cell(c)
            r = r1.rollout(model, model.norm, c, dict(task=torch.zeros(1, dtype=torch.long)), eps, seed=0)
            res["cells"][b1.cell_name(c)] = dict(success=r["success"], steps=r["steps"])
            print(mode, b1.cell_name(c), r["success"], f"{r['seconds']:.0f}s", flush=True)
        json.dump(res, open(f"{d}/eval/fd_{mode}__{g}.json", "w", encoding="utf-8"))

def cmd_report(a):
    def vals(d, tag, gs):
        v = []
        for g in gs:
            f = f"{d}/eval/{tag}__{g}.json"
            if not os.path.exists(f): return None
            v += [x for c in json.load(open(f, encoding="utf-8"))["cells"].values() for x in c["success"]]
        return round(sum(v) / len(v), 3)
    S3 = ["single_HB", "single_OW", "single_TC"]; P3 = ["pair_HBOW", "pair_HBTC", "pair_OWTC"]
    out = {}
    for seed in (0, 1, 2):
        d = sdir(seed)
        out[f"seed{seed}"] = dict(
            fd_sum_pairs=vals(d, "fd_sum", P3), fd_full_pairs=vals(d, "fd_full", P3),
            fd_sum_triple=vals(d, "fd_sum", ["triple"]), fd_full_triple=vals(d, "fd_full", ["triple"]),
            fd_singles=vals(d, "fd_full", S3), fd_panel=vals(d, "fd_full", ["panel"]),
            fd_sum_by_pair={g: vals(d, "fd_sum", [g]) for g in P3}, fd_full_by_pair={g: vals(d, "fd_full", [g]) for g in P3},
            routed_smax_pairs=vals(d, "routed_smax", P3), joint_pairs=vals(d, "joint", P3), poco_sum_pairs=vals(d, "routed_sum", P3),
            routed_smax_singles=vals(d, "routed_smax", S3), routed_smax_panel=vals(d, "routed_smax", ["panel"]),
            joint_singles=vals(d, "joint", S3), joint_panel=vals(d, "joint", ["panel"]))
    out["mean"] = {k: round(sum(out[f"seed{s}"][k] for s in range(3)) / 3, 3) for k in out["seed0"] if isinstance(out["seed0"][k], float)}
    json.dump(out, open(r1s.S0 + "/report_fd.json", "w", encoding="utf-8"), indent=1)
    for k, v in out.items(): print(k, json.dumps(v))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group"); p.add_argument("--steps", type=int, default=72000); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pdrop", type=float, default=0.25)
    a = p.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)

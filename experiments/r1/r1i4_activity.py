"""I4 offline activity pass (PLAN_INTERP_I45.md section I4: "Offline, batched: d_k = v_k - v_base along none's logged path for every part,
routed or not, and for 8 random rank-8 deltas per part on its LoRA layers"; norm matching over 4,096 base-train demo states).
  python r1i4_activity.py --seed S [--out DIR] [--smoke] [--ndraw 8]  ->  {OUT}/s{S}/activity.npz, activity.json
Reads {OUT}/s{S}/{group}.npz for every group present; uses only arm none's chunks (ep_arm == 0).  GPU if available (the full pass is
~28 models x ~10^5 chunks x 10 flow steps; CPU is ~hours, an A40 about a minute).

Literal-reading choices:
 - ||d|| = L2 norm over the whole (H=16, 8) chunk velocity, per (chunk, flow step); A_k(t) is taken in the scorer as the mean over the 10 steps.
 - Flow step i is evaluated at the logged x_i (path[i]) and time i/10, with the chunk-start state, task index 0, exactly as in sampling.
 - Random delta: the part's own rank-8 LoRA structure on every LoRALinear (qkv, proj, fc1, fc2 of every block), A ~ N(0,1)/sqrt(r) (the training
   init), B ~ N(0,1) times one scalar chosen by multiplicative fixed-point iteration (10 rounds, stop at 0.5%) so that the mean ||v_rand - v_base||
   over the 4,096 states equals the part's mean ||v_part - v_base|| over the same states.
 - The 4,096 states: base-train cells (r1.make_split(0)["train"] minus dropped), ceil(4096/ncells) uniformly drawn (episode, t) per cell, chunk =
   the H demo actions from t (last action repeated), x_t = (1-tau) x0 + tau x1 with tau ~ U(0,1), x0 ~ N(0,1), RandomState/torch seed 20261129
   (same states for every part and seed).
"""
import argparse, copy, json, math, os, sys, time
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1, r1s
from r1s import *  # noqa (np, torch, TASKS, GROUPS, RES, H, load_cell, dropped, load_model)
from lifelong_ood.b1_policy import LoRALinear

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm_states(norm, n=4096, seed=20261129):
    rng = np.random.RandomState(seed)
    cells = [c for c in r1.make_split(0)["train"] if c not in dropped()]
    per = math.ceil(n / len(cells)); S, X1 = [], []
    for c in cells:
        o, ac = load_cell(c)
        i = rng.randint(0, len(o), per); t = rng.randint(0, o.shape[1], per)
        idx = np.minimum(t[:, None] + np.arange(H)[None], o.shape[1] - 1)
        S.append(norm.state(o[i, t]).astype(np.float32)); X1.append(norm.act(ac[i[:, None], idx]).astype(np.float32))
    S = torch.tensor(np.concatenate(S)[:n]); X1 = torch.tensor(np.concatenate(X1)[:n])
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(X1.shape, generator=g); tau = torch.rand(len(X1), generator=g)
    return S, (1 - tau[:, None, None]) * x0 + tau[:, None, None] * X1, tau


@torch.no_grad()
def vel(m, state, x, t, bs=4096):
    out = []
    task = torch.zeros(state.shape[0], dtype=torch.long)
    for i in range(0, len(state), bs):
        out.append(m.velocity(x[i:i + bs].to(DEV), t[i:i + bs].to(DEV), state[i:i + bs].to(DEV), task[i:i + bs].to(DEV)).cpu())
    return torch.cat(out)


def mean_dnorm(m, vb, state, x, t):
    return float((vel(m, state, x, t) - vb).flatten(1).norm(dim=1).mean())


def rand_model(part, seed, target, vb, state, x, t):
    g = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(part)
    lins = [l for l in m.modules() if isinstance(l, LoRALinear) and l.r]
    B0 = []
    for l in lins:
        r = l.r
        l.A.data = (torch.randn(r, l.base.in_features, generator=g) / math.sqrt(r)).to(l.A.device)
        B0.append(torch.randn(l.base.out_features, r, generator=g).to(l.A.device))
    s = 0.01
    for _ in range(10):
        for l, b in zip(lins, B0): l.B.data = b * s
        got = mean_dnorm(m, vb, state, x, t)
        if abs(got / target - 1) < 0.005: break
        s *= target / got
    return m, s, got


def cmd_act(a):
    OUT = a.out or os.environ.get("I4_OUT") or RES + ("/i4_smoke" if a.smoke else "/i4")
    d = f"{RES}/tokctl/s{a.seed}"
    base, norm, _ = load_model(RES + "/free/base.pt"); base.to(DEV).eval()
    parts = {k: load_model(f"{d}/parts/{k}.pt")[0].to(DEV).eval() for k in TASKS}
    S, X, T = norm_states(norm)
    vb = vel(base, S, X, T)
    models, meta = [], dict(seed=a.seed, target={}, rand={})
    for ki, k in enumerate(TASKS):
        meta["target"][k] = mean_dnorm(parts[k], vb, S, X, T)
        models.append(parts[k])
    rnd = []
    for ki, k in enumerate(TASKS):
        row = []
        for j in range(a.ndraw):
            m, s, got = rand_model(parts[k], 1000 * a.seed + 10 * ki + j + 1, meta["target"][k], vb, S, X, T)
            row.append(m); meta["rand"][f"{k}_{j}"] = dict(scale=s, matched_norm=got, target=meta["target"][k])
        rnd.append(row)
    print("norm matching", json.dumps(meta, indent=0)[:1500], flush=True)
    outz = {}
    for g in GROUPS:
        f = f"{OUT}/s{a.seed}/{g}.npz"
        if not os.path.exists(f): continue
        z = np.load(f)
        rows = np.where(z["ep_arm"][z["ch_ep"]] == 0)[0]
        N = len(rows); t0 = time.time()
        Ap = np.zeros((N, len(TASKS), 10), np.float32); Ar = np.zeros((N, len(TASKS), a.ndraw, 10), np.float32)
        state, path = z["ch_state"][rows], z["ch_path"][rows]
        for b in range(0, N, 256):
            sl = slice(b, min(b + 256, N)); n = sl.stop - sl.start
            st = torch.tensor(state[sl]).repeat_interleave(10, 0)
            x = torch.tensor(path[sl]).reshape(n * 10, H, 8)
            tt = torch.tensor(np.tile(np.arange(10) / 10, n), dtype=torch.float32)
            v0 = vel(base, st, x, tt)
            for ki in range(len(TASKS)):
                Ap[sl, ki] = (vel(parts[TASKS[ki]], st, x, tt) - v0).flatten(1).norm(dim=1).reshape(n, 10).numpy()
                for j in range(a.ndraw):
                    Ar[sl, ki, j] = (vel(rnd[ki][j], st, x, tt) - v0).flatten(1).norm(dim=1).reshape(n, 10).numpy()
        outz[g + "__rows"] = rows; outz[g + "__A_part"] = Ap; outz[g + "__A_rand"] = Ar
        print(g, N, "chunks", f"{time.time() - t0:.0f}s", flush=True)
    outz["tasks"] = np.array(TASKS)
    np.savez_compressed(f"{OUT}/s{a.seed}/activity.npz", **outz)
    json.dump(meta, open(f"{OUT}/s{a.seed}/activity.json", "w", encoding="utf-8"))
    print("saved", f"{OUT}/s{a.seed}/activity.npz", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--ndraw", type=int, default=8)
    cmd_act(p.parse_args())

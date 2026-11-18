"""RAA-s (docs/notes/raas_plan.md) phase-0 OFFLINE GATE.  No model, no rollout, no simulator: reads the RAA-c LeRobot training set (the exact states and
actions the RAA-s expert would be trained on: lol/libero_g8v_raac, columns state / actions / episode_index only, no images).

Servo term: s = clip(K * rel_xyz, -M, +M) while the channel role id is 2 (target, held), else 0; rel_xyz = the state's first three numbers (metres).
The head applies s to every step of the 10-step action chunk (the channel is read once per chunk), so the fit uses the CHUNK-CONSISTENT pairs: for every held
row t and j = 0..9, demo action xyz at min(t + j, T - 1) against s(t).  The per-step (j = 0) numbers are reported as well.
M = the pooled p99 of |demo action xyz| over held rows (per-step).  K = least squares of the chunk-consistent pairs (1-D, grid + golden refine); the unclipped
closed form is reported as a check.  R^2 = 1 - SSE / SST about the demo's own mean (pooled over xyz); residual variance ratio = var(a - s) / var(a)
(per dim summed; per task and pooled); the mean-square ratio is also reported (the expert also has to learn the residual's mean).  Gate (pre-set in the task
brief): K > 0 and pooled residual variance ratio < 0.9.

    python experiments/raas_gate.py        (openpi venv, CPU, via sbatch)
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"
NORM = "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json"
OUT = os.path.join(ROOT, "results", "flawed", "raas_gate.json")
H = 10
KNOWN = ["K%d" % i for i in range(1, 9)]


def load():
    eps = []
    files = sorted(glob.glob(os.path.join(LR, "data", "chunk-000", "episode_*.parquet")))
    assert len(files) == 400, len(files)
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=["state", "actions", "episode_index"]).to_pydict()
        st = np.asarray(t["state"], np.float64)
        ac = np.asarray(t["actions"], np.float64)
        assert st.shape[1] == 4 and ac.shape[1] == 7 and set(t["episode_index"]) == {i}, (f, st.shape, ac.shape)
        eps.append((KNOWN[i // 50], st, ac))
    return eps


def pairs(eps):
    """-> per task dict of rel [n, 3], act [n, H, 3] (chunk actions, clamped at the episode end), a0 [n, 3] (per-step action at t); held rows only."""
    out = {}
    for task, st, ac in eps:
        T = len(ac)
        held = np.where(st[:, 3] > 1.5)[0]
        if len(held) == 0:
            continue
        idx = np.minimum(held[:, None] + np.arange(H)[None, :], T - 1)
        d = out.setdefault(task, dict(rel=[], act=[], a0=[], eps=0, T=0))
        d["rel"].append(st[held, :3])
        d["act"].append(ac[idx][:, :, :3])
        d["a0"].append(ac[held, :3])
        d["eps"] += 1
        d["T"] += int(T)
    for d in out.values():
        for k in ("rel", "act", "a0"):
            d[k] = np.concatenate(d[k])
    return out


def clip(K, rel, M):
    return np.clip(K * rel, -M, M)


def sse(K, rel, act, M):                                        # act [n, H, 3]
    return float(((act - clip(K, rel, M)[:, None, :]) ** 2).sum())


def fit_k(rel, act, M):
    ks = np.concatenate([np.linspace(0.0, 60.0, 1201)[1:], np.linspace(60.0, 400.0, 341)[1:]])
    e = np.array([sse(k, rel, act, M) for k in ks])
    i = int(np.argmin(e))
    lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
    g = (5 ** 0.5 - 1) / 2
    for _ in range(60):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if sse(a, rel, act, M) < sse(b, rel, act, M):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def stats(rel, act, a0, K, M):
    """R^2 and variance ratios of the servo against the chunk-consistent actions (act [n, H, 3]) and the per-step actions (a0 [n, 3])."""
    s = clip(K, rel, M)
    A = act.reshape(-1, 3)
    S = np.repeat(s[:, None, :], H, axis=1).reshape(-1, 3)
    r = A - S
    res = {}
    for name, a, sv in (("chunk", A, S), ("step", a0, s)):
        rr = a - sv
        sst = ((a - a.mean(0)) ** 2).sum()
        res[name] = dict(n=int(len(a)), r2=float(1 - (rr ** 2).sum() / sst), var_ratio=float(rr.var(0).sum() / a.var(0).sum()),
                         ms_ratio=float((rr ** 2).sum() / (a ** 2).sum()), r2_per_dim=[float(1 - (rr[:, d] ** 2).sum() / ((a[:, d] - a[:, d].mean()) ** 2).sum()) for d in range(3)],
                         var_ratio_per_dim=[float(rr[:, d].var() / a[:, d].var()) for d in range(3)], resid_mean=[float(v) for v in rr.mean(0)],
                         act_mean=[float(v) for v in a.mean(0)], act_std=[float(v) for v in a.std(0)])
    by_j = []
    for j in range(H):
        a = act[:, j, :]
        rr = a - s
        by_j.append(dict(j=j, r2=float(1 - (rr ** 2).sum() / ((a - a.mean(0)) ** 2).sum()), var_ratio=float(rr.var(0).sum() / a.var(0).sum())))
    res["by_j"] = by_j
    res["clip_frac"] = float((np.abs(K * rel) > M).mean())
    return res


def main():
    eps = load()
    D = pairs(eps)
    rel = np.concatenate([D[t]["rel"] for t in sorted(D)])
    act = np.concatenate([D[t]["act"] for t in sorted(D)])
    a0 = np.concatenate([D[t]["a0"] for t in sorted(D)])
    M = float(np.percentile(np.abs(a0), 99))                                   # pooled over held rows and the xyz dims, per-step demo actions
    Kls = float((rel[:, None, :] * act).sum() / (rel ** 2).sum() / H)          # unclipped closed form (chunk-consistent), a check
    K = fit_k(rel, act, M)
    pooled = stats(rel, act, a0, K, M)
    per_task = {}
    for t in sorted(D):
        d = D[t]
        per_task[t] = dict(episodes=d["eps"], held_rows=int(len(d["rel"])), **stats(d["rel"], d["act"], d["a0"], K, M),
                           k_task_only=fit_k(d["rel"], d["act"], M))
    ns = json.load(open(NORM, encoding="utf-8"))["norm_stats"]
    q01, q99 = np.asarray(ns["actions"]["q01"][:3]), np.asarray(ns["actions"]["q99"][:3])
    gate = bool(K > 0 and pooled["chunk"]["var_ratio"] < 0.9)
    res = dict(gate_pass=gate, rule="K > 0 and pooled residual variance ratio (chunk-consistent) < 0.9", K=K, K_unclipped_closed_form=Kls, max_step=M,
               units="K in action units per metre of target-relative xyz; max_step in raw action units (LIBERO delta-eef commands in [-1, 1])",
               held_rows=int(len(rel)), held_rows_per_task={t: int(len(D[t]["rel"])) for t in sorted(D)}, rel_abs_p50_cm=[float(100 * v) for v in np.percentile(np.abs(rel), 50, axis=0)],
               pooled=pooled, per_task=per_task,
               action_norm=dict(q01=q01.tolist(), q99=q99.tolist(), note="openpi quantile norm: (x - q01) / (q99 - q01 + 1e-6) * 2 - 1; servo in normalised units = s * 2 / (q99 - q01 + 1e-6)",
                                servo_norm_per_unit=(2.0 / (q99 - q01 + 1e-6)).tolist()),
               actions_are="raw absolute delta-eef commands (pi05 libero, extra_delta_transform=False): NOT action-minus-state, so the servo is a function of the channel only",
               data=LR, n_episodes=400)
    if os.path.exists(OUT):
        print("[raas-gate] refusing to overwrite", OUT)
        sys.exit(2)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print("[raas-gate] K %.3f (closed form unclipped %.3f), max_step %.4f, held rows %d" % (K, Kls, M, len(rel)))
    print("[raas-gate] pooled chunk: R2 %.4f var_ratio %.4f ms_ratio %.4f | per-step: R2 %.4f var_ratio %.4f" % (
        pooled["chunk"]["r2"], pooled["chunk"]["var_ratio"], pooled["chunk"]["ms_ratio"], pooled["step"]["r2"], pooled["step"]["var_ratio"]))
    for t in sorted(per_task):
        p = per_task[t]
        print("[raas-gate] %s rows %5d chunk R2 %.3f var_ratio %.3f | step R2 %.3f var_ratio %.3f | k_task %.2f" % (
            t, p["held_rows"], p["chunk"]["r2"], p["chunk"]["var_ratio"], p["step"]["r2"], p["step"]["var_ratio"], p["k_task_only"]))
    print("[raas-gate] GATE", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()

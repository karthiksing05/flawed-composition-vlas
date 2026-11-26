"""Gate T0 (temporal composition): label-free event segmenter + per-segment base flow loss on the R1 stream draws.
Registration: compositional-architecture results/tables/compose_stream_prediction.json, last addendum temporal_T0_registered;
spec docs/PLAN_TEMPORAL_COMPOSITION.md section 5.  Offline, CPU, no training, no rollouts.
  stats | loss --seed S | report  [--smoke]  -> results/r1/seg/t0_stats.json, t0_loss{S}.json, t0.json (t0_smoke*.json with --smoke)

FIXED by the registration (do not change): eps_s = 3 x p99 of a slot's step displacement over the first 5 steps of base-train demos;
event = change of M_t lasting 3 steps; S0 before the first event, S1 while the object is in M_t, S2 after; null rule = p95 over the 12
panel null draws of PREDICTIONS_bc.md (leave-one-out for the null control).  Labels (HB/OW/TC, objective) only score.

Literal-reading choices (also written to the JSON under "choices"):
 C1 slots = world positions: object obs[0:3], obstacle obs[14:17], goal obs[28:31] (77-dim stripped state; robot = proprio 45:77).
 C2 step displacement at t = ||pos[t]-pos[t-1]||, t >= 1 (displacement at t = 0 is 0); "first 5 steps" = displacements t = 1..5.
    p99 pooled per slot over those 5 displacements of every episode of every base-train cell (r1.make_split(0)["train"], not dropped).
    "exceeds" is strict (>).  A slot with p99 = 0 has eps = 0, so any motion exceeds it.
 C3 raw M_t[t] = {slots with displacement > eps}.  The committed set starts empty.  A different raw set that persists for 3 consecutive
    steps is an event, dated at its first step, and the committed set switches to it from that step on.
 C4 S1 = steps where the object is in the committed set.  S0 = steps before the first S1 step (so S0 ends at the first event that puts
    the object in M; events of other slots before it stay in S0 and are counted).  S2 = steps after the first S1 step that are not S1.
    If the object never enters M the episode is all S0.
 C5 episode "ok" (T0-P1) = S0, S1, S2 all non-empty and the object's committed membership is exactly one contiguous run.  Push is
    segmented by the same rule (S1 = the object moving = pushing contact).
 C6 per-chunk loss = r1s.val_loss's own computation (same seeded generator, same order, 8 reps, chunk starts 0,5,...,495), kept per
    chunk and averaged over chunk starts t whose step t is in the segment.  Whole-episode value = mean over all chunk starts (equals
    r1s.val_loss; asserted on the first draw).
 C7 null p95 = np.percentile(.., 95) (linear) over the 12 null draws per segment (NaN-aware); leave-one-out = same over the other 11.
 C8 T0-P2 "top-loss segment" = highest raw mean loss among S0, S1, S2 (not the excess over null).  The excess-ranked top is logged.
 C9 L1: ProprioHeld (copied from lifelong_ood/policy/parts_flow.py) on grip = obs[73:75] (finger qpos), cmd = act[:, 7] > 0 closes,
    z = obs[68] (eef z; proprio 21:24 = eef pos), tw by pgate_fit on all base-train rows.  "Grasping episodes" = objective != Push.
    Agreement = L1 fires and S1 exists and |L1 onset - S1 onset| <= 5; an episode where either is missing counts as disagreement.
 C10 whole-episode null for T0-P3 = p95 of the 12 null draws' own whole-episode loss under C6 (the registered .318 is logged beside it).
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1, r1s
from r1s import *  # noqa  (np, torch, load_cell, Data, load_model, RES, b1, b1_data, TASKS)

OUT = RES + "/seg"
POS = {"object": (0, 3), "obstacle": (14, 17), "goal": (28, 31)}
SLOTN = list(POS)
GRIP, CMD, ZIDX = (73, 75), 7, 68
DEBOUNCE, EPS_MULT, NEARLY = 3, 3.0, 5
TRUTH = {"HB": "object", "OW": "obstacle", "TC": "goal"}
SEGN = ["S0", "S1", "S2"]


def disp(obs):
    """obs (E,T,77) -> (E,T,3) step displacement of the three non-robot slots (0 at t=0)"""
    d = np.zeros(obs.shape[:2] + (3,), np.float64)
    for k, (lo, hi) in enumerate(POS.values()):
        p = obs[..., lo:hi].astype(np.float64)
        d[:, 1:, k] = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)
    return d


def calibrate(cells):
    pool = [[], [], []]
    n = 0; byobj = {}
    for c in cells:
        o, _ = load_cell(c)
        d = disp(o)[:, 1:6]
        for k in range(3):
            pool[k].append(d[..., k].ravel())
        byobj.setdefault(c[1], []).append(d[..., 0].ravel())
        n += len(o)
    p99 = [float(np.percentile(np.concatenate(p), 99)) for p in pool]
    return dict(p99=dict(zip(SLOTN, p99)), eps=dict(zip(SLOTN, [EPS_MULT * v for v in p99])), n_demos=n,
                diag_object_p99_by_object_value={k: float(np.percentile(np.concatenate(v), 99)) for k, v in byobj.items()})


def committed(raw):
    """clean implementation of committed(): the committed set is the state after the last event dated <= t"""
    T = len(raw); cur = np.zeros(3, bool); evs = []
    cand, start, cnt = None, 0, 0
    for t in range(T):
        r = raw[t]
        if (r == cur).all():
            cand, cnt = None, 0
            continue
        if cand is not None and (r == cand).all():
            cnt += 1
        else:
            cand, start, cnt = r.copy(), t, 1
        if cnt == DEBOUNCE:
            cur = cand.copy(); evs.append((start, cur.copy())); cand, cnt = None, 0
    M = np.zeros((T, 3), bool); s = 0; st = np.zeros(3, bool)
    for (t0, m) in evs:
        M[s:t0] = st; s, st = t0, m
    M[s:] = st
    return M, [(t, [SLOTN[k] for k in range(3) if m[k]]) for t, m in evs]


def segment(obs, eps):
    """obs (E,T,77) -> seg (E,T) in {0,1,2}, and per-episode info dicts"""
    d = disp(obs); e = np.array([eps[s] for s in SLOTN])
    raw = d > e[None, None]
    E, T = obs.shape[:2]
    seg = np.zeros((E, T), np.int8); info = []
    for i in range(E):
        M, ev = committed(raw[i])
        o = M[:, 0]
        runs = int((np.diff(np.r_[0, o.astype(int)]) == 1).sum())
        if o.any():
            s_on = int(np.argmax(o))
            seg[i, s_on:] = 2; seg[i, o] = 1
        else:
            s_on = None
        n = [int((seg[i] == k).sum()) for k in range(3)]
        first = ev[0] if ev else None
        info.append(dict(s_on=s_on, runs=runs, n=n, n_events=len(ev), ok=bool(min(n) > 0 and runs == 1),
                         first_event_is_object=bool(first is not None and first[1] == ["object"]) if first else None,
                         other_before=bool(s_on is not None and any(t < s_on for t, _ in ev))))
    return seg, info


class ProprioHeld:   # verbatim from lifelong_ood/policy/parts_flow.py (LADDER_RESULTS 1.2)
    def __init__(self, tw, lift=0.01):
        self.tw, self.lift = float(tw), float(lift)
        self.on, self.zmin = False, None

    def step(self, grip, cmd, z):
        w, z = float(grip[0]) - float(grip[1]), float(z)
        if not float(cmd) > 0:
            self.on, self.zmin = False, None
            return False
        self.zmin = z if self.zmin is None else min(self.zmin, z)
        if not self.on and w >= self.tw and z - self.zmin >= self.lift:
            self.on = True
        elif self.on and w < self.tw:
            self.on = False
        return self.on


def l1_fit(cells):
    W, Cm = [], []
    for c in cells:
        o, a = load_cell(c)
        g = o[..., GRIP[0]:GRIP[1]].astype(np.float64).reshape(-1, 2)
        W.append(g[:, 0] - g[:, 1]); Cm.append((a[..., CMD] > 0).reshape(-1))
    w = np.concatenate(W); cl = np.concatenate(Cm)
    return dict(tw=float(0.5 * (w.min() + np.quantile(w[cl], 0.01))), w_min=float(w.min()),
                w_close_q01=float(np.quantile(w[cl], 0.01)), close_frac=float(cl.mean()))


def l1_onsets(obs, act, tw):
    out = []
    for i in range(len(obs)):
        det = ProprioHeld(tw); on = None
        for t in range(obs.shape[1]):
            if det.step(obs[i, t, GRIP[0]:GRIP[1]], act[i, t, CMD], obs[i, t, ZIDX]) and on is None:
                on = t
        out.append(on)
    return out


def chunk_losses(model, D, reps=8):
    """r1s.val_loss's computation, kept per (episode, chunk start); returns (E, nchunk) mean over reps, and the chunk-start steps"""
    g = torch.Generator(device=D.S.device).manual_seed(123)
    E, T = D.S.shape[0], D.T
    t = torch.arange(0, T, 5, device=D.S.device)
    L = np.zeros((E, len(t)))
    model.eval()
    with torch.no_grad():
        for rep_ in range(reps):
            tq = time.time()
            for e in range(E):
                idx = (t[:, None] + D.ar[None]).clamp(max=T - 1)
                a = D.A[e][idx]; s = D.S[e][t]
                x0 = torch.randn(a.shape, device=a.device, generator=g)
                tt = torch.rand(len(t), device=a.device, generator=g)
                task = torch.zeros(len(t), dtype=torch.long, device=a.device)
                l = model.loss(s, a, task=task, reduce=False, x0=x0, t=tt)
                L[e] += l.reshape(len(t), -1).sum(1).cpu().numpy()
                if rep_ == 0 and e < 2: print("    ep", e, round(time.time() - tq, 1), "s", flush=True)
    return L / reps, t.cpu().numpy()


def draw_record(model, norm, parts, eps, check=False):
    obs = np.concatenate([p[1] for p in parts])
    seg, info = segment(obs, eps)
    D = Data(parts, norm, "free", "cpu")
    t0_ = time.time(); L, t = chunk_losses(model, D); print("  chunk_losses", round(time.time() - t0_, 1), "s", flush=True)
    if check:
        ref = r1s.val_loss(model, D); print("  val_loss ref", ref, flush=True)
        assert abs(ref - L.mean()) < 1e-6 * max(1, abs(ref)), (ref, L.mean())
    sg = seg[:, t]
    loss = [float(L[sg == k].mean()) if (sg == k).any() else float("nan") for k in range(3)]
    return dict(whole=float(L.mean()), loss=loss, chunks=[int((sg == k).sum()) for k in range(3)],
                steps=[float((seg == k).mean()) for k in range(3)], ok_frac=float(np.mean([x["ok"] for x in info])))


def p95(v):
    v = np.asarray(v, float)
    return float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else float("nan")


def cmd_stats(a):
    pass  # default torch threads (OMP_NUM_THREADS=1 from the sbatch), as r1bc.py
    t_start = time.time()
    sp = r1.make_split(0); dr = r1s.dropped()
    train = [c for c in sp["train"] if c not in dr]
    allc = [c for c in r1s.all_cells() if c not in dr]
    if a.smoke:
        train = train[:4]; allc = allc[:2]
    cal = calibrate(train); eps = cal["eps"]
    print("calibration", json.dumps(cal), flush=True)
    l1 = l1_fit(train); print("L1 fit", json.dumps(l1), flush=True)

    # ---- segmentation stats over all stored cells (episodes kept by load_cell)
    byobj = {}
    for c in allc:
        o, ac = load_cell(c)
        seg, info = segment(o, eps)
        on = l1_onsets(o, ac, l1["tw"])
        s = byobj.setdefault(c[3], dict(n=0, ok=0, s0_empty=0, s2_empty=0, s1_empty=0, multi_run=0, other_before=0, first_not_object=0,
                                        n_events=[], seglen=[[], [], []], l1_n=0, l1_fire=0, l1_agree=0, l1_both=0, lag=[], first_ev=[]))
        for i, x in enumerate(info):
            s["n"] += 1; s["ok"] += x["ok"]; s["s0_empty"] += x["n"][0] == 0; s["s1_empty"] += x["n"][1] == 0; s["s2_empty"] += x["n"][2] == 0
            s["multi_run"] += x["runs"] > 1; s["other_before"] += bool(x["other_before"]); s["first_not_object"] += x["first_event_is_object"] is False
            s["n_events"].append(x["n_events"])
            for k in range(3):
                s["seglen"][k].append(x["n"][k])
            s["l1_n"] += 1; s["l1_fire"] += on[i] is not None
            if on[i] is not None and x["s_on"] is not None:
                s["l1_both"] += 1; s["lag"].append(on[i] - x["s_on"])
            s["l1_agree"] += bool(on[i] is not None and x["s_on"] is not None and abs(on[i] - x["s_on"]) <= NEARLY)
        print("cell", b1.cell_name(c), "done", flush=True)
    stats = {}
    for k, s in byobj.items():
        lag = np.array(s["lag"]) if s["lag"] else np.array([np.nan])
        d = dict(n=s["n"], ok_frac=s["ok"] / s["n"], s0_empty=s["s0_empty"] / s["n"], s1_empty=s["s1_empty"] / s["n"], s2_empty=s["s2_empty"] / s["n"],
                 multi_run=s["multi_run"] / s["n"], other_event_before_object=s["other_before"] / s["n"],
                 first_event_not_object=s["first_not_object"] / s["n"], n_events_mean=float(np.mean(s["n_events"])),
                 seg_len_mean=[float(np.mean(v)) for v in s["seglen"]],
                 l1=dict(fire_frac=s["l1_fire"] / s["l1_n"], agree_within5=s["l1_agree"] / s["l1_n"], both=s["l1_both"],
                         agree_given_both=float(np.mean(np.abs(lag) <= NEARLY)) if s["lag"] else None,
                         lag_q10_50_90=[float(x) for x in np.nanquantile(lag, [.1, .5, .9])]))
        stats[k] = d
    grasp = [k for k in byobj if k != "Push"]
    n_g = sum(byobj[k]["l1_n"] for k in grasp)
    l1_all = dict(grasping_objectives=grasp, n=n_g, agree_within5=sum(byobj[k]["l1_agree"] for k in grasp) / n_g,
                  agree_given_both=float(np.mean(np.abs(np.concatenate([byobj[k]["lag"] for k in grasp])) <= NEARLY)))
    for k, d in stats.items():
        print("STATS", k, json.dumps(d), flush=True)
    print("L1 grasping", json.dumps(l1_all), flush=True)

    json.dump(dict(calibration=cal, l1_fit=l1, segmentation=stats, l1_grasping=l1_all), open(f"{OUT}/t0{'_smoke' if a.smoke else ''}_stats.json", "w", encoding="utf-8"), indent=1)
    print("stats done", time.time() - t_start, flush=True)


def cmd_loss(a):
    sp = r1.make_split(0); dr = r1s.dropped()
    train = [c for c in sp["train"] if c not in dr]
    if a.smoke:
        train = train[:4]
    cal = calibrate(train); eps = cal["eps"]
    # ---- per-segment loss on the 9 stream draws and 12 null draws
    ck = torch.load(RES + "/free/base.pt", map_location="cpu", weights_only=False)
    norm = b1_data.Norm.from_dict(ck["norm"])
    model, _, _ = load_model(RES + "/free/base.pt", "cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    tasks, nulls = [], []
    seeds = (a.seed,)
    for seed in seeds:
        _, src = r1s.sources(seed)
        rng = np.random.RandomState(seed)
        demos = {k: r1s.draw(src[k], rng) for k in TASKS}   # identical draws to r1s.py / r1bc.py
        for k in TASKS:
            r = draw_record(model, norm, demos[k], eps, check=(len(tasks) == 0))
            r.update(seed=seed, task=k, truth=TRUTH[k]); tasks.append(r)
            print("task", seed, k, json.dumps(r), flush=True)
            if a.smoke: break
        panel = [c for c in sp["panel"] if c not in dr]
        nrng = np.random.RandomState(100 + seed)
        for i, combo in enumerate(itertools.combinations(panel, 3)):
            dd = r1s.draw(list(combo), nrng)
            r = draw_record(model, norm, dd, eps)
            r.update(seed=seed, idx=i, cells=[b1.cell_name(c) for c in combo]); nulls.append(r)
            print("null", seed, i, json.dumps(r), flush=True)
            if a.smoke: break
        if a.smoke: break

    json.dump(dict(tasks=tasks, nulls=nulls), open(f"{OUT}/t0{'_smoke' if a.smoke else ''}_loss{a.seed}.json", "w", encoding="utf-8"), indent=1)
    print("loss done", flush=True)


def cmd_report(a):
    t_start = time.time()
    SD = f"{OUT}/t0{'_smoke' if a.smoke else ''}"
    st = json.load(open(SD + "_stats.json", encoding="utf-8"))
    cal, l1, stats, l1_all = st["calibration"], st["l1_fit"], st["segmentation"], st["l1_grasping"]
    tasks, nulls = [], []
    for seed in ((0,) if a.smoke else (0, 1, 2)):
        d = json.load(open(f"{SD}_loss{seed}.json", encoding="utf-8")); tasks += d["tasks"]; nulls += d["nulls"]
    NL = np.array([r["loss"] for r in nulls]); NW = np.array([r["whole"] for r in nulls])
    th = [p95(NL[:, k]) for k in range(3)]; thw = p95(NW)
    loo = []
    for i in range(len(nulls)):
        rest = np.delete(NL, i, 0)
        loo.append([bool(NL[i, k] > p95(rest[:, k])) if np.isfinite(NL[i, k]) else False for k in range(3)])
    loo_cnt = [int(sum(x[k] for x in loo)) for k in range(3)]
    rows = []
    for r in tasks:
        L = np.array(r["loss"]); ex = [bool(np.isfinite(L[k]) and L[k] > th[k]) for k in range(3)]
        fin = np.where(np.isfinite(L), L, -np.inf)
        top = int(np.argmax(fin))
        exc = np.where(np.isfinite(L), L - np.array(th), -np.inf); top_exc = int(np.argmax(exc))
        want = {"HB": (1,), "OW": (0,), "TC": (1, 2)}[r["task"]]
        rows.append(dict(seed=r["seed"], task=r["task"], new_factor=r["truth"], loss=r["loss"], null_p95=th, exceeds=ex,
                         top_raw=SEGN[top], top_excess=SEGN[top_exc], p2_hit=top in want, whole=r["whole"],
                         whole_exceeds=bool(r["whole"] > thw), chunks=r["chunks"], seg_step_frac=r["steps"], ok_frac=r["ok_frac"]))
    n_p2 = sum(x["p2_hit"] for x in rows)
    ow = [x for x in rows if x["task"] == "OW"]
    p3_s0 = sum(x["exceeds"][0] for x in ow); p3_whole = sum(x["whole_exceeds"] for x in ow)
    ow_inside = sum(not any(x["exceeds"]) for x in ow)
    p1_obj = {k: v["ok_frac"] for k, v in stats.items()}
    p1a = all(v >= .90 for v in p1_obj.values()); p1b = l1_all["agree_within5"] >= .85
    verdict = {
        "T0-P1": dict(holds=bool(p1a and p1b), segmentation_ok_frac=p1_obj, segmentation_all_ge_90=p1a, l1_agree=l1_all["agree_within5"], l1_ge_85=p1b),
        "T0-P2": dict(holds=bool(n_p2 >= 8), hits=int(n_p2), of=len(rows)),
        "T0-P3": dict(holds=bool(p3_s0 == len(ow) and p3_whole == 0), ow_s0_exceeds=int(p3_s0), ow_whole_exceeds=int(p3_whole), of=len(ow),
                      whole_null_p95_own=thw, whole_null_p95_registered_bc=0.318),
        "T0-P4": dict(holds=bool(max(loo_cnt) <= 2), loo_exceed_per_segment=dict(zip(SEGN, loo_cnt)), of=len(nulls)),
        "falsifier": dict(p2_le_6=bool(n_p2 <= 6), ow_inside_null_every_segment_on_ge2_seeds=bool(ow_inside >= 2), ow_seeds_inside=int(ow_inside)),
    }
    verdict["falsifier"]["triggered"] = bool(verdict["falsifier"]["p2_le_6"] or verdict["falsifier"]["ow_inside_null_every_segment_on_ge2_seeds"])
    for k, v in verdict.items():
        print("VERDICT", k, json.dumps(v), flush=True)
    res = dict(smoke=a.smoke, calibration=cal, l1_fit=l1, segmentation=stats, l1_grasping=l1_all, null=dict(loss=NL.tolist(), whole=NW.tolist(), p95=th,
               whole_p95=thw, loo_exceed=loo, loo_counts=loo_cnt, draws=nulls), tasks=rows, verdict=verdict, choices=__doc__, seconds=time.time() - t_start)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/t0{'_smoke' if a.smoke else ''}.json", "w", encoding="utf-8"), indent=1)
    print("done", time.time() - t_start, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd"); p.add_argument("--smoke", action="store_true"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"stats": cmd_stats, "loss": cmd_loss, "report": cmd_report}[a.cmd](a)

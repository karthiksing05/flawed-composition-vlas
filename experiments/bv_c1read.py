"""PLAN_BINDVLA 12.3 C1 dev read: paired task-cluster CIs, CD failure slope, predicted vs observed success."""
import json
import sys

import numpy as np

R = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [R, R + "/experiments"]
import bv_eval as E  # noqa: E402
import bv_c  # noqa: E402
from lifelong_ood.policy import bindvla as BV  # noqa: E402

sp = json.load(open(R + "/results/l90c/split.json", encoding="utf-8"))
bin_of = {t: k for k, v in sp["dev"].items() for t in v}
dev = {"c1": E.per_task("bv_devlog", "bindvla_c1main@20000"), "main2": E.per_task("bv_dev", "bindvla_main2@60000"),
       "vanilla": E.per_task("bv_dev", "bindvla_vanilla@60000"), "main2_log": E.per_task("bv_devlog", "bindvla_main2@60000")}
par = {"c1": E.per_task("bv_parity", "bindvla_c1main@20000"), "main2": E.per_task("bv_parity", "bindvla_main2@60000")}
tids = sorted(set.intersection(*[set(v) for v in dev.values()]))
rng = np.random.default_rng(0)
out = {"n_tasks": len(tids), "n_c1_episodes": int(sum(dev["c1"][t]["n"] for t in tids))}
S = {k: np.array([dev[k][t]["sr"] for t in tids]) for k in dev}
for k in S:
    out[f"dev_{k}"] = dict(mean=float(S[k].mean()), by_bin={b: float(np.mean([S[k][i] for i, t in enumerate(tids) if bin_of[t] == b]))
                                                        for b in ("0.1", "0.3", "0.5", "part")})


def paired(a, b):
    d = S[a] - S[b]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                tasks_up=int((d > 0).sum()), tasks_down=int((d < 0).sum()))


out["c1_minus_main2"] = paired("c1", "main2")
out["c1_minus_vanilla"] = paired("c1", "vanilla")
out["c1_minus_main2_log"] = paired("c1", "main2_log")
for k in ("c1", "vanilla", "main2"):
    xs = np.array([float(bin_of[t]) for t in tids if bin_of[t] != "part"])
    ys = np.array([1 - S[k][i] for i, t in enumerate(tids) if bin_of[t] != "part"])
    bb = []
    for _ in range(5000):
        j = rng.integers(0, len(xs), len(xs))
        if len(set(xs[j])) > 1:
            bb.append(np.polyfit(xs[j], ys[j], 1)[0])
    out[f"cd_failure_slope_{k}"] = dict(slope=float(np.polyfit(xs, ys, 1)[0]), ci95=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))])
for k in par:
    v = [x["sr"] for x in par[k].values()]
    out[f"parity_{k}"] = dict(tasks=len(v), mean=float(np.mean(v)))
# held-out-layout tasks (not trained by C1)
HO = {"libero_90:37", "libero_object:9", "libero_spatial:8", "libero_spatial:9"}
out["parity_c1_trained_72"] = float(np.mean([x["sr"] for t, x in par["c1"].items() if t not in HO]))
out["parity_c1_heldout_4"] = {t: par["c1"][t]["sr"] for t in HO if t in par["c1"]}
# predicted vs observed: per dev task hit_obj x hit_goal (closed-loop first-query slot-on rates, cldiag) x verb competence (parity)
cfg = BV.make_config("bindvla")


def verb(lang):
    p = BV.role_parse(lang, cfg)
    return p[0]["verb"] if p else ""


for k, tag, cl in (("c1", "bindvla_c1main@20000", "cldiag_bindvla_c1main_20000.json"), ("main2", "bindvla_main2@60000", "cldiag_bindvla_main2_60000.json")):
    rows = json.load(open(R + "/results/bindvla/" + cl, encoding="utf-8"))["rows"]
    comp_by_verb = {}
    for t, x in par[k].items():
        comp_by_verb.setdefault(verb(x["language"]), []).append(x["sr"])
    allc = float(np.mean([x["sr"] for x in par[k].values()]))
    pred, obs = [], []
    for t in tids:
        rs = [r for r in rows if r["tid"] == t]
        ho = [r["obj_on_named_start"] for r in rs if r.get("obj_on_named_start") is not None]
        hg = [r["goal_on_target_start"] for r in rs if r.get("goal_on_target_start") is not None]
        if not ho:
            continue
        lang = (dev["c1"] if k == "c1" else dev["main2"])[t]["language"]
        c = comp_by_verb.get(verb(lang))
        c = float(np.mean(c)) if c else allc
        pred.append(float(np.mean(ho)) * (float(np.mean(hg)) if hg else 1.0) * c)
        obs.append(float(S[k][tids.index(t)]))
    pred, obs = np.array(pred), np.array(obs)
    bb = []
    for _ in range(5000):
        j = rng.integers(0, len(pred), len(pred))
        if pred[j].std() > 0 and obs[j].std() > 0:
            bb.append(np.corrcoef(pred[j], obs[j])[0, 1])
    out[f"pred_vs_obs_{k}"] = dict(tasks=len(pred), mean_pred=float(pred.mean()), mean_obs=float(obs.mean()),
                                   mae=float(np.abs(pred - obs).mean()),
                                   r=float(np.corrcoef(pred, obs)[0, 1]) if obs.std() > 0 else None,
                                   r_ci95=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))] if bb else None)
out["per_task_c1"] = {t: dict(bin=bin_of[t], c1=float(S["c1"][i]), main2=float(S["main2"][i]), vanilla=float(S["vanilla"][i])) for i, t in enumerate(tids)}
json.dump(out, open(R + "/results/bindvla/c1_read.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_task_c1"}, indent=1))

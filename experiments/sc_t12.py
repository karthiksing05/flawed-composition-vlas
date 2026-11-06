"""T12 (docs/SCAFFOLD_RESULTS.md 25; m3_prereg sc_t12): the verb as a slot.  Arm V = T11's Um checkpoints, no new build, with the mode the words name.

    python experiments/sc_t12.py modes              (openpi venv) -> results/flawed/sc_t12_modes.json, and prints the parse of all 16 instructions
    python experiments/sc_t12.py roll --index 0     (LIBERO venv, CPU)
    python experiments/sc_t12.py read               -> results/flawed/sc_t12_read.json
SC_SMOKE=1: 2 states, results under sc_t12_smoke.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_t4 as T4                                                          # noqa: E402

OUT, SEEDS = T4.OUT, T4.SEEDS
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_t12_smoke2" if SMOKE else "sc_t12")
MODES = os.path.join(OUT, "sc_t12_modes.json")
PUSH = ("K8", "N8")
B, MARGIN = 2000, 0.10


def modes():
    import fa1_common as FC
    import pp_p_train as PT
    from lifelong_ood.policy import parts_flow as PF
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    names, lang = list(FC.KNOWN), PT.instructions()
    H = PT.pgate_held(D, PT.pgate_fit_tasks(names)) > 0.5
    out = {}
    for i, c in enumerate(names):
        idx = D["task"] == i
        ds = np.unique(D["demo"][idx])
        n = sum(bool(H[idx & (D["demo"] == d)].any()) for d in ds)
        out[lang[c]] = "close" if 2 * n > len(ds) else "open"
        print("[t12]", c, lang[c], "demos with a hold", n, "of", len(ds), "->", out[lang[c]], flush=True)
    with open(MODES + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(MODES + ".tmp", MODES)
    w = PF.verb_words(out)
    print("[t12] verb words", json.dumps(w), flush=True)
    for c in sorted(lang):
        print("[t12]", c, lang[c], "->", PF.verb_parse(w, lang[c]), flush=True)


def combos():
    n2 = T4.tasks()
    if SMOKE:
        return [("V", 10, T4.task(n2, c, "familiar")["tid"]) for c in PUSH]
    return [("V", s, T4.task(n2, c, l)["tid"]) for s, c, l in itertools.product(SEEDS, PUSH, ("familiar", "new"))]


def roll(arm, seed, tid):
    import torch
    import fa1_common as FC
    from fa1h_roll import episode
    from pp_p_roll import load_any
    from lifelong_ood.policy import parts_flow as PF
    fn = os.path.join(OD, f"{arm}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = T4.tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc11_Um_s{seed}", "P.npz"))
    assert m.pgate.get("motion"), "no motion flag in the checkpoint"
    verb = PF.verb_parse(PF.verb_words(json.load(open(MODES, encoding="utf-8"))), t["language"])
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"], verb=verb)
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], verb=verb, arrive=e.get("arrive"), held_step=e["held_step"], roles=e.get("roles"), steps=e["steps"], pgate=e.get("pgate"), final=e["final"].get(e["roles"][0]) if e.get("roles") else None, final_grip=e["final_grip"]))
        print("[t12]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = T4.tasks()
    ks = n2["states"]["main"]

    def succ(d, arm, s, c, l):
        E = json.load(open(os.path.join(OUT, d, f"{arm}_s{s}_{T4.task(n2, c, l)['tid']}.json"), encoding="utf-8"))["eps"]
        by = {int(e["trial"]): e for e in E}
        return np.array([float(by[k]["success"]) for k in ks]), E

    def arr(d, arm, c, l):
        return np.array([succ(d, arm, s, c, l)[0] for s in SEEDS])

    def boot(a, b=None):                                                    # [seeds, states], over states with seeds fixed
        rng = np.random.default_rng(0)
        x = a if b is None else a - b
        v = [float(x[:, rng.choice(x.shape[1], x.shape[1])].mean()) for _ in range(B)]
        return dict(mean=float(x.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(arm={}, delta={})
    for c, l in itertools.product(PUSH, ("familiar", "new")):
        V = arr("sc_t12", "V", c, l)
        res["arm"][f"{c}|{l}"] = boot(V)
        res["delta"][f"{c}|{l}"] = boot(V, arr("sc_t11", "Um", c, l))
        E = [e for s in SEEDS for e in succ("sc_t12", "V", s, c, l)[1]]
        res["arm"][f"{c}|{l}"].update(verb=E[0]["verb"], arrived=float(np.mean([e["arrive"] is not None for e in E])), event=float(np.mean([bool(e["pgate"] and e["pgate"]["event_step"] is not None) for e in E])), final_mean=np.mean([e["final"] for e in E if e["final"]], 0).tolist())
    n8, k8 = res["arm"]["N8|familiar"], res["delta"]["K8|familiar"]
    res["push_label"] = "VERB COMPOSES" if n8["mean"] >= 0.50 and n8["ci"][0] >= 0.25 else ("VERB PARTIAL" if n8["ci"][0] > 0.05 else "VERB DOES NOT COMPOSE")
    res["k8_label"] = "K8 HOLDS" if k8["ci"][0] >= -MARGIN else "K8 HURT"
    print("[t12]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t12_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("modes", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "modes":
        modes()
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        roll(*combos()[a.index])
    else:
        read()

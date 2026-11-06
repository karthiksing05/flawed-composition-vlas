"""T13 (docs/SCAFFOLD_RESULTS.md 26; m3_prereg sc_t13): the push verb on objects with other geometry.  Arm V = T11's Um checkpoints, no new build, T12's mode with object words explained.

    python experiments/sc_t13.py make               (any venv) -> results/flawed/sc_t13_bddl/*.bddl
    python experiments/sc_t13.py roll --index 0     (LIBERO venv, CPU; also prints parses on smoke)
    python experiments/sc_t13.py read               -> results/flawed/sc_t13_read.json
SC_SMOKE=1: 2 states, seed 10, familiar, results under sc_t13_smoke.
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
OD = os.path.join(OUT, "sc_t13_smoke" if SMOKE else "sc_t13")
BD = os.path.join(OUT, "sc_t13_bddl")
MODES = os.path.join(OUT, "sc_t12_modes.json")
TASKS = dict(P1=("push the cream cheese to the front of the stove", "On cream_cheese_1 main_table_stove_front_region"),
             P2=("push the wine bottle to the front of the stove", "On wine_bottle_1 main_table_stove_front_region"))
B = 2000


def make():
    import fc_n2_tasks as N
    text = open(N.TEMPLATE, encoding="utf-8").read()
    os.makedirs(BD, exist_ok=True)
    for k, (lang, atom) in TASKS.items():
        with open(os.path.join(BD, k + ".bddl"), "w", encoding="utf-8") as fh:
            fh.write(N.bddl_of(text, lang, atom, atom.split()[1]))
        print("[t13] wrote", k, lang, flush=True)


def explained(lex):
    return {w for w, p in lex["pg"].items() if p > 0}


def combos():
    if SMOKE:
        return [("V", 10, c, "familiar") for c in TASKS]
    return [("V", s, c, l) for s, c, l in itertools.product(SEEDS, TASKS, ("familiar", "new"))]


def roll(arm, seed, c, layout):
    import torch
    import fa1_common as FC
    from fa1h_roll import episode
    from pp_p_roll import load_any
    from lifelong_ood.policy import parts_flow as PF
    fn = os.path.join(OD, f"{arm}_s{seed}_{c}_{layout}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = T4.tasks()
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc11_Um_s{seed}", "P.npz"))
    words = PF.verb_words(json.load(open(MODES, encoding="utf-8")), explained(m.role_lex))
    if SMOKE:
        print("[t13] verb words", json.dumps(words), flush=True)
        for t in n2["tasks"]:
            if t["layout"] == "familiar":
                print("[t13] parse", t["compound"], t["language"], PF.role_parse(m.role_lex, t["language"]), PF.verb_parse(words, t["language"]), flush=True)
    lang = TASKS[c][0]
    verb = PF.verb_parse(words, lang)
    print("[t13] parse", c, lang, PF.role_parse(m.role_lex, lang), verb, flush=True)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{T4.task(n2, 'K8', layout)['tid']}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        e = episode(m, os.path.join(BD, c + ".bddl"), S[int(b["trial"])], lang, b["seed_base"], verb=verb)
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], verb=verb, arrive=e.get("arrive"), held_step=e["held_step"], roles=e.get("roles"), steps=e["steps"], pgate=e.get("pgate"), final=e["final"].get(e["roles"][0]) if e.get("roles") else None, final_grip=e["final_grip"]))
        print("[t13]", arm, seed, c, layout, b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=c, layout=layout, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    ks = T4.tasks()["states"]["main"]

    def eps(s, c, l):
        E = json.load(open(os.path.join(OD, f"V_s{s}_{c}_{l}.json"), encoding="utf-8"))["eps"]
        by = {int(e["trial"]): e for e in E}
        return [by[k] for k in ks]

    def boot(a):
        rng = np.random.default_rng(0)
        v = [float(a[:, rng.choice(a.shape[1], a.shape[1])].mean()) for _ in range(B)]
        return dict(mean=float(a.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(arm={}, labels={})
    for c, l in itertools.product(TASKS, ("familiar", "new")):
        A = [eps(s, c, l) for s in SEEDS]
        r = boot(np.array([[float(e["success"]) for e in x] for x in A]))
        E = [e for x in A for e in x]
        r.update(verb=E[0]["verb"], roles=E[0]["roles"], arrived=float(np.mean([e["arrive"] is not None for e in E])), final_mean=np.mean([e["final"] for e in E if e["final"]], 0).tolist())
        res["arm"][f"{c}|{l}"] = r
    for c in TASKS:
        r = res["arm"][f"{c}|familiar"]
        res["labels"][c] = "VERB COMPOSES" if r["mean"] >= 0.50 and r["ci"][0] >= 0.25 else ("VERB PARTIAL" if r["ci"][0] > 0.05 else "VERB DOES NOT COMPOSE")
    n = sum(v != "VERB DOES NOT COMPOSE" for v in res["labels"].values())
    res["reach"] = {2: "GENERAL", 1: "SOME OBJECTS", 0: "BOWL ONLY"}[n]
    print("[t13]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t13_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("make", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "make":
        make()
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        roll(*combos()[a.index])
    else:
        read()

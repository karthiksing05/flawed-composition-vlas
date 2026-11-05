"""S1 (docs/SCAFFOLD_RESULTS.md 3; m3_prereg sc_lex): does the lexicon's refusal remove silent wrong parses?  CPU, openpi venv, no rollouts.
Given roles are read for scoring only.  -> results/flawed/sc_lex.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as T                                                      # noqa: E402
from lifelong_ood.scaffold import Lexicon                                   # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
READ = [f"K{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, 7)]


def score(lex, lang, comps):
    """-> {compound: exact | refused | wrong}, with the refusal's reason."""
    out = {}
    for c in comps:
        p = lex.parse(lang[c])
        if p.abstained:
            out[c] = dict(result="refused", reason=p.reason, tokens=list(p.tokens))
        else:
            out[c] = dict(result="exact" if [p.object, p.target] == list(FC.roles(lang[c])) else "wrong", parsed=[p.object, p.target])
    return out


def count(sc, k, comps):
    return sum(sc[c]["result"] == k for c in comps)


def main():
    fn = os.path.join(OUT, "sc_lex.json")
    assert not os.path.exists(fn), "read exists"
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    lang = T.instructions()
    comps = sorted(lang)
    known = list(FC.KNOWN)
    pg = T.pgate_fit_tasks(known)
    held = T.pgate_held(D, pg)
    sall = T.build_situations(known, D, held)
    full = score(Lexicon(sall), lang, comps)
    res = dict(situations=sall, full=full, full_exact=count(full, "exact", READ), full_refused=count(full, "refused", READ), full_wrong=count(full, "wrong", READ))
    print("[lex] full", json.dumps({c: v["result"] for c, v in full.items()}), flush=True)
    loto = {}
    for c in known:
        if lang[c] not in sall:
            continue
        sc = score(Lexicon({s: v for s, v in sall.items() if s != lang[c]}), lang, comps)
        loto[c] = dict(parses=sc, exact=count(sc, "exact", READ), refused=count(sc, "refused", READ), wrong=[x for x in READ if sc[x]["result"] == "wrong"], self_result=sc[c]["result"])
        print("[lex] without", c, json.dumps({k: loto[c][k] for k in ("exact", "refused", "wrong", "self_result")}), flush=True)
    res["leave_one_task_out"] = loto
    sites = T.demo_sites(D, held)
    one = {}
    for j in range(50):
        sj = {lang[c]: [sites[(t, j)]["g"], sites[(t, j)]["r"]] for t, c in enumerate(known) if (t, j) in sites}
        sc = score(Lexicon(sj), lang, READ)
        one[j] = dict(exact=count(sc, "exact", READ), refused=count(sc, "refused", READ), wrong=[x for x in READ if sc[x]["result"] == "wrong"])
    res["one_demo"] = dict(per_index=one, mean_exact=float(np.mean([v["exact"] for v in one.values()])), indices_with_wrong=[j for j, v in one.items() if v["wrong"]])
    print("[lex] one-demo", json.dumps({k: res["one_demo"][k] for k in ("mean_exact", "indices_with_wrong")}), flush=True)
    silent = sorted({x for v in loto.values() for x in v["wrong"]} | {c for c, v in loto.items() if v["self_result"] == "wrong" and c in READ})
    res["label"] = "OVER-REFUSES" if res["full_exact"] < len(READ) else ("SILENT WRONG" if silent or res["one_demo"]["indices_with_wrong"] else "REFUSES SAFELY")
    res["silent_wrong_tasks"] = silent
    print("[lex]", res["label"], "silent", silent, flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

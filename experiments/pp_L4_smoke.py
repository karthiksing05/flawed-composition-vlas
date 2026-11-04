"""LADDER_RESULTS 5 (L4) smoke checks and the induction's comparison with the given roles.  CPU, openpi venv.

  rows       --old <path of the committed pp_p_train.py>: flag off, rows() equals it bitwise, without and with the L1 event
  roles      LADDER_RESULTS 5.4: per-demo sites, per-lexicon parses of all 16 compounds, one-demo lexicons, leave-one-task-out, token
             shuffles, plain prediction; 5.5: parse locality.  Given roles are read here for the comparison only.
  roll       the flag-off K5 familiar rollouts (results/flawed/pp_L4_smoke_roll) against the committed L0 and L1 files
  rollagree  descriptive: the event against the oracle in pp_L4_roll
-> results/flawed/pp_L4_smoke_<mode>.json (must not exist)
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_L1_smoke as S1                                                    # noqa: E402
import pp_p_train as T                                                      # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
BUILDS = dict(full=[], nobottle=["K4", "K5"], noplate=["K1"])
CREATE = dict(nobottle="K4", noplate="K1")
READ = [f"K{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, 7)]


def rows_check(old_path):
    spec = importlib.util.spec_from_file_location("pp_p_train_old", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.ROOT = ROOT
    dev = torch.device("cpu")
    res = {}
    pg = T.pgate_fit_tasks(list(FC.KNOWN))
    for label, p in (("oracle", None), ("event", pg)):
        for name, tasks, k in [(c, [x for x in FC.KNOWN if x not in ex], None) for c, ex in BUILDS.items()] + [("create_K4", ["K4"], 10)]:
            a, b = T.rows(dev, tasks, k, pgate=p), old.rows(dev, tasks, k, pgate=p)
            same = sorted(a[0]) == sorted(b[0]) and all(torch.equal(a[0][x], b[0][x]) for x in a[0])
            same = same and torch.equal(a[1], b[1]) and a[2] == b[2] and np.array_equal(a[3], b[3]) and a[4:] == b[4:]
            res[f"{label}|{name}"] = bool(same)
            print("[L4 smoke] rows", label, name, same, flush=True)
    m, meta = PF.PartsFlow.load(os.path.join(FC.DATA, "ckpt", "pp_full_L1_s10", "P.npz"))
    res["L1_ckpt_role_lex_none"] = bool(m.role_lex is None and "roles" not in meta)
    return dict(all_equal=bool(all(res.values())), cases=res)


def score(lex, lang, comps):
    out = {}
    for c in comps:
        p = PF.role_parse(lex, lang[c])
        out[c] = dict(parsed=p, given=list(FC.roles(lang[c])), exact=bool(p is not None and list(p) == list(FC.roles(lang[c]))))
    return out


def plain(situations, instr):
    """The comparator: no referents, no explaining away.  For each role, the slot with the largest summed P(site = slot | token)."""
    toks = instr.lower().split()
    res = []
    for i in (0, 1):
        sc = {}
        for w in toks:
            has = [s for s in situations if w in s.lower().split()]
            for s in has:
                sc[situations[s][i]] = sc.get(situations[s][i], 0.0) + 1.0 / len(has)
        if i == 1 and res[0] in sc:
            del sc[res[0]]
        if not sc:
            return None
        res.append(max(sorted(sc), key=sc.get))
    return res


def roles():
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    lang = T.instructions()
    comps = sorted(lang)
    known = list(FC.KNOWN)
    pg = T.pgate_fit_tasks(known)
    held = T.pgate_held(D, pg)
    sites = T.demo_sites(D, held)
    out = dict(instructions=lang, pgate=pg)
    # per demo
    per = {}
    for t, c in enumerate(known):
        g0 = FC.roles(lang[c])
        v = [x for (tt, _), x in sites.items() if tt == t]
        per[c] = dict(demos_with_event=len(v), g_ok=sum(x["g"] == g0[0] for x in v), r_ok=sum(x["r"] == g0[1] for x in v),
                      pairs={}, margin_g_q10_50=[float(q) for q in np.quantile([x["margin_g"] for x in v], [0.1, 0.5])] if v else None,
                      margin_r_q10_50=[float(q) for q in np.quantile([x["margin_r"] for x in v], [0.1, 0.5])] if v else None)
        for x in v:
            k = f"{x['g']}|{x['r']}"
            per[c]["pairs"][k] = per[c]["pairs"].get(k, 0) + 1
        print("[L4 roles] demos", c, json.dumps(per[c]), flush=True)
    out["per_demo"] = per
    # lexicons of the builds, before and after creation
    lex = {}
    for b, ex in BUILDS.items():
        tasks = [c for c in known if c not in ex]
        pb = T.pgate_fit_tasks(tasks)
        hb = T.pgate_held(D, pb)
        sb = T.build_situations(tasks, D, hb)
        lb = PF.role_lexicon(sb)
        lex[b] = dict(lexicon=lb, parses=score(lb, lang, comps))
        print("[L4 roles] lexicon", b, json.dumps(lb["ref"]), flush=True)
        print("[L4 roles] parse", b, "exact", sorted(c for c, v in lex[b]["parses"].items() if v["exact"]), "not", {c: v["parsed"] for c, v in lex[b]["parses"].items() if not v["exact"]}, flush=True)
        if b in CREATE:
            for k in (1, 5, 10):
                sk = T.build_situations([CREATE[b]], D, hb, k)
                lk = PF.role_lexicon(dict(sb, **sk))
                changed = [x for x in sb if PF.role_parse(lb, x) != PF.role_parse(lk, x)]
                lex[f"{b}+k{k}"] = dict(situation=sk, parses=score(lk, lang, comps), library_parses_changed=changed)
                v = lex[f"{b}+k{k}"]["parses"]
                print("[L4 roles] parse", b, "k", k, sk, "not exact", {c: x["parsed"] for c, x in v.items() if not x["exact"]}, "changed", changed, flush=True)
    out["lexicons"] = lex
    # parse locality (LADDER_RESULTS 5.5), full lexicon
    full = lex["full"]["lexicon"]
    bad = []
    for i, a in enumerate(READ):
        for b in READ[i + 1:]:
            ga, gb = FC.roles(lang[a]), FC.roles(lang[b])
            pa, pb_ = PF.role_parse(full, lang[a]), PF.role_parse(full, lang[b])
            for r in (0, 1):
                if ga[r] == gb[r] and (pa is None or pb_ is None or pa[r] != pb_[r]):
                    bad.append([a, b, r])
    out["parse_locality"] = dict(holds=not bad, violations=bad)
    print("[L4 roles] parse locality", out["parse_locality"], flush=True)
    # robustness: one-demo lexicons
    one = {}
    for j in range(50):
        sj = {}
        for t, c in enumerate(known):
            v = sites.get((t, j))
            if v:
                sj[lang[c]] = [v["g"], v["r"]]
        lj = PF.role_lexicon(sj)
        sc = score(lj, lang, comps)
        one[j] = dict(read_exact=sum(sc[c]["exact"] for c in READ), k7_exact=bool(sc["K7"]["exact"]), wrong=[c for c in READ if not sc[c]["exact"]])
    out["one_demo"] = dict(per_index=one, all_read_exact=sum(v["read_exact"] == len(READ) for v in one.values()),
                           mean_read_exact=float(np.mean([v["read_exact"] for v in one.values()])), n=50)
    print("[L4 roles] one-demo lexicons: all 12 read tasks exact in", out["one_demo"]["all_read_exact"], "of 50; mean", out["one_demo"]["mean_read_exact"],
          {j: v["wrong"] for j, v in one.items() if v["wrong"]}, flush=True)
    # robustness: leave one known task out
    sall = lex["full"]["lexicon"]["situations"]
    loto = {}
    for c in known:
        if lang[c] not in sall:
            continue
        lo = PF.role_lexicon({s: v for s, v in sall.items() if s != lang[c]})
        sc = score(lo, lang, comps)
        loto[c] = dict(itself=sc[c], read_exact=sum(sc[x]["exact"] for x in READ), wrong=[x for x in READ if not sc[x]["exact"]])
    out["leave_one_task_out"] = loto
    print("[L4 roles] LOTO", json.dumps({c: dict(self_parse=v["itself"]["parsed"], read_exact=v["read_exact"], wrong=v["wrong"]) for c, v in loto.items()}), flush=True)
    # robustness: token shuffles
    rng = np.random.default_rng(0)
    sh = {}
    for c in READ:
        toks = lang[c].lower().split()
        n = 0
        for _ in range(20):
            p = PF.role_parse(full, " ".join(rng.permutation(toks).tolist()))
            n += int(p is not None and list(p) == list(FC.roles(lang[c])))
        sh[c] = n
    out["token_shuffle_exact_of_20"] = sh
    print("[L4 roles] shuffles", sh, flush=True)
    # comparator: plain prediction
    pl = {c: dict(parsed=plain(sall, lang[c]), exact=plain(sall, lang[c]) == list(FC.roles(lang[c]))) for c in comps}
    out["plain_prediction"] = pl
    print("[L4 roles] plain prediction exact on read tasks", sum(pl[c]["exact"] for c in READ), "of 12; wrong", {c: pl[c]["parsed"] for c in READ if not pl[c]["exact"]}, flush=True)
    return out


def roll_check():
    res = {}
    for nm, ref in (("P2naf", "pp_cons_roll"), ("P2L1", "pp_L1_roll")):
        a = json.load(open(os.path.join(OUT, "pp_L4_smoke_roll", f"{nm}_s10_main_8.json"), encoding="utf-8"))
        b = json.load(open(os.path.join(OUT, ref, f"{nm}_s10_main_8.json"), encoding="utf-8"))
        ea, eb = a["tasks"]["8"]["episodes"], b["tasks"]["8"]["episodes"]
        diffs = [dict(trial=x.get("trial"), field=k) for x, y in zip(ea, eb) for k in sorted(set(x) | set(y)) if k != "secs" and x.get(k) != y.get(k)]
        res[nm] = dict(language=a["tasks"]["8"]["language"], success_rate=a["tasks"]["8"]["success_rate"], episodes=len(ea),
                       equal=bool(len(ea) == len(eb) and not diffs), diffs=diffs)
        print("[L4 smoke] roll", nm, json.dumps(res[nm]), flush=True)
    return dict(all_equal=all(v["equal"] for v in res.values()), arms=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rows", "roles", "roll", "rollagree"))
    ap.add_argument("--old", default="")
    a = ap.parse_args()
    fn = os.path.join(OUT, f"pp_L4_smoke_{a.mode}.json")
    assert not os.path.exists(fn), f"{fn} exists"
    r = (rows_check(a.old) if a.mode == "rows" else roles() if a.mode == "roles" else roll_check() if a.mode == "roll"
         else S1.roll_agree("pp_L4_roll"))
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1)
    print("[L4 smoke] wrote", fn, flush=True)


if __name__ == "__main__":
    main()

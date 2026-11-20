"""SR 83 (A2 distributedness probe): the registered read (CPU).  The mapping from outputs to labels is fixed in docs/notes/a2_plan.md and experiments/prereg/a2.py
BEFORE the main runs; nothing here is chosen after seeing data.

    python experiments/a2_read.py selftest
    python experiments/a2_read.py read --model pi05|raasl --dirs DIR[+DIR..] --out results/flawed/a2/a2_read_<model>.json      (refuses to overwrite)
    python experiments/a2_read.py plumb --dirs DIR[+DIR..]                        (smoke: plumbing numbers only, no effect sizes)

Effect measure.  For an observation, d5(chunk) = the sum of the first 5 xyz deltas of the unnormalised chunk (steer_text.chunk_delta); u_cf = the unit vector
from the end effector to the counterfactual entity, u_cor = to the entity the clean instruction names (simulator positions, measurement only).
    s_cf(chunk) = d5 . u_cf          s_cor(chunk) = d5 . u_cor
Effect of a component c on observation i:  E[i, c] = s_cf(patched chunk) - s_cf(clean chunk of the same batch);  total effect T[i] = the same for the row 'total'
(the counterfactual run itself).
Registered statistics, on the 72 component writes U1 = {vlm, exp} x layer 0-17 x {attn, mlp}, patched at all positions:
    e_c = mean_i E[i, c];   e+ = max(e_c, 0);   T = mean_i T[i]
    PR = (sum e+)^2 / sum (e+)^2,  PRn = PR / 72;   top1 = max e+ / sum e+;   top3 = the three largest / sum e+;   coverage = sum e+ / T
    chan_share = e(in.chan) / T (RAA-sL: state token + servo input patched, prefix clean);  also in.prefix, in.tok, in.servo
Selectivity.  intact_s = mean s_cor(clean) on set s;  abl_s(b) = mean s_cor(ablated);  drop(b, s) = (intact_s - abl_s(b)) / intact_s where the lower CI bound of
intact_s > 0 (else 'n/i');  sel(b, s) = drop(b, s) - drop(b, K).
Bootstrap: paired over observations, B = 2000, rng 0 (numpy default_rng(0)), clusters resampled within set (F7: both instructions of a layout are one cluster), 95% percentile CI.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
B_BOOT = 2000
NL = 18
U1 = ["%s.%d.%s.all" % (w, li, k) for w in ("vlm", "exp") for li in range(NL) for k in ("attn", "mlp")]
HEADS = ["%s.%d.head.%d" % (w, li, h) for w in ("vlm", "exp") for li in (3, 9, 15) for h in range(8)]
SETS = ("F7", "N", "K")
MARGIN_SEL = 0.2
TOP1_LOC, TOP1_DIST, PRN_DIST, CHAN_LOC = 0.5, 0.3, 0.2, 0.7


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def pr_stats(e, n_comp):
    ep = np.maximum(np.asarray(e, np.float64), 0.0)
    s = ep.sum()
    if s <= 0:
        return dict(PRn=np.nan, top1=np.nan, top3=np.nan, PR=np.nan)
    pr = s * s / (ep * ep).sum()
    srt = np.sort(ep)[::-1]
    return dict(PR=pr, PRn=pr / n_comp, top1=srt[0] / s, top3=srt[:3].sum() / s)


def load(dirs):
    obs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            names = [str(n) for n in z["names"]]
            ch = z["chunks"].astype(np.float64)
            ref = z["ref"].astype(int)
            eef = np.asarray(meta["eef"], np.float64)
            ucf, ucor = unit(np.asarray(meta["p_cf"]) - eef), unit(np.asarray(meta["p_cor"]) - eef)
            d5 = ch[:, :5, :3].sum(1)
            obs.append(dict(meta=meta, names=names, scf=d5 @ ucf, scor=d5 @ ucor, ref=ref, d5=d5))
    return obs


def table(obs, name_list):
    """-> effects [n_obs, len(name_list)] (NaN where a row is missing) and the total effect [n_obs]."""
    E = np.full((len(obs), len(name_list)), np.nan)
    T = np.full(len(obs), np.nan)
    for i, o in enumerate(obs):
        ix = {n: j for j, n in enumerate(o["names"])}
        for k, n in enumerate(name_list):
            if n in ix:
                j = ix[n]
                E[i, k] = o["scf"][j] - o["scf"][o["ref"][j]]
        if "total" in ix:
            j = ix["total"]
            T[i] = o["scf"][j] - o["scf"][o["ref"][j]]
    return E, T


def abl_table(obs, abl_names):
    A = np.full((len(obs), len(abl_names)), np.nan)
    I = np.full(len(obs), np.nan)
    for i, o in enumerate(obs):
        ix = {n: j for j, n in enumerate(o["names"])}
        I[i] = o["scor"][0]
        for k, n in enumerate(abl_names):
            if n in ix:
                A[i, k] = o["scor"][ix[n]]
    return A, I


def boot_idx(obs, rng):
    """One replicate: obs indices, clusters resampled with replacement within each set."""
    out = []
    for s in SETS:
        cl = {}
        for i, o in enumerate(obs):
            if o["meta"]["set"] == s:
                cl.setdefault(o["meta"]["cluster"], []).append(i)
        keys = sorted(cl)
        if not keys:
            continue
        for k in rng.integers(0, len(keys), len(keys)):
            out += cl[keys[k]]
    return np.array(out, int)


def ci(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def label_in(point, c, lo=None, hi=None):
    """Region label for a CI c = [l, u]: SUPPORTED if the CI lies in [lo, hi] (None = open), REVERSED if it lies entirely outside on the other side, else NOT SUPPORTED."""
    l, u = c
    if not (np.isfinite(l) and np.isfinite(u)):
        return "UNDEFINED"
    inside = (lo is None or l >= lo) and (hi is None or u <= hi)
    return "SUPPORTED" if inside else "NOT SUPPORTED"


def analyze(model, obs):
    n = len(obs)
    assert n, "no observations"
    rng = np.random.default_rng(0)
    comp = U1 + HEADS + (["in.chan", "in.prefix", "in.tok", "in.servo"] if model == "raasl" else [])
    comp_res = ["%s.%d.resid.%s" % (w, li, p) for w in ("vlm", "exp") for li in range(NL) for p in (("all", "ins") if w == "vlm" else ("all",))]
    comp_ins = ["vlm.%d.%s.ins" % (li, k) for li in range(NL) for k in ("attn", "mlp")]
    allc = comp + comp_res + comp_ins
    E, T = table(obs, allc)
    abl = ["abl.vlm.b%d.%s" % (b, p) for b in range(4) for p in ("all", "ins")] + ["abl.exp.b%d" % b for b in range(2)]
    if model == "raasl":
        abl += ["abl.chan", "abl.tok", "abl.servo"]
    A, I = abl_table(obs, abl)
    sets = np.array([o["meta"]["set"] for o in obs])
    col = {c: k for k, c in enumerate(allc)}
    u1 = [col[c] for c in U1]
    hd = [col[c] for c in HEADS]

    def stats(idx):
        r = {}
        for tag, m in [("ALL", np.ones(len(idx), bool))] + [(s, sets[idx] == s) for s in SETS]:
            ii = idx[m]
            if len(ii) == 0:
                continue
            Tm = np.nanmean(T[ii])
            e = np.nanmean(E[ii][:, u1], axis=0)
            st = pr_stats(e, len(u1))
            r[tag + ".T"] = Tm
            for k, v in st.items():
                r[tag + "." + k] = v
            r[tag + ".coverage"] = np.maximum(e, 0).sum() / Tm if Tm != 0 else np.nan
            r[tag + ".sum_e"] = e.sum()
            eh = np.nanmean(E[ii][:, hd], axis=0)
            sh = pr_stats(eh, len(hd))
            r[tag + ".head_PRn"], r[tag + ".head_top1"] = sh["PRn"], sh["top1"]
            if model == "raasl":
                for nm in ("in.chan", "in.prefix", "in.tok", "in.servo"):
                    r[tag + "." + nm.replace("in.", "share_")] = np.nanmean(E[ii][:, col[nm]]) / Tm if Tm != 0 else np.nan
            for c in ("vlm", "exp"):
                cc = [col[x] for x in U1 if x.startswith(c)]
                r[tag + ".mass_" + c] = np.maximum(np.nanmean(E[ii][:, cc], axis=0), 0).sum() / max(np.maximum(e, 0).sum(), 1e-12)
        # selectivity
        intact = {s: np.nanmean(I[idx][sets[idx] == s]) for s in SETS if (sets[idx] == s).any()}
        for s in intact:
            r["intact." + s] = intact[s]
        for k, b in enumerate(abl):
            for s in intact:
                m = sets[idx] == s
                r["abl.%s.%s" % (b, s)] = np.nanmean(A[idx][m][:, k])
        return r
    pt = stats(np.arange(n))
    reps = [stats(boot_idx(obs, rng)) for _ in range(B_BOOT)]
    keys = list(pt)
    res = {}
    for k in keys:
        res[k] = dict(point=float(pt[k]), ci=ci([r.get(k, np.nan) for r in reps]))
    # drop / selectivity (derived per replicate so the CI is paired)
    sel = {}
    for b in abl:
        for s in ("F7", "N", "K"):
            def drop(r, s=s, b=b):
                i0 = r.get("intact." + s, np.nan)
                return (i0 - r.get("abl.%s.%s" % (b, s), np.nan)) / i0 if i0 and np.isfinite(i0) else np.nan
            low = ci([r.get("intact." + s, np.nan) for r in reps])[0]
            informative = bool(np.isfinite(low) and low > 0)
            sel[(b, s)] = dict(informative=informative,
                               drop=dict(point=float(drop(pt)), ci=ci([drop(r) for r in reps])) if informative else None)
        for s in ("F7", "N"):
            def sd(r, s=s, b=b):
                def dr(ss):
                    i0 = r.get("intact." + ss, np.nan)
                    return (i0 - r.get("abl.%s.%s" % (b, ss), np.nan)) / i0 if i0 and np.isfinite(i0) else np.nan
                return dr(s) - dr("K")
            if sel[(b, s)]["informative"] and sel[(b, "K")]["informative"]:
                sel[(b, s)]["sel"] = dict(point=float(sd(pt)), ci=ci([sd(r) for r in reps]))
    # labels (registered regions)
    lab = {}
    for tag in ("ALL",) + SETS:
        if tag + ".T" not in res:
            continue
        tl = res[tag + ".T"]["ci"][0]
        gate = bool(np.isfinite(tl) and tl > 0)
        lab[tag + ".gate_T_lower_gt_0"] = gate
        if gate:
            lab[tag + ".top1_lt_.3"] = label_in(res[tag + ".top1"]["point"], res[tag + ".top1"]["ci"], hi=TOP1_DIST - 1e-12)
            lab[tag + ".PRn_ge_.2"] = label_in(res[tag + ".PRn"]["point"], res[tag + ".PRn"]["ci"], lo=PRN_DIST)
            lab[tag + ".localised_top1_ge_.5"] = label_in(res[tag + ".top1"]["point"], res[tag + ".top1"]["ci"], lo=TOP1_LOC)
            if model == "raasl":
                lab[tag + ".chan_share_ge_.7"] = label_in(res[tag + ".share_chan"]["point"], res[tag + ".share_chan"]["ci"], lo=CHAN_LOC)
        else:
            lab[tag + ".verdict"] = "UNTESTABLE (total effect not above 0)"
    for (b, s), v in sel.items():
        if "sel" in v:
            c = v["sel"]["ci"]
            v["label"] = "SELECTIVE" if c[0] > MARGIN_SEL else ("INVERSE" if c[1] < -MARGIN_SEL else "NOT SELECTIVE")
    # component profile (point estimates)
    prof = {c: float(np.nanmean(E[:, col[c]])) for c in allc}
    ser = {"%s|%s" % k: v for k, v in sel.items()}
    n_sel = sum(1 for k, v in sel.items() if v.get("label") == "SELECTIVE")
    n_info = sum(1 for k, v in sel.items() if "label" in v)
    doc = dict(model=model, n_obs=n, n_by_set={s: int((sets == s).sum()) for s in SETS}, B=B_BOOT, stats=res, labels=lab, selectivity=ser,
               n_selective_cells=n_sel, n_informative_sel_cells=n_info, profile=prof)
    return doc


def read(model, dirs, out):
    doc = analyze(model, load(dirs))
    if os.path.exists(out):
        raise SystemExit("refusing to overwrite %s" % out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=float)
    show(doc, model)


def show(doc, model):
    r = doc["stats"]
    print("[a2-read] %s n_obs %d by set %s B %d" % (doc["model"], doc["n_obs"], doc["n_by_set"], doc["B"]))
    for tag in ("ALL",) + SETS:
        if tag + ".T" not in r:
            continue
        f = lambda k: "%.3f [%.3f, %.3f]" % (r[tag + "." + k]["point"], r[tag + "." + k]["ci"][0], r[tag + "." + k]["ci"][1])
        s = "[a2-read] %-3s T %s  PRn %s  top1 %s  top3 %s  coverage %s  headPRn %s  head_top1 %s  mass_vlm %s" % (
            tag, f("T"), f("PRn"), f("top1"), f("top3"), f("coverage"), f("head_PRn"), f("head_top1"), f("mass_vlm"))
        print(s)
        if model == "raasl":
            print("[a2-read] %-3s share chan %s prefix %s tok %s servo %s" % (tag, f("share_chan"), f("share_prefix"), f("share_tok"), f("share_servo")))
    print("[a2-read] labels", json.dumps(doc["labels"]))
    for k, v in doc["selectivity"].items():
        if v.get("sel"):
            print("[a2-read] sel %-22s drop %s  sel %s -> %s" % (k, "%.2f [%.2f, %.2f]" % (v["drop"]["point"], *v["drop"]["ci"]),
                                                                  "%.2f [%.2f, %.2f]" % (v["sel"]["point"], *v["sel"]["ci"]), v["label"]))
    print("[a2-read] selective cells %d of %d informative" % (doc["n_selective_cells"], doc["n_informative_sel_cells"]))


def plumb(dirs):
    """Smoke: plumbing numbers only (no effect sizes, no component profile): replay vs the server, identity rows, batch consistency."""
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(p, allow_pickle=False)
            names = [str(n) for n in z["names"]]
            ch = z["chunks"].astype(np.float64)
            ref = z["ref"].astype(int)
            cap = z["cap_actions"].astype(np.float64)
            line = dict(file=os.path.basename(p), rows=len(names), replay_vs_server=float(np.abs(ch[0, :, :7] - cap[:, :7]).max()),
                        nbatch=int(len(set(ref.tolist()))))
            refs = sorted(set(ref.tolist()))
            line["ref_spread"] = float(max(np.abs(ch[r] - ch[refs[0]]).max() for r in refs))
            line["ident"] = {n: float(np.abs(ch[i] - ch[ref[i]]).max()) for i, n in enumerate(names) if n.startswith("ident.")}
            line["finite"] = bool(np.isfinite(ch).all())
            print("[a2-plumb]", json.dumps(line))


def selftest():
    rng = np.random.default_rng(0)
    one = np.zeros(72)
    one[5] = 1.0
    s = pr_stats(one, 72)
    assert abs(s["PRn"] - 1 / 72) < 1e-12 and s["top1"] == 1.0, s
    s = pr_stats(np.ones(72), 72)
    assert abs(s["PRn"] - 1.0) < 1e-12 and abs(s["top1"] - 1 / 72) < 1e-12 and abs(s["top3"] - 3 / 72) < 1e-12, s
    s = pr_stats(np.array([1.0, 1.0, -5.0, 0.0]), 4)
    assert abs(s["PR"] - 2.0) < 1e-12 and abs(s["top1"] - 0.5) < 1e-12, s
    assert np.allclose(unit([3, 0, 4]), [0.6, 0, 0.8]) and not unit([0, 0, 0]).any()
    # bootstrap clusters: F7 pairs stay together, the draw is deterministic
    obs = [dict(meta=dict(set="F7", cluster="F7_%d" % (i // 2))) for i in range(6)] + [dict(meta=dict(set="N", cluster="N_%d" % i)) for i in range(4)]
    a = boot_idx(obs, np.random.default_rng(0))
    b = boot_idx(obs, np.random.default_rng(0))
    assert (a == b).all() and len(a) == 10
    f7 = a[a < 6]
    assert all((x // 2 * 2 in f7) and (x // 2 * 2 + 1 in f7) for x in f7)
    # d5 / projection on a synthetic chunk
    ch = np.zeros((10, 7))
    ch[:5, 0] = 0.2
    d5 = ch[:5, :3].sum(0)
    assert abs(d5 @ unit([1, 0, 0]) - 1.0) < 1e-12
    # the whole analysis on SYNTHETIC observations (no probe output is read): one dominant write, a channel row carrying .9 of the total, a channel ablation that hits N and F7 only
    rng2 = np.random.default_rng(1)
    names = ["ref", "total"] + U1 + HEADS + ["in.chan", "in.prefix", "in.tok", "in.servo"]
    names += ["%s.%d.resid.%s" % (w, li, p) for w in ("vlm", "exp") for li in range(NL) for p in (("all", "ins") if w == "vlm" else ("all",))]
    names += ["vlm.%d.%s.ins" % (li, k) for li in range(NL) for k in ("attn", "mlp")]
    abl = ["abl.vlm.b%d.%s" % (b, p) for b in range(4) for p in ("all", "ins")] + ["abl.exp.b%d" % b for b in range(2)] + ["abl.chan", "abl.tok", "abl.servo"]
    names += abl
    sobs = []
    for i in range(30):
        st = ("F7", "N", "K")[i % 3]
        scf, scor = np.zeros(len(names)), np.ones(len(names))
        ix = {n: j for j, n in enumerate(names)}
        scf[ix["total"]] = 1.0 + 0.05 * rng2.standard_normal()
        for n in U1 + HEADS:
            scf[ix[n]] = 0.01 * rng2.random()
        scf[ix["vlm.17.mlp.all"]] = 0.7
        scf[ix["in.chan"]] = 0.9
        scor[ix["abl.chan"]] = 0.9 if st == "K" else 0.2
        for n in abl:
            if n != "abl.chan":
                scor[ix[n]] = 0.8
        sobs.append(dict(meta=dict(set=st, cluster="%s_%d" % (st, i)), names=names, scf=scf, scor=scor, ref=np.zeros(len(names), int), d5=None))
    doc = analyze("raasl", sobs)
    s = doc["stats"]
    assert s["ALL.top1"]["point"] > 0.5 and doc["labels"]["ALL.localised_top1_ge_.5"] == "SUPPORTED", s["ALL.top1"]
    assert doc["labels"]["ALL.top1_lt_.3"] == "NOT SUPPORTED" and 0.85 < s["ALL.share_chan"]["point"] < 0.95
    assert doc["labels"]["ALL.chan_share_ge_.7"] == "SUPPORTED"
    assert doc["selectivity"]["abl.chan|N"]["label"] == "SELECTIVE" and doc["selectivity"]["abl.chan|F7"]["label"] == "SELECTIVE"
    assert doc["selectivity"]["abl.tok|N"]["label"] == "NOT SELECTIVE"
    assert doc["n_by_set"] == {"F7": 10, "N": 10, "K": 10}
    # a flat profile: distributed
    for o in sobs:
        for n in U1:
            o["scf"][names.index(n)] = 0.014
    doc = analyze("raasl", sobs)
    assert doc["stats"]["ALL.PRn"]["point"] > 0.95 and doc["labels"]["ALL.top1_lt_.3"] == "SUPPORTED" and doc["labels"]["ALL.PRn_ge_.2"] == "SUPPORTED"
    print("[a2-read] selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "read", "plumb"))
    ap.add_argument("--model", default="")
    ap.add_argument("--dirs", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "plumb":
        plumb(a.dirs.split("+"))
    else:
        read(a.model, a.dirs.split("+"), a.out)

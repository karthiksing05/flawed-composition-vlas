"""Fig 6 (probe diagram + dose x servo grid), A13 (probe profile) and A15 (route dot plot).

Outputs (paper/figs): fig_probe.{pdf,png}  (figure*, 6.75 x 2.55 in)
                      figA13.{pdf,png}     (figure*, probe profile of pi0.5 and RAA-sL, bands by set)
                      figA15.{pdf,png}     (single column, routes: where-share and N1-N6 success)
Data (all read from results/flawed, nothing invented):
  results/flawed/a2/a2_read_pi05.json, a2_read_raasl.json        profile, stats (ALL.T, ALL.coverage, ALL.share_chan), selectivity
  results/flawed/raaf/a2/raaf_probe_{A,A0,B,Aplus}.json          rule.where_share (route), ref_shares ALL.in.tok (RAA-sL)
  results/flawed/raaf/raaf_dev_read_v3.json                      phaseA.success["<arm>|N"], ["<arm>|K"]  (points; the file has no intervals)
Hard-coded from DRAFT prose (comment citing the section):
  A+0 probe: results/flawed/raaf/a2/raaf_probe_Aplus0.json (route label "Aplus" is a carried-over string), A+0 dev N/K: raaf_dev_read_v4.json,
  the kept-small cell was not run (DRAFT 6.4 grid), pass bar .70 / fail line .30 (DRAFT 6.4),
  the layer-14 cliff words (DRAFT 4.8), 72 writes = 2 x 18 layers x (attn, mlp) (DRAFT 4.8).
Run: sbatch experiments/fig_b.sh fig_probe.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import fig_style as S  # noqa: E402

FS = 6.5
NL = 18


def jl(*p):
    with open(os.path.join(S.RES, *p)) as f:
        return json.load(f)


def fmt(v):
    """.274 style (no leading zero; minus sign kept)."""
    s = "%.3f" % v
    return s.replace("0.", ".", 1) if abs(v) < 1 else s


def txt(ax, x, y, s, color=S.INK, ha="center", va="center", weight="normal", fs=FS, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=weight, linespacing=1.05, zorder=11, **kw)


# ------------------------------------------------------------------ data
def load():
    D = {}
    D["pi05"] = jl("a2", "a2_read_pi05.json")
    D["raasl"] = jl("a2", "a2_read_raasl.json")
    routes = {}
    for r in ("A", "A0", "B", "Aplus"):
        j = jl("raaf", "a2", "raaf_probe_%s.json" % r)
        w = j["rule"]["where_share"]
        routes[r] = (w["point"], w["ci"][0], w["ci"][1])
    # A+0 (post hoc, prereg tag raafp0): the file's "route" label says "Aplus" by a carried-over string; it IS A+0.
    w = jl("raaf", "a2", "raaf_probe_Aplus0.json")["rule"]["where_share"]
    routes["Aplus0"] = (w["point"], w["ci"][0], w["ci"][1])
    ref = jl("raaf", "a2", "raaf_probe_A.json")["ref_shares"]["ALL.in.tok"]
    routes["raasl"] = (ref["point"], ref["ci"][0], ref["ci"][1])
    D["routes"] = routes
    dev = jl("raaf", "raaf_dev_read_v3.json")["phaseA"]["success"]
    D["dev"] = {a: (dev[a + "|N"], dev[a + "|K"]) for a in ("raasl", "A", "B", "A0", "Aplus", "Actrl")}
    dev4 = jl("raaf", "raaf_dev_read_v4.json")["phaseA"]["success"]
    D["dev"]["Aplus0"] = (dev4["Aplus0|N"], dev4["Aplus0|K"])
    return D


def writes(d):
    names = ["%s.%d.%s.all" % (w, l, k) for w in ("vlm", "exp") for l in range(NL) for k in ("attn", "mlp")]
    e = np.array([d["profile"][n] for n in names])
    return names, e


def band_shares(d):
    """share of the summed positive write effect per band (early/late VLM = layers 0-8 / 9-17; early/late expert likewise)."""
    names, e = writes(d)
    pos = np.maximum(e, 0)
    cov = pos.sum() / d["stats"]["ALL.T"]["point"]
    print("coverage from profile %.3f vs registered %.3f" % (cov, d["stats"]["ALL.coverage"]["point"]))
    out = []
    for w, w_lab in (("vlm", "VLM"), ("exp", "expert")):
        for lo, hi, lab in ((0, 8, "early"), (9, 17, "late")):
            m = [i for i, n in enumerate(names) if n.startswith(w + ".") and lo <= int(n.split(".")[1]) <= hi]
            out.append(("%s %s" % (lab, w_lab), pos[m].sum() / pos.sum()))
    return out


# ------------------------------------------------------------------ Fig 6
def fig_probe(D):
    W, H = S.W2, 2.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    d = D["pi05"]
    T = d["stats"]["ALL.T"]

    # ---------------- left: probe diagram
    txt(ax, 0.03, 2.43, "Activation patching, 72 writes", fs=8, weight="bold", ha="left")
    S.box(ax, (0.03, 1.90), 1.55, 0.34, "clean run\ninstruction names X", "plain", fontsize=FS)
    S.box(ax, (1.78, 1.90), 1.75, 0.34, "counterfactual run\ninstruction names Y", "plain", fontsize=FS)
    # grid of write sites: rows VLM and expert, 36 columns = 18 layers x (attn, mlp)
    gx0, gs, gp = 0.45, 0.064, 0.0735
    rows = {"VLM": 1.57, "expert": 1.44}
    for rname, y in rows.items():
        txt(ax, 0.40, y + gs / 2, rname, ha="right")
        for c in range(36):
            ax.add_patch(Rectangle((gx0 + c * gp, y), gs, gs, fc="#e6e6e6" if c % 2 == 0 else "white", ec=S.GREYINK,
                                   lw=0.4, zorder=2))
    # one highlighted site: expert layer 14, attention (the largest write by mean effect, DRAFT 4.8)
    hc = 2 * 14
    hx, hy = gx0 + hc * gp, rows["expert"]
    ax.add_patch(Rectangle((hx, hy), gs, gs, fc=S.NAVY, ec=S.NAVY, lw=0.8, zorder=4))
    S.arrow(ax, (2.9, 1.90), (hx + gs / 2 + 0.01, hy + gs + 0.01), color=S.CUE, lw=1.4, head=7, rad=-0.25)
    txt(ax, 3.00, 1.76, "paste this write", color=S.CUE, ha="left")
    S.arrow(ax, (0.8, 1.90), (0.8, 1.66), color=S.INK, lw=0.9, head=6)
    txt(ax, 0.86, 1.78, "others stay clean", ha="left")
    txt(ax, gx0 + 36 * gp / 2, 1.35, "layers 0-17, each with attn + MLP", color=S.MUTED)
    txt(ax, 3.18, 1.50, "effect: chunk\nmoves toward Y", ha="left")
    # residual-stream profile (VLM residual patched at layer l)
    axp = fig.add_axes([0.52 / W, 0.66 / H, 1.38 / W, 0.42 / H])
    for mdl, col, lab in (("pi05", S.GREY, "pi0.5"), ("raasl", S.NAVY, "RAA-sL")):
        prof = [D[mdl]["profile"]["vlm.%d.resid.all" % l] for l in range(NL)]
        axp.plot(range(NL), prof, "-o", color=col, ms=1.8, lw=1.0, label=lab)
    axp.axvline(13.5, color=S.CUE, lw=0.8, ls=":")
    axp.set_xlim(-0.5, 17.5)
    axp.set_ylim(-0.1, 1.9)
    axp.set_xticks([0, 6, 12, 17])
    axp.tick_params(labelsize=6.5, length=2, pad=1)
    axp.set_yticks([0, 1])
    axp.set_xlabel("patched VLM residual, layer", fontsize=FS, labelpad=1)
    axp.set_title("layer-14 cliff", fontsize=FS, loc="left", pad=2, color=S.CUE)
    axp.text(1.0, 1.28, "pi0.5", fontsize=6.5, color=S.MUTED, va="center")
    axp.text(1.0, 0.78, "RAA-sL", fontsize=6.5, color=S.NAVY, va="center")
    # per-band share of the positive effect (pi0.5) and the channel inputs of RAA-sL
    axb = fig.add_axes([2.70 / W, 0.62 / H, 1.32 / W, 0.62 / H])
    bs = band_shares(d)
    print("pi0.5 band shares:", ", ".join("%s %.3f" % (b[0], b[1]) for b in bs))
    ch =D["raasl"]["stats"]["ALL.share_chan"]
    labs = [b[0] for b in bs][::-1]
    vals = [b[1] for b in bs][::-1]
    axb.barh(range(len(vals)), vals, color=S.GREY, height=0.7, zorder=3)
    axb.barh(len(vals) + 0.9, ch["point"], color=S.NAVY, height=0.7, zorder=3,
             xerr=[[ch["point"] - ch["ci"][0]], [ch["ci"][1] - ch["point"]]], error_kw=dict(lw=0.8, capsize=1.5))
    axb.set_yticks(list(range(len(vals))) + [len(vals) + 0.9])
    axb.set_yticklabels(labs + ["RAA-sL\nchannel in"], fontsize=6.5)
    axb.set_xlim(0, 1)
    axb.set_xticks([0, .5, 1])
    axb.tick_params(labelsize=6.5, length=2, pad=1)
    axb.set_title("share of positive write effect", fontsize=FS, loc="left", pad=2)
    for i, v in enumerate(vals):
        axb.text(v + 0.02, i, fmt(v), fontsize=6.5, va="center")
    axb.text(0.02, len(vals) + 0.9, fmt(ch["point"]), fontsize=6.5, va="center", ha="left", color="white", zorder=5)
    txt(ax, 0.03, 0.20, "where-share = effect of patching the where-input\n(token, crop) / total effect (pi0.5 total %s [%s, %s])\n"
        "blue bar: RAA-sL servo + state token, 95%% CI"
        % (fmt(T["point"]), fmt(T["ci"][0]), fmt(T["ci"][1])), ha="left")
    ax.plot([4.20, 4.20], [0.1, 2.4], color="#dddddd", lw=0.8)

    # ---------------- right: dose x servo grid
    txt(ax, 4.30, 2.43, "Where-share by data dose and servo", fs=8, weight="bold", ha="left")
    R = D["routes"]
    cx0, cw, ch_, gap = 4.86, 0.58, 0.50, 0.04
    cols = [("none", ""), ("small", "1-9 cm"), ("large", "19-38 cm")]
    for j, (a, b) in enumerate(cols):
        txt(ax, cx0 + j * (cw + gap) + cw / 2, 2.22, a + ("\n" + b if b else "\n "))
    # (row, col) -> (label, share tuple or state)
    cells = {(0, 0): ("RAA-sL", R["raasl"]), (0, 1): ("not run", "notrun"), (0, 2): ("A+", R["Aplus"]),
             (1, 0): ("A0", R["A0"]), (1, 1): ("A", R["A"]), (1, 2): ("A+0", R["Aplus0"])}
    ramp = cm.Blues
    for (r, c), (lab, val) in cells.items():
        x = cx0 + c * (cw + gap)
        y = 1.58 - r * (ch_ + gap)
        if val == "notrun":
            ax.add_patch(Rectangle((x, y), cw, ch_, fc="#f2f2f2", ec=S.GREYINK, lw=0.6, zorder=2))
            txt(ax, x + cw / 2, y + ch_ / 2, "not run", color=S.MUTED)
            continue
        if val == "pending":
            ax.add_patch(Rectangle((x, y), cw, ch_, fc="white", ec=S.INK, lw=0.8, hatch="////", zorder=2))
            txt(ax, x + cw / 2, y + ch_ / 2, "A+0\npending", weight="bold", bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"))
            continue
        v, lo, hi = val
        fcol = ramp(0.08 + 0.80 * min(max(v, 0) / 0.70, 1.0))
        ax.add_patch(Rectangle((x, y), cw, ch_, fc=fcol, ec=S.INK, lw=0.8, zorder=2))
        body = fmt(v) if abs(hi - lo) < 0.02 else "%s\n[%s, %s]" % (fmt(v), fmt(lo), fmt(hi))
        txt(ax, x + cw / 2, y + ch_ / 2, lab + "\n" + body, color=S.INK if v < 0.4 else "white")
        ax.texts[-1].set_fontweight("normal")
    for r, lab in enumerate(("servo\nkept", "servo\noff")):
        txt(ax, cx0 - 0.06, 1.58 - r * (ch_ + gap) + ch_ / 2, lab, ha="right")
    # route B footnote
    b = R["B"]
    txt(ax, 4.30, 0.60, "Route B (crop, no dose): %s [%s, %s]" % (fmt(b[0]), fmt(b[1]), fmt(b[2])), ha="left")
    # colour ramp with the pass bar
    rx0, rw = 4.88, 1.80
    for i in range(60):
        ax.add_patch(Rectangle((rx0 + i * rw / 60, 0.30), rw / 60 + 0.002, 0.10, fc=ramp(0.08 + 0.80 * i / 59), ec="none"))
    ax.add_patch(Rectangle((rx0, 0.30), rw, 0.10, fc="none", ec=S.GREYINK, lw=0.5))
    txt(ax, rx0 - 0.05, 0.35, "share", ha="right")
    txt(ax, rx0, 0.20, "0", va="center")
    txt(ax, rx0 + rw, 0.20, ".70 = pass bar", ha="right")
    ax.plot([rx0 + rw, rx0 + rw], [0.27, 0.43], color=S.INK, lw=1.2)
    S.save(fig, "fig_probe", tight=False)


# ------------------------------------------------------------------ A13
def fig_a13(D):
    fig = plt.figure(figsize=(S.W2, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.35, 1.35], height_ratios=[1.0, 0.85], hspace=0.95, wspace=0.42)
    names, _ = writes(D["pi05"])
    # (a) residual profile
    ax = fig.add_subplot(gs[0, 0])
    for mdl, col, lab in (("pi05", S.GREY, "pi0.5"), ("raasl", S.NAVY, "RAA-sL")):
        prof = [D[mdl]["profile"]["vlm.%d.resid.all" % l] for l in range(NL)]
        ins = [D[mdl]["profile"]["vlm.%d.resid.ins" % l] for l in range(NL)]
        ax.plot(range(NL), prof, "-o", color=col, ms=2.2, lw=1.0, label=lab)
        ax.plot(range(NL), ins, "--", color=col, lw=0.9)
    ax.axvline(13.5, color=S.CUE, lw=0.8, ls=":")
    ax.set_xlabel("VLM residual, layer")
    ax.set_ylabel("effect")
    ax.set_title("a  residual profile", loc="left", fontsize=8)
    ax.plot([], [], "--", color=S.GREYINK, lw=0.9, label="instruction only")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.4, borderpad=0.1, labelspacing=0.15)
    ax.set_ylim(-0.05, 2.3)
    # (b) the 72 writes, share of the total effect
    for k, (w, lab) in enumerate((("vlm", "VLM"), ("exp", "action expert"))):
        ax = fig.add_subplot(gs[0, 1 + k])
        for mdl, col, off in (("pi05", S.GREY, -0.2), ("raasl", S.NAVY, 0.2)):
            T = D[mdl]["stats"]["ALL.T"]["point"]
            for kind, hatch in (("attn", None), ("mlp", "////")):
                e = [D[mdl]["profile"]["%s.%d.%s.all" % (w, l, kind)] / T for l in range(NL)]
                xs = np.arange(NL) + off + (0.0 if kind == "attn" else 0.0)
                ax.bar(xs, e, width=0.38, color=col, hatch=hatch, edgecolor="white" if hatch is None else col,
                       lw=0.0 if hatch is None else 0.3, alpha=1.0 if kind == "attn" else 0.55, zorder=3)
        ax.set_xlabel("%s layer" % lab)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("%s  %s writes / total effect" % ("bc"[k], lab), loc="left", fontsize=8)
        ax.set_ylim(-0.15, 0.48)
        if k == 0:
            ax.set_ylabel("share of total")
        if k == 1:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(fc=S.GREY, label="pi0.5 attn"), Patch(fc=S.GREY, alpha=.55, hatch="////", ec=S.GREY, label="pi0.5 MLP"),
                               Patch(fc=S.NAVY, label="RAA-sL attn"), Patch(fc=S.NAVY, alpha=.55, hatch="////", ec=S.NAVY, label="RAA-sL MLP")],
                      fontsize=6.0, frameon=False, ncol=2, loc="upper right", handlelength=1.2, columnspacing=0.8, borderpad=0.1)
    # (d) band-by-set cells (pi0.5): drop of the correct-target displacement when the band's writes are zeroed
    ax = fig.add_subplot(gs[1, :])
    sel = D["pi05"]["selectivity"]
    bands = [("abl.vlm.b0.all", "VLM 0-3\nall"), ("abl.vlm.b0.ins", "VLM 0-3\ninstr."), ("abl.vlm.b1.all", "VLM 4-8\nall"),
             ("abl.vlm.b1.ins", "VLM 4-8\ninstr."), ("abl.vlm.b2.all", "VLM 9-13\nall"), ("abl.vlm.b2.ins", "VLM 9-13\ninstr."),
             ("abl.vlm.b3.all", "VLM 14-17\nall"), ("abl.vlm.b3.ins", "VLM 14-17\ninstr."), ("abl.exp.b0", "expert\n0-8"),
             ("abl.exp.b1", "expert\n9-17")]
    sets = ["F7", "K"]
    M = np.full((2, len(bands)), np.nan)
    L = [[""] * len(bands) for _ in sets]
    for j, (b, _) in enumerate(bands):
        for i, s in enumerate(sets):
            c = sel["%s|%s" % (b, s)]
            if c["informative"]:
                M[i, j] = c["drop"]["point"]
                L[i][j] = fmt(c["drop"]["point"])
    im = ax.imshow(np.clip(M, -1, 1), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b[1] for b in bands], fontsize=6.5)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["F7 (relational\nlanguage)", "K (trained\ntasks)"], fontsize=6.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(2):
        for j in range(len(bands)):
            if L[i][j]:
                ax.text(j, i, L[i][j], ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 else S.INK)
    for j, (b, _) in enumerate(bands):
        lab = sel["%s|F7" % b]["label"]
        ax.text(j, -0.72, lab.replace("NOT SELECTIVE", "not\nselective").lower() if lab else "", ha="center", va="center",
                fontsize=6.5, color=S.FAIL if lab == "INVERSE" else S.MUTED, clip_on=False)
    ax.set_title("d  zeroing a band's writes: drop of the correct-target displacement (negative = it rises)",
                 loc="left", fontsize=8, pad=22)
    S.save(fig, "figA13")
    print("A13 selective cells:", D["pi05"]["n_selective_cells"], "informative:", D["pi05"]["n_informative_sel_cells"])


# ------------------------------------------------------------------ A15
def fig_a15(D):
    fig, axes = plt.subplots(1, 2, figsize=(S.W1, 2.45), sharey=True, gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.12})
    rows = [("RAA-sL", "raasl", "raasl"), ("A0", "A0", "A0"), ("B", "B", "B"), ("A", "A", "A"), ("A+", "Aplus", "Aplus"),
            ("A+ctrl", "none", "Actrl"), ("A+0", "Aplus0", "Aplus0")]
    y = {r[0]: len(rows) - 1 - i for i, r in enumerate(rows)}
    ax = axes[0]
    R = D["routes"]
    for lab, rk, _ in rows:
        if rk == "none":
            ax.text(0.02, y[lab], "no where-input", fontsize=6.5, color=S.MUTED, va="center", style="italic")
            continue
        if rk is None:
            ax.text(0.02, y[lab], "pending", fontsize=6.5, color=S.MUTED, va="center", style="italic")
            continue
        v, lo, hi = R[rk]
        col = S.ARM["RAA-sL"] if rk == "raasl" else S.ARM["RAA-c"]
        ax.errorbar(v, y[lab], xerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=3.5, capsize=1.5, lw=1.0)
    ax.axvline(0.70, color=S.OK, lw=0.9, ls="--")
    ax.axvline(0.30, color=S.FAIL, lw=0.9, ls=":")
    ax.text(0.70, len(rows) - 0.45, "pass .70", fontsize=6.5, color=S.OK, ha="center", va="bottom",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.text(0.30, len(rows) - 0.45, "fail .30", fontsize=6.5, color=S.FAIL, ha="center", va="bottom",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xlim(-0.08, 0.95)
    ax.set_xticks([0, .5])
    ax.set_xlabel("learned where-share", fontsize=7)
    ax.set_yticks([y[r[0]] for r in rows])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) + 0.2)
    ax = axes[1]
    for lab, rk, dk in rows:
        if dk is None:
            ax.text(0.02, y[lab], "pending", fontsize=6.5, color=S.MUTED, va="center", style="italic")
            continue
        n, k = D["dev"][dk]
        ax.plot(k, y[lab], "s", color=S.GREYINK, mfc="white", ms=3.5)
        ax.plot(n, y[lab], "o", color=S.ARM["RAA-sL"] if rk == "raasl" else S.ARM["RAA-c"], ms=3.5)
    ax.plot([], [], "o", color=S.ARM["RAA-c"], label="N1-N6")
    ax.plot([], [], "s", color=S.GREYINK, mfc="white", label="K1-K6")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.2, borderpad=0.1)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("success (dev. states)", fontsize=7)
    ax.tick_params(left=False)
    ax.set_xticks([0, .5, 1])
    for a in axes:
        a.grid(True, axis="y", color="#eeeeee", lw=0.5)
    S.save(fig, "figA15")


if __name__ == "__main__":
    D = load()
    fig_probe(D)
    fig_a13(D)
    fig_a15(D)

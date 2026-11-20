"""Paper figures for paper/DRAFT.md.  Writes PDF + PNG to paper/figs/.

Reads only committed result JSONs under results/flawed/ and numbers that exist only
in doc tables (Table 1 of paper/DRAFT.md, which cites docs/ARCH_FAILURES.md and
docs/SCAFFOLD_RESULTS.md).  Every hard-coded number carries its source in a comment.
Run on a compute node:  python experiments/paper_figs.py   (numpy + matplotlib only).
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "flawed")
OUT = os.path.join(REPO, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

# ---------------------------------------------------------------- palette
# Frozen pi0.5 grey, ReGuide black, our arms one blue hue family (distinguished
# by shade and marker so the figure also reads in greyscale).
GREY, BLACK = "#8c8c8c", "#000000"
ARM = {  # key: (label, colour, marker, filled)
    "I":     ("frozen pi0.5", GREY, "o", True),
    "r0":    ("R0", "#c6dbef", "o", True),
    "r2":    ("R2", "#6baed6", "s", True),
    "raa":   ("RAA", "#2171b5", "o", True),
    "raao":  ("RAA-o", "#08519c", "^", True),
    "raac":  ("RAA-c", "#4292c6", "v", True),
    "raasb": ("RAA-s", "#08306b", "D", True),
    "raal":  ("RAA-L", "#2171b5", "o", False),
    "raasl": ("RAA-sL", "#08306b", "D", False),
    "rgp":   ("ReGuide-perc", BLACK, "P", True),
}
# model colours for the atlas (pi0.5 grey to match the frozen reference)
MODEL = {"pi0.5": GREY, "pi0": "#E69F00", "GR00T N1": "#009E73",
         "X-VLA": "#CC79A7", "SmolVLA": "#D55E00"}


def jload(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion with k successes of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, name + "." + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# =================================================================== fig 1
def fig_atlas():
    # Source: paper/DRAFT.md Table 1, which cites docs/ARCH_FAILURES.md (five-model
    # atlas, fresh states) and docs/SCAFFOLD_RESULTS.md SR 45.1, 45.3, 45.6, 50, 51, 53.
    models = ["pi0.5", "pi0", "GR00T N1", "X-VLA", "SmolVLA"]
    # (value, lo, hi); None = no read
    f1_s1 = {"pi0.5": (.450, .28, .61), "pi0": (.225, .05, .43), "GR00T N1": (.500, .26, .74),
             "X-VLA": (.379, .04, .73), "SmolVLA": (.054, .0, .14)}
    f1_s2 = {"pi0.5": (1.000, 1.0, 1.0), "pi0": (.908, .792, 1.0), "GR00T N1": (.533, .167, .858)}
    f2 = {"pi0.5": (.0, .0, .0), "pi0": (.650, .39, .88), "GR00T N1": (.017, .0, .07),
          "X-VLA": (.192, .0, .53), "SmolVLA": (.833, .67, .96)}
    # F6 cost d - b (SR 45.6); pi0 not testable (gate .017, SR 70)
    f6 = {"pi0.5": (.38, .28, .48), "GR00T N1": (.60, .51, .68)}
    # F1m high-carry share of grasped N4/N6 episodes (SR 51); pi0.5 .225 is the
    # intact share quoted in section 4.2 C3 (SR 53).  GR00T is a lower bound (>= .78),
    # pi0 a range (.67-1.00), X-VLA 1.00.  No intervals in the record.
    f1m_pt = {"pi0.5": .225, "X-VLA": 1.00}
    f1m_lb = {"GR00T N1": .78}
    f1m_rng = {"pi0": (.67, 1.00)}
    # F7 relational choice equals ambiguous choice (SR 45.1, 45.3): SP and NG families
    f7_pt = {"pi0.5": (.925, 1.00), "GR00T N1": (1.00, 1.00)}
    f7_rng = (.75, 1.00)  # pi0, X-VLA, SmolVLA: agreement 75-100% on 4 of 6 cells (SR 45.3)

    fig, axes = plt.subplots(1, 5, figsize=(7.6, 2.7), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1]})
    y = {m: len(models) - 1 - i for i, m in enumerate(models)}

    def ci(ax, d, off=0.0, marker="o", filled=True, size=4.5):
        for m, (v, lo, hi) in d.items():
            c = MODEL[m]
            ax.errorbar(v, y[m] + off, xerr=[[v - lo], [hi - v]], fmt=marker, color=c,
                        mfc=c if filled else "white", mec=c, ms=size, capsize=2, lw=1.1)

    ax = axes[0]
    ci(ax, f1_s1, off=0.13)
    ci(ax, f1_s2, off=-0.13, marker="s", filled=False)
    ax.set_title("F1 place\nexcess failure E")
    ax.set_xlim(-0.08, 1.08)
    ax.plot([], [], "o", color="k", label="scene 1")
    ax.plot([], [], "s", color="k", mfc="white", label="scene 2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8)
    ax.set_yticks([y[m] for m in models])
    ax.set_yticklabels(models)
    ax.axvline(0, color="k", lw=0.6)

    ax = axes[1]
    for m, v in f1m_pt.items():
        ax.plot(v, y[m], "o", color=MODEL[m], ms=4.5)
    for m, v in f1m_lb.items():
        ax.plot(v, y[m], ">", color=MODEL[m], ms=5)
    for m, (lo, hi) in f1m_rng.items():
        ax.plot([lo, hi], [y[m]] * 2, "-", color=MODEL[m], lw=3, solid_capstyle="butt")
    ax.set_title("F1m carry\nhigh-carry share")
    ax.set_xlim(-0.05, 1.08)
    ax.text(0.02, 0.02, "SmolVLA: no read\n> lower bound", transform=ax.transAxes, fontsize=6.5, color="#555555")

    ax = axes[2]
    ci(ax, f2)
    ax.set_title("F2 grasp\nexcess failure E")
    ax.set_xlim(-0.08, 1.08)
    ax.axvline(0, color="k", lw=0.6)

    ax = axes[3]
    for m, (a, b) in f7_pt.items():
        ax.plot(a, y[m] + 0.12, "o", color=MODEL[m], ms=4.5)
        ax.plot(b, y[m] - 0.12, "s", color=MODEL[m], ms=4.5, mfc="white")
    for m in ("pi0", "X-VLA", "SmolVLA"):
        ax.plot(list(f7_rng), [y[m]] * 2, "-", color=MODEL[m], lw=3, solid_capstyle="butt", alpha=0.85)
    ax.plot([], [], "o", color="k", label="SP")
    ax.plot([], [], "s", color="k", mfc="white", label="NG")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8)
    ax.set_title("F7 relation\nchoice = ambiguous")
    ax.set_xlim(0.5, 1.05)

    ax = axes[4]
    ci(ax, f6)
    ax.text(0.02, y["pi0"], "not testable", fontsize=6.5, color="#555555", va="center")
    ax.set_title("F6 hidden target\ncost d - b")
    ax.set_xlim(-0.05, 0.85)

    for ax in axes:
        ax.set_ylim(-0.6, len(models) - 0.4)
    for ax in axes[1:]:
        ax.tick_params(left=False)
    save(fig, "fig_atlas")


# =================================================================== fig 2
def per_task(fname, arm, tasks):
    d = jload(fname)
    n = len(d["states"]["dev"]) + len(d["states"]["new"])
    out = {}
    for t in tasks:
        key = "%s|%s" % (arm, t)
        out[t] = d["per_task"][key]["success"]
    pooled = {}
    return out, n, d


def fig_arms():
    # (arm key, source file, key inside file)
    srcs = [("r0", "raa_read.json", "r0"), ("r2", "raa_read.json", "r2"),
            ("raa", "raa_read.json", "raa"), ("raao", "raao_read.json", "raao"),
            ("raac", "raac_read.json", "raac"), ("raasb", "raas_b_read.json", "raasb"),
            ("raal", "raal_read.json", "raal"), ("raasl", "raasl_read.json", "raasl"),
            ("rgp", "raa_read.json", "rgp")]
    N = ["N1", "N2", "N3", "N4", "N5", "N6"]
    K = ["K1", "K2", "K3", "K4", "K5", "K6"]
    cache = {}
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharey=True)
    fig.subplots_adjust(hspace=0.4)
    for ax, tasks, pool, title in ((axes[0], N, "N", "New pairings N1-N6"),
                                   (axes[1], K, "K", "Trained tasks K1-K6")):
        cats = tasks + [pool]
        xs = np.arange(len(cats), dtype=float)
        xs[-1] += 0.5  # gap before the pooled column
        na = len(srcs)
        w = 0.78 / na
        for i, (arm, fn, key) in enumerate(srcs):
            if fn not in cache:
                cache[fn] = jload(fn)
            d = cache[fn]
            n = len(d["states"]["dev"]) + len(d["states"]["new"])
            vals = [d["per_task"]["%s|%s" % (key, t)]["success"] for t in tasks]
            pooled = d["success"]["%s|%s" % (key, pool)]
            vals.append(pooled)
            ns = [n] * len(tasks) + [n * len(tasks)]
            lo, hi = zip(*[wilson(v * m, m) for v, m in zip(vals, ns)])
            lab, col, mk, filled = ARM[arm]
            xx = xs + (i - (na - 1) / 2) * w
            ax.errorbar(xx, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                        fmt=mk, color=col, mfc=col if filled else "white", mec=col, ms=3.8,
                        capsize=1.2, lw=0.8, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels(tasks + ["pooled"])
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("success")
        ax.axvline(xs[-2] + 0.75, color="#cccccc", lw=0.8)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.42), frameon=False,
                   columnspacing=1.0, handletextpad=0.2)
    save(fig, "fig_arms")


# =================================================================== fig 3
def fig_removal():
    files = [("SR 80  RAA-L", "raal_wa_read.json", ARM["raal"][1]),
             ("SR 82  RAA-sL", "raasl_wa_read.json", ARM["raasl"][1])]
    data = [(lab, jload(fn), col) for lab, fn, col in files]
    panels = [("D1  F7 relational: full vs noR", "F7", "noR", "remove resolver R"),
              ("D2  F6 hidden target: full vs noM", "F6_b", "noM", "remove pose memory M")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (title, cell, rem, rem_lab) in zip(axes, panels):
        for gi, (lab, d, col) in enumerate(data):
            for j, (arm, hatch, alpha) in enumerate((("full", None, 1.0), (rem, "////", 0.55))):
                v, lo, hi = d["success"]["%s|%s" % (arm, cell)]
                x = gi + (j - 0.5) * 0.36
                ax.bar(x, v, width=0.34, color=col, alpha=alpha, hatch=hatch, edgecolor="white" if hatch is None else col,
                       linewidth=0.5)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", capsize=2, lw=0.9)
        # frozen pi0.5 reference (the I arm in the same read)
        ref = data[0][1]["success"]["I|%s" % cell]
        ax.axhline(ref[0], color=GREY, lw=1.4, ls="--")
        ax.axhspan(ref[1], ref[2], color=GREY, alpha=0.15, lw=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([l for l, _, _ in data])
        ax.set_xlim(-0.55, 1.65)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.0)
        ax.bar([], [], color="#555555", label="full")
        ax.bar([], [], color="#555555", alpha=0.55, hatch="////", label=rem_lab)
        ax.plot([], [], color=GREY, lw=1.4, ls="--", label="frozen pi0.5")
        ax.legend(loc="upper right", frameon=False, handlelength=1.4, fontsize=7)
    axes[0].set_ylabel("success")
    # sanity: both reads must carry the same frozen reference
    for cell in ("F7", "F6_b"):
        a = data[0][1]["success"]["I|%s" % cell][0]
        b = data[1][1]["success"]["I|%s" % cell][0]
        assert abs(a - b) < 1e-9, (cell, a, b)
    save(fig, "fig_removal")


# =================================================================== fig 4
def fig_moved():
    d = jload("mt_read.json")["cells"]
    arms = ["I", "r2", "raa", "rgp"]
    cells = [("K1_in", "K1\nin range"), ("T_near", "K1 plate\nnear"), ("T_far", "K1 plate\nfar"),
             ("K2_in", "K2\nin range"), ("O1_near", "K1 bowl\nnear"), ("O1_far", "K1 bowl\nfar"),
             ("O2_near", "K2 bowl\nnear"), ("O2_far", "K2 bowl\nfar")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        lab, col, mk, filled = ARM[a]
        vals = [d[a][c]["success"] for c, _ in cells]
        v = np.array([x[0] for x in vals])
        lo = np.array([x[1] for x in vals])
        hi = np.array([x[2] for x in vals])
        xx = np.arange(len(cells)) + (i - (len(arms) - 1) / 2) * w
        ax.errorbar(xx, v, yerr=[v - lo, hi - v], fmt=mk, color=col, ms=4, capsize=1.5, lw=0.9, label=lab)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([l for _, l in cells], fontsize=6.8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("success")
    ax.set_title("Success by cell (40 states each)", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False, columnspacing=1.0,
              handletextpad=0.2)
    ax = axes[1]
    tc = [("K1_in", "in range"), ("T_near", "plate\nnear"), ("T_far", "plate\nfar")]
    for i, a in enumerate(arms):
        lab, col, mk, filled = ARM[a]
        v = np.array([d[a][c]["share_all"][0] for c, _ in tc])
        lo = np.array([d[a][c]["share_all"][1] for c, _ in tc])
        hi = np.array([d[a][c]["share_all"][2] for c, _ in tc])
        xx = np.arange(len(tc)) + (i - (len(arms) - 1) / 2) * w * 1.6
        ax.errorbar(xx, v, yerr=[v - lo, hi - v], fmt=mk, color=col, ms=4, capsize=1.5, lw=0.9)
    ax.set_xticks(range(len(tc)))
    ax.set_xticklabels([l for _, l in tc], fontsize=7)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("share")
    ax.set_title("Release at trained plate position", loc="left", fontsize=8)
    save(fig, "fig_moved")


if __name__ == "__main__":
    fig_atlas()
    fig_arms()
    fig_removal()
    fig_moved()

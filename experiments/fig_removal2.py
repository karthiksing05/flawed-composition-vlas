"""Fig 5 (re-cut of paper_figs.fig_removal): module removal, D1 (resolver R, F7) and D2 (pose memory M, F6), 1 x 2.

Per docs/notes/fig_plan.md section 7 the W5 where-module panel is NOT here (appendix A14).  Four groups per panel:
two development-state reads (SR 80 RAA-L, SR 82 RAA-sL) and the two fresh-state reads of the module-owned policy
RAA-sL-W5 (SR 87 and SR 91).  Bars: full vs the policy with the module removed (hatched); grey tick = frozen pi0.5 on the
same states; error bars = registered 95% intervals.  The SR 90 read (W6, third fresh set) is in Table 2c and not drawn.
Data (success = [point, lo, hi]):
  results/flawed/raal_wa_read.json, raasl_wa_read.json, raasw5_wa_read_w5.json, raasw5r_wa_read_w5.json
  cells F7 (D1, noR) and F6_b (D2, noM); frozen = arm I.
Verdict words are the registered ones in DRAFT 6.2 (D1 SUPPORTED everywhere; D2 NOT SUPPORTED on RAA-L, REVERSED on RAA-sL,
SUPPORTED on both fresh sets); the gated primaries read from raasw5_wa_gated.json / raasw5r_wa_gated.json are asserted.
Run: sbatch experiments/fig_b.sh fig_removal2.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import fig_style as S  # noqa: E402


PAPER = False   # --paper: anonymous-review variant (set 1 / set 3 labels, "held-out"), written to paper/figs/paper/


def jl(name):
    with open(os.path.join(S.RES, name)) as f:
        return json.load(f)


def main():
    groups = [  # label, file, colour, verdicts D1, D2 (DRAFT 6.2)
        ("RAA-L\nSR 80", "raal_wa_read.json", S.ARM["RAA-L"], "SUPPORTED", "NOT SUPP."),
        ("RAA-sL\nSR 82", "raasl_wa_read.json", S.ARM["RAA-sL"], "SUPPORTED", "REVERSED"),
        ("W5\nSR 87", "raasw5_wa_read_w5.json", S.ARM["RAA-sL"], "SUPPORTED", "SUPPORTED"),
        ("W5\nSR 91", "raasw5r_wa_read_w5.json", S.ARM["RAA-sL"], "SUPPORTED", "SUPPORTED"),
    ]
    if PAPER:
        groups = [("RAA-L", ) + groups[0][1:], ("RAA-sL", ) + groups[1][1:],
                  ("W5\nset 1", ) + groups[2][1:], ("W5\nset 3", ) + groups[3][1:]]
    data = [jl(g[1]) for g in groups]
    for fn in ("raasw5_wa_gated.json", "raasw5r_wa_gated.json"):
        assert jl(fn)["W5"]["primary_gated"] == "DISSOCIATION SHOWN", fn
    panels = [("D1  F7: remove resolver R", "F7", "noR", 1),
              ("D2  F6: remove pose memory M", "F6_b", "noM", 2)]
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 2.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (title, cell, rem, k) in zip(axes, panels):
        for gi, (g, d) in enumerate(zip(groups, data)):
            col = g[2]
            fresh = gi >= 2
            for j, (arm, hatch, alpha) in enumerate((("full", None, 1.0), (rem, "////", 0.5))):
                v, lo, hi = d["success"]["%s|%s" % (arm, cell)]
                x = gi + (j - 0.5) * 0.36
                ax.bar(x, v, width=0.34, color=col, alpha=alpha, hatch=hatch,
                       edgecolor="white" if hatch is None else col, linewidth=0.5, zorder=3)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", capsize=1.6, lw=0.8, zorder=4)
            full = d["success"]["full|" + cell][0]
            wo = d["success"]["%s|%s" % (rem, cell)][0]
            ref = d["success"]["I|" + cell][0]
            ax.plot([gi - 0.40, gi + 0.40], [ref, ref], color=S.GREY, lw=1.6, ls="--", zorder=5)
            ax.text(gi, 1.015, "%s\n%+.2f" % (g[3 + k - 1], full - wo), ha="center", va="bottom", fontsize=6.5,
                    color=S.INK if (fresh or g[3 + k - 1] == "SUPPORTED") else S.FAIL, linespacing=1.0)
            print(cell, g[0].replace("\n", " "), "full %.3f %s %.3f diff %+.3f frozen %.3f" % (full, rem, wo, full - wo, ref))
        ax.axvline(1.5, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, loc="left", fontsize=8, pad=22)
        for xc, lab in ((0.5, "development states"), (2.5, "held-out sets" if PAPER else "fresh states")):
            ax.annotate(lab, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -27), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color=S.MUTED)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("success")
    sw = [Patch(facecolor="#555555", edgecolor="white", label="full"),
          Patch(facecolor="#555555", alpha=0.5, hatch="////", edgecolor="#555555", label="module removed"),
          Line2D([], [], color=S.GREY, lw=1.6, ls="--", label="frozen pi0.5, same states")]
    fig.legend(handles=sw, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.80, bottom=0.30)
    S.save(fig, "fig_removal2", sub="paper" if PAPER else None, tight=False)


if __name__ == "__main__":
    PAPER = "--paper" in sys.argv[1:]
    main()

"""Fig 4: architecture figure (docs/notes/fig_plan.md section 4, "Architecture figure spec").

Left: the standard distributed pi0.5 (openpi pi05_libero: SigLIP -> 18-layer VLM prefix with the language tokens ->
prefix KV cache -> 18-layer flow-matching action expert -> chunk of 10 steps x 7-D, 5 executed per query).
Right: the de-distributed proposal (DRAFT sections 5.3 and 6.1-6.3): parse + resolver R, pose memory M, role channel,
where-module (servo; RAA-sL-W5, closed-gripper command W2), the same frozen network.

No data are plotted.  Numbers drawn and their sources:
  layer-14 keys/values read site                  DRAFT 4.8 (SR 83), hard-coded
  state "not read in the served model"            DRAFT 4.2 (SR 53); openpi pi05_libero discrete_state_input=False
  state-token share -.007 (dotted arrow)          DRAFT 6.3 (SR 83); asserted against results/flawed/raaf/a2/raaf_probe_A.json ref_shares ALL.in.tok
  servo-input share .878 (solid arrow)            DRAFT 6.3; asserted against results/flawed/a2/a2_read_raasl.json stats ALL.share_servo
  chunk 10 steps, 5 executed                      openpi config pi05_libero action_horizon=10; examples/libero main.py replan_steps=5
Run:  sbatch experiments/fig_b.sh fig_arch.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
import fig_style as S  # noqa: E402

W, H = S.W2, 3.45
FS = 6.5
NX, NW = 0.85, 1.30            # network column
LW = 0.62                      # language / images box width
RX, RW = 2.35, 0.95            # right column
DOT = (0, (1.2, 1.8))


def tag(ax, x, y, text):
    ax.text(x, y, text, fontsize=FS, color="white", ha="center", va="center", fontweight="bold", zorder=12,
            bbox=dict(boxstyle="round,pad=0.18", fc=S.FAIL, ec="white", lw=0.6))


def txt(ax, x, y, s, color=S.INK, ha="center", va="center", weight="normal", fs=FS):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=weight, linespacing=1.05, zorder=11)


def panel(ax, X0, right):
    P = lambda x, y: (X0 + x, y)

    def bx(x, y, w, h, label, kind):
        return S.box(ax, P(x, y), w, h, label, kind, fontsize=FS)

    def ar(p0, p1, **kw):
        return S.arrow(ax, P(*p0), P(*p1), **kw)

    cx = NX + NW / 2
    # inputs
    bx(NX, 0.30, LW, 0.26, "language", "input")
    bx(NX + NW - LW, 0.30, LW, 0.26, "images\n(2 views)", "input")
    bx(RX, 0.30, RW, 0.26, "state:\nnot read", "ghost")
    ar((RX, 0.43), (NX + NW + 0.05, 0.43), color=S.GREYINK, lw=0.8, ls=(0, (2, 2)), head=5)
    S.cross(ax, (X0 + NX + NW + 0.12, 0.43), size=0.035, lw=1.2, color=S.GREYINK)
    # SigLIP
    bx(NX + NW - LW, 1.02, LW, 0.18, "SigLIP", "frozen" if right else "released")
    # VLM prefix
    bx(NX, 1.30, NW, 0.62, "", "frozen" if right else "released")
    txt(ax, X0 + cx, 1.76, "VLM prefix", weight="bold")
    txt(ax, X0 + cx, 1.52, "18 layers, frozen\n+ language tokens" if right else "18 layers\n+ language tokens")
    # action expert
    bx(NX, 2.24, NW, 0.52, "", "expert" if right else "released")
    txt(ax, X0 + cx, 2.62, "action expert", weight="bold")
    txt(ax, X0 + cx, 2.42, "18 layers, flow\nmatching" if not right else "18 layers; keeps\nthe motor skill")
    # output chunk
    bx(NX, 2.90, NW, 0.24, "10 x 7-D chunk", "plain")
    ar((cx, 2.76), (cx, 2.90), lw=1.0)
    # KV cache arrow (the narrow read site)
    ar((cx, 1.92), (cx, 2.24), color=S.NAVY, lw=2.6, head=7)
    txt(ax, X0 + cx - 0.10, 2.08, "layer-14 K, V:\nnarrow read site", color=S.NAVY, ha="right")
    # image path
    if not right:
        ar((NX + NW - LW / 2, 0.56), (NX + NW - LW / 2, 1.02), lw=0.9)
        ar((NX + LW / 2, 0.56), (NX + LW / 2, 1.30), lw=0.9)
    ar((NX + NW - LW / 2, 1.20), (NX + NW - LW / 2, 1.30), lw=0.9)

    if not right:
        # failure tags: phase column on the left, tags on the right at the same heights
        ys = (3.02, 2.70, 2.38)
        for lab, y, name in zip(("(a) grasp", "(b) carry", "(c) release"), ys, ("F2", "F1m", "F1")):
            txt(ax, X0 + 0.02, y, lab, ha="left")
            tag(ax, X0 + RX + 0.25, y, name)
            ar((RX + 0.10, y), (NX + NW + 0.02, 3.02 if name == "F2" else 2.55), color=S.FAIL, lw=0.8, head=5)
        ar((0.80, 3.06), (0.80, 2.30), color=S.GREYINK, lw=1.0, head=6)
        txt(ax, X0 + RX + 0.50, 2.04, "nothing in the\nnet owns where\nthe arm goes", color=S.FAIL, ha="left")
        # F7, F6
        tag(ax, X0 + 0.30, 0.82, "F7")
        txt(ax, X0 + 0.30, 0.60, "no relation\nstep", color=S.FAIL)
        ar((0.52, 0.82), (NX + LW / 2 - 0.02, 0.82), color=S.FAIL, lw=0.8, head=5)
        tag(ax, X0 + RX + 0.25, 0.82, "F6")
        txt(ax, X0 + RX + 0.50, 0.82, "no memory\nacross queries", color=S.FAIL, ha="left")
        ar((RX + 0.08, 0.82), (NX + NW - LW / 2 + 0.02, 0.82), color=S.FAIL, lw=0.8, head=5)
    else:
        # parse + resolver R on the language path, pose memory M on the image path
        bx(NX, 0.66, LW, 0.34, "parse +\nresolver R", "new")
        bx(NX + NW - LW, 0.66, LW, 0.34, "pose\nmemory M", "new")
        ar((NX + LW / 2, 0.56), (NX + LW / 2, 0.66), lw=0.9)
        ar((NX + LW / 2, 1.00), (NX + LW / 2, 1.30), lw=0.9)
        ar((NX + NW - LW / 2, 0.56), (NX + NW - LW / 2, 0.66), lw=0.9)
        ar((NX + NW - LW / 2, 1.00), (NX + NW - LW / 2, 1.02), lw=0.9)
        tag(ax, X0 + 0.30, 0.90, "F7")
        txt(ax, X0 + 0.30, 0.68, "returns if\nR removed", color=S.FAIL)
        ar((0.52, 0.90), (NX - 0.02, 0.86), color=S.FAIL, lw=0.8, head=5)
        tag(ax, X0 + RX + 0.25, 0.90, "F6")
        txt(ax, X0 + RX + 0.42, 0.94, "returns if", color=S.FAIL, ha="left")
        txt(ax, X0 + RX + 0.42, 0.82, "M removed", color=S.FAIL, ha="left")
        ar((RX + 0.08, 0.90), (NX + NW + 0.02, 0.86), color=S.FAIL, lw=0.8, head=5)
        # role channel, where-module
        bx(RX, 1.42, RW, 0.62, "role channel:\ntarget - (eef\n+ in-hand)", "new")
        bx(RX, 2.42, RW, 0.70, "where-module\n(servo): owns\ncarry xy, release\ngate; W2: grip\nclosed", "new")
        ar((RX + 0.30, 2.04), (RX + 0.30, 2.42), color=S.NAVY, lw=1.5, head=7)
        txt(ax, X0 + RX + 0.38, 2.23, ".878", color=S.NAVY, ha="left")
        # state token: dotted arrow into the network, not read
        ar((RX, 1.76), (NX + NW + 0.01, 2.46), color=S.NAVY, lw=1.2, ls=DOT, head=6, rad=-0.30)
        txt(ax, X0 + RX + RW / 2, 1.22, "dotted: state token,\nshare -.007", color=S.NAVY)
        # where-module writes the chunk's xy
        ar((RX, 2.99), (NX + NW + 0.01, 3.02), color=S.NAVY, lw=1.6, head=7)
        tag(ax, X0 + RX + RW / 2, 3.24, "F1  F1m")


def main():
    a2 = json.load(open(os.path.join(S.RES, "a2", "a2_read_raasl.json")))["stats"]
    servo = a2["ALL.share_servo"]["point"]
    ref = json.load(open(os.path.join(S.RES, "raaf", "a2", "raaf_probe_A.json")))["ref_shares"]["ALL.in.tok"]["point"]
    assert abs(servo - 0.878) < 0.0006 and abs(ref - (-0.007)) < 0.0006, (servo, ref)
    print("servo share %.4f, state-token share %.4f" % (servo, ref))

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    PX = [0.02, 3.44]
    panel(ax, PX[0], right=False)
    panel(ax, PX[1], right=True)
    txt(ax, PX[0] + 0.02, 3.36, "Standard pi0.5: distributed", fs=8, weight="bold", ha="left")
    txt(ax, PX[1] + 0.02, 3.36, "De-distributed proposal", fs=8, weight="bold", ha="left")
    ax.plot([3.38, 3.38], [0.2, 3.4], color="#dddddd", lw=0.8)
    # legend along the bottom, laid out by measured text widths
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ly, x = 0.08, 0.10

    def adv(t, pad=0.07):
        bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
        return bb.x1 + pad

    for lab, kind in (("frozen weights", "frozen"), ("released weights", "released"), ("trained expert", "expert"),
                      ("new module", "new")):
        S.box(ax, (x, ly - 0.06), 0.20, 0.13, "", kind)
        x = adv(txt(ax, x + 0.25, ly, lab, ha="left"))
    tag(ax, x + 0.08, ly, "F")
    x = adv(txt(ax, x + 0.20, ly, "failure site", ha="left"))
    S.box(ax, (x, ly - 0.06), 0.20, 0.13, "", "ghost")
    x = adv(txt(ax, x + 0.25, ly, "input not read", ha="left"))
    S.arrow(ax, (x, ly), (x + 0.28, ly), color=S.NAVY, lw=1.4, head=6)
    x = adv(txt(ax, x + 0.33, ly, "probe share", ha="left"))
    S.save(fig, "fig_arch", tight=False)


if __name__ == "__main__":
    main()

"""Rung R2w of the matched ladder (docs/SCAFFOLD_RESULTS.md 37): the control for AC-VLA (arXiv 2607.15714), which hides the WHOLE
wrist view while the gripper holds, and with it the held object's identity and all geometry.  R2 (sc_rv.py rview2) instead paints
only the non-active entities and keeps geometry.

    rwrist  R1's prompts (rsup: the role phrase the hold event picks); on every frame where the hold is on (T16.held_rows, exactly
            as sc_rv.frames_of computes it) the wrist image is replaced by a constant mid-grey (FILL = 128 in every channel).
            Before the hold, and the agentview image always, the frames are rsup's.  K7 and K8 (T16.given -> None) stay whole and
            unfilled, as in every rung.

Everything else is sc_rv.py's, which this file imports and extends in its own process only (sc_rv.py itself is not edited):
the same rendered demos (--root), the same lex threshold file, the same LeRobot conversion and norm code.  The training config
pi05_g8v_rwrist_ae is inserted into lifelong_ood/policy/l90c_config.py's AE / NAMES tables at runtime (install_lc), exactly as
the g8v entries are written there; experiments/c_train_pi05_w.py does the same for training and export.

Fill value: 128.  openpi maps uint8 pixels to [-1, 1], so 128 is ~0, the middle of the vision tower's input range; black (0) is
the far end of it (-1) and is also what openpi's pad and its absent-camera placeholder use, though an absent camera is ALSO
masked out of attention (image_mask False), which a fill cannot do.  AC-VLA itself masks the wrist tokens in attention during
training (its sec. 3.3) and says nothing about a pixel value, so no fill reproduces it exactly; grey is the neutral choice.

At test the hold comes from the same ProprioHeld that RoleView steps for rsup / rview2 prompt switching (RoleView.det, stepped on
the served state and the previous chunk's gripper command), so the fill switches on the same query as the prompt does.  The
client's wrist image (already resized by the client) is replaced by the same constant; a constant image is unchanged by the
resize, so this equals filling the raw render first.  No segmentation is needed, so the client runs without --seg, as for R1.

    python experiments/sc_rv_w.py lex     [--suffix S]                     sc_rv's (loads results/flawed/sc_rv_lex<S>.json)
    python experiments/sc_rv_w.py convert [--suffix S] [--root R]          LeRobot lol/libero_g8v_rwrist<S> + sc_rv_rwrist<S>_manifest.json
    python experiments/sc_rv_w.py norm    [--suffix S]                     -> checkpoints_l90c/assets/pi05_g8v_rwrist<S>/lol/libero_g8v_rwrist<S>
    python experiments/sc_rv_w.py serve   --ckpt DIR --port P [--seed 13] [--config NAME] [--serve-fill 1|0]
    python experiments/sc_rv_w.py check   [--root R] [--n 2] [--suffix S]  per-task frame and fill counts; contact sheets in
                                                                            results/flawed/sc_rv_check_w

openpi venv (python 3.11) for all of them; kept parseable under python 3.8 like sc_rv.py.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_rv as RV                                                          # noqa: E402

ARM = "rwrist"
FILL = 128
CONFIG = "pi05_g8v_" + ARM + "_ae"
CHECK_DIR = os.path.join(RV.OUT, "sc_rv_check_w")


def install_lc():
    """pi05_g8v_rwrist_ae into l90c_config's tables, in this process only: AE entry as the g8v entries are written (assets dir
    name, repo), the name appended to NAMES, and register()'s default (bound to the old NAMES at definition) rebound to it."""
    from lifelong_ood.policy import l90c_config as LC
    if CONFIG not in LC.AE:
        LC.AE[CONFIG] = ("pi05_g8v_" + ARM, "lol/libero_g8v_" + ARM)
    if CONFIG not in LC.NAMES:
        LC.NAMES = tuple(LC.NAMES) + (CONFIG,)
    LC.register.__defaults__ = (LC.NAMES,)
    return LC


def frames_of_w(z, c, X, T16, roles):
    """rsup's frames (sc_rv.frames_of with arm rsup: same prompts, same held, images untouched), then the wrist image replaced by
    FILL on held frames of routed tasks.  -> per frame (image, wrist, prompt, held, 0, filled wrist px)."""
    P = T16.given(str(z["language"]))
    for img, wimg, task, h, na, nw in RV._frames_of_base(z, c, "rsup", X, T16, roles):
        assert na == 0 and nw == 0
        if h and P is not None:
            w = np.asarray(wimg)
            wimg = np.full(w.shape, FILL, w.dtype)
            nw = int(w.shape[0] * w.shape[1])
        yield img, wimg, task, h, na, nw


def install():
    """The rwrist arm into sc_rv's tables and frames_of, in this process only."""
    if ARM not in RV.ARMS:
        RV.ARMS = RV.ARMS + (ARM,)
        RV.PAINTED = RV.PAINTED + (ARM,)                                    # convert then writes per-phase fill stats to the manifest
        RV.REPO[ARM] = "lol/libero_g8v_" + ARM
        RV.FULL[ARM] = "pi05_g8v_" + ARM
        RV._frames_of_base = RV.frames_of

        def frames_of(z, c, arm, X, T16, roles):
            if arm == ARM:
                return frames_of_w(z, c, X, T16, roles)
            return RV._frames_of_base(z, c, arm, X, T16, roles)
        RV.frames_of = frames_of
    install_lc()


class WristGate:
    """The policy RoleView calls: fills the wrist image while RoleView's own ProprioHeld is on, for a routed task."""

    def __init__(self, policy, fill=True):
        self.p, self.fill, self.view = policy, bool(fill), None
        self.metadata = getattr(policy, "metadata", {})

    def infer(self, o):
        v = self.view
        on = bool(v.det.on)                                                 # what det.step returned this query (RoleView's `on`)
        n = 0
        if on and v.w_routed and self.fill:
            w = np.asarray(o["observation/wrist_image"])
            o = dict(o)
            o["observation/wrist_image"] = np.full(w.shape, FILL, w.dtype)
            n = int(w.shape[0] * w.shape[1])
        logging.info(f"[rvw] t={o.get('meta/t')} held={on} routed={v.w_routed} prompt={o['prompt']!r} wrist filled={n}")
        return self.p.infer(o)


class WristView(RV.RoleView):
    """sc_rv.RoleView in rsup mode (prompts only) around a WristGate."""

    def __init__(self, policy, X, fill=True, replan=5):
        gate = WristGate(policy, fill)
        super().__init__(gate, X, mode=None, replan=replan)
        gate.view = self
        self.w_routed = False

    def infer(self, obs):
        self.w_routed = self.T16.given(str(obs["prompt"])) is not None
        return super().infer(obs)


def serve_cmd(a):
    RV._openpi()
    install()
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert a.arm == ARM, a.arm
    name = a.config or CONFIG
    X = json.load(open(RV.lex_path(a.suffix), encoding="utf-8"))
    logging.info(f"[rvw] serve arm {a.arm} config {name} ckpt {a.ckpt} tw {X['tw']:.6f} lift {X['lift']} fill {FILL} "
                 f"serve_fill {bool(a.serve_fill)}")
    bk = PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)
    warm(pol)
    serve(WristView(pol, X, fill=bool(a.serve_fill)), a.port)


def check(a):
    """Per task over the first n demos: frames and filled wrist frames per phase, with every agentview frame, every prompt and
    every unfilled wrist frame asserted equal to rsup's; totals; one contact sheet per task (demo 0: 3 frames before the hold, 3
    during it; columns agent raw | agent served | wrist raw | wrist served)."""
    RV._openpi()
    install()
    from PIL import Image, ImageDraw
    T16 = RV.t16()
    X = json.load(open(RV.lex_path(a.suffix), encoding="utf-8"))
    by_c, _ = RV.roles_of()
    os.makedirs(CHECK_DIR, exist_ok=True)
    S = RV.SIZE
    tot = {}
    for c in RV.KNOWN:
        fs = RV.demo_files(a.root, c)[: a.n]
        if not fs:
            continue
        st = {}
        for f in fs:
            z = dict(np.load(f))
            ref = list(RV._frames_of_base(z, c, "rsup", X, T16, by_c))
            for i, (img, wimg, task, h, na, nw) in enumerate(RV.frames_of(z, c, ARM, X, T16, by_c)):
                r = ref[i]
                assert task == r[2] and h == r[3] and np.array_equal(img, r[0]), (c, f, i)
                assert (nw > 0) or np.array_equal(wimg, r[1]), (c, f, i)
                assert (nw == 0) or (int(np.asarray(wimg).min()) == FILL == int(np.asarray(wimg).max())), (c, f, i)
                s = st.setdefault("held" if h else "pre", [0, 0])
                s[0] += 1
                s[1] += int(nw > 0)
        for ph, s in sorted(st.items()):
            t = tot.setdefault(ph, [0, 0])
            t[0] += s[0]
            t[1] += s[1]
            print("[rvw] check %s %-4s demos %d frames %6d  wrist filled %6d" % (c, ph, len(fs), s[0], s[1]), flush=True)
        z = dict(np.load(fs[0]))
        rows = list(RV.frames_of(z, c, ARM, X, T16, by_c))
        pre = [i for i, r in enumerate(rows) if not r[3]]
        hel = [i for i, r in enumerate(rows) if r[3]]
        pick = (list(np.array(pre)[np.linspace(0, len(pre) - 1, 3 if hel else 6).astype(int)]) if pre else []) + \
               (list(np.array(hel)[np.linspace(0, len(hel) - 1, 3).astype(int)]) if hel else [])
        sheet = Image.new("RGB", (4 * S, len(pick) * S), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        for r, i in enumerate(pick):
            img, wimg, task, h, na, nw = rows[i]
            for col, x in enumerate((z["image"][i], img, z["wrist"][i], wimg)):
                sheet.paste(Image.fromarray(np.asarray(x, np.uint8)), (col * S, r * S))
            dr.rectangle([0, r * S, 4 * S, r * S + 14], fill=(0, 0, 0))
            dr.text((3, r * S + 1), "%s demo %d t=%d held=%d prompt=%r wrist filled=%d" % (c, int(z["demo"]), i, int(h), task, nw),
                    fill=(255, 255, 255))
        out = os.path.join(CHECK_DIR, "%s_%s%s.png" % (c, ARM, a.suffix))
        sheet.save(out)
        print("[rvw] check %s contact sheet -> %s (frames %s)" % (c, out, [int(i) for i in pick]), flush=True)
    for ph, t in sorted(tot.items()):
        print("[rvw] check TOTAL %-4s frames %6d  wrist filled %6d" % (ph, t[0], t[1]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("lex", "convert", "norm", "serve", "check"))
    ap.add_argument("--root", default=RV.RV_ROOT)
    ap.add_argument("--arm", default=ARM, choices=(ARM,))
    ap.add_argument("--suffix", default="", help="appended to dataset, norm and lex names (smoke: _smoke)")
    ap.add_argument("--n", type=int, default=2, help="check: demos per task")
    ap.add_argument("--port", type=int)
    ap.add_argument("--ckpt")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--config", default="", help="serve: config name override (default pi05_g8v_rwrist_ae)")
    ap.add_argument("--serve-fill", type=int, default=1, choices=(0, 1),
                    help="serve: 1 (default) fills the wrist view at test as in training; 0 = train-only masking (AC-VLA's inference)")
    a = ap.parse_args()
    if a.what == "serve":
        serve_cmd(a)
    elif a.what == "check":
        check(a)
    else:
        install()
        dict(lex=RV.lex, convert=RV.convert, norm=RV.norm)[a.what](a)

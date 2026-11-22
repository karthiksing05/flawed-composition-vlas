"""Cue schematics.  Interface: draw_cue(key, ax, compact=False); keys F1 F1s2 F1m F2 F7 F6 NL MV.
Axis contract: limits (0,1) x (0,1), equal aspect.  STEP 0 STUB: placeholder boxes only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S  # noqa: E402

KEYS = ["F1", "F1s2", "F1m", "F2", "F7", "F6", "NL", "MV"]


def draw_cue(key, ax, compact=False):
    assert key in KEYS, key
    S.blank_axes(ax)
    S.box(ax, (0.02, 0.02), 0.96, 0.96, "cue " + key, "ghost")
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(S.W2, 3.4))
    for k, ax in zip(KEYS, axes.flat):
        draw_cue(k, ax)
    S.save(fig, "fig_cues_all")

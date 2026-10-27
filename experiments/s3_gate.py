"""PLAN_SOLVE S3a confidence gate (docs/PLAN_S3.md): binder confidence features -> one unit score -> gate (both role masks or none).

Per role, the binder (s2_serve.Binder.masks) reports conf = [margin, p_top, det, mass, sig_max, K]:
  margin   head softmax over the detector candidates: top-1 minus top-2 probability (1.0 when there is one candidate)
  p_top    the chosen candidate's softmax probability
  det      the chosen box's GroundingDINO score
  mass     tokens in the role mask
  sig_max  max crop-cell sigmoid of the chosen candidate
  K        number of detector candidates (0: no box -> zero mask)
SCORES (the pre-declared family; S3a picks one + its threshold on TRAINING cells only):
  margin  = min over roles of margin
  det     = min over roles of det
  joint   = min over roles of (margin * det), with a role of mass <= 1 (a degenerate one-token mask) scoring 0
Gate: score >= tau -> both binder masks as served; else both masks zero (the adapter's edit is then exactly zero = zs).
"""
import numpy as np

F = dict(margin=0, p_top=1, det=2, mass=3, sig_max=4, K=5)
SCORES = ("margin", "det", "joint")


def score(co, cg, name):
    co, cg = np.asarray(co, float), np.asarray(cg, float)
    if name == "margin":
        return float(min(co[0], cg[0]))
    if name == "det":
        return float(min(co[2], cg[2]))
    if name == "joint":
        return float(min(c[0] * c[2] * (c[3] > 1) for c in (co, cg)))
    raise ValueError(name)

"""SR 86 Q2 part d (exploratory): chunk-level overshoot of the module-owned servo.  For consecutive held queries (5 steps apart) in the channel: displacement per chunk at |rel_xy| 1-4 cm, and the fraction
that pass through the target (dot(rel_t, rel_t+5) < 0 with |rel_t+5| >= 1 cm); W3 queries with own = true vs RAA-sL held queries.   python experiments/sr86_q2d.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr86_q2 as Q
for arm in ("wasw3", "wasl"):
    LG = Q.load_logs(arm)[0]
    disp_sat, disp_near, over, n_near, disp_near_lo = [], [], 0, 0, []
    for (tid, tr), L in LG.items():
        if tid not in (0, 1, 4, 5, 9): continue
        for (c0, w0), (c1, w1) in zip(L[:-1], L[1:]):
            if c0["role"] != "target" or c1["role"] != "target" or c1["t"] - c0["t"] != 5: continue
            if arm == "wasw3" and not w0["own"]: continue
            r0, r1 = np.array(c0["rel_cm"][:2], float), np.array(c1["rel_cm"][:2], float)
            d0 = np.linalg.norm(r0)
            if d0 > 4: disp_sat.append(np.linalg.norm(r1 - r0)) if d0 > 6 else None
            if 1.0 <= d0 <= 4.0:
                n_near += 1; disp_near.append(np.linalg.norm(r1 - r0))
                if r0 @ r1 < 0 and np.linalg.norm(r1) >= 1.0: over += 1
    print(arm, "chunks with 1<=|rel_xy|<=4 cm: n=%d displacement per 5 steps median %.2f p90 %.2f cm; pass-through (overshoot >= 1 cm) %.2f | chunks |rel|>6: displacement median %.2f cm (n=%d)" % (
        n_near, np.median(disp_near), np.percentile(disp_near, 90), over / max(n_near, 1), np.median(disp_sat) if disp_sat else -1, len(disp_sat)))

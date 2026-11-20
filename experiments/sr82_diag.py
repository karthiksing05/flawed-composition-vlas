"""SR 82 diagnostic (not registered): why RAA-sL full collapses on F6_b (rgc tid 8). Read-only on results/flawed/wasl (chan logs, *_wa.json, *_track.npz).
   python experiments/sr82_diag.py   (run via sbatch cpu; prints tables only)"""
import json, os, sys, collections
import numpy as np
D = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/wasl"
TID = 8
def chan(arm):
    out = collections.defaultdict(list)
    for f in sorted(os.listdir(D)):
        if f.startswith("chan_") and f.endswith(".jsonl"):
            for l in open(os.path.join(D, f), encoding="utf-8"):
                r = json.loads(l)
                if r["tid"] == TID and r["arm"] == arm:
                    out[r["ep"]].append(r)
    return out
for arm in ("full", "noM"):
    J = json.load(open(f"{D}/{arm}_{TID}.json"))["tasks"][str(TID)]["episodes"]
    succ = {e["trial"]: e["success"] for e in J}
    steps = {e["trial"]: e["steps"] for e in J}
    C = chan(arm)
    W = json.load(open(f"{D}/{arm}_{TID}_wa.json"))["eps"]
    Z = np.load(f"{D}/{arm}_{TID}_track.npz")
    names = list(Z["names"])
    print(f"=== arm {arm} tid {TID}: eps {len(J)} success {sum(succ.values())}  names {names}")
    print("trial succ steps | nq_obj nq_tgt tgt_omitted | tgt src counts | tgt err_cm med/max | first-held-tgt rel_cm | last tgt rel_cm / true_cm | final ee-plate xy cm, bowl-plate xy cm")
    for i, w in enumerate(W):
        tr = w["trial"]
        recs = C.get(tr, [])
        tg = [r for r in recs if r["role"] == "target"]
        ob = [r for r in recs if r["role"] == "object"]
        src = collections.Counter(r["src"] for r in tg)
        er = [r["err_cm"] for r in tg]
        om = sum(r["omitted"] for r in tg)
        pos, ee = Z[f"e{i}_pos"], Z[f"e{i}_ee"]
        pl = names.index("plate_1") if "plate_1" in names else None
        b = names.index("akita_black_bowl_1")
        fin = ""
        if pl is not None:
            fin = "ee-plate %.1f bowl-plate %.1f (plate moved %.1f)" % (100*np.linalg.norm(ee[-1][:2]-pos[-1][pl][:2]), 100*np.linalg.norm(pos[-1][b][:2]-pos[-1][pl][:2]), 100*np.linalg.norm(pos[-1][pl]-pos[0][pl]))
        f0 = tg[0]["rel_cm"] if tg else None
        l0 = (tg[-1]["rel_cm"], tg[-1]["true_cm"]) if tg else None
        print(tr, succ.get(tr), steps.get(tr), "|", len(ob), len(tg), om, "|", dict(src), "|", ("%.1f/%.1f" % (np.median(er), max(er))) if er else "-", "|", f0, "|", l0, "|", fin)
    # aggregate
    allt = [r for e in C.values() for r in e if r["role"] == "target"]
    print("AGG target queries", len(allt), "omitted", sum(r["omitted"] for r in allt), "src", dict(collections.Counter(r["src"] for r in allt)),
          "err med %.2f p90 %.2f" % (np.median([r["err_cm"] for r in allt]), np.percentile([r["err_cm"] for r in allt], 90)) if allt else "")
    for s in sorted(set(r["src"] for r in allt)):
        e = [r["err_cm"] for r in allt if r["src"] == s]
        n = [np.linalg.norm(r["rel_cm"]) for r in allt if r["src"] == s]
        print("  src", s, "n", len(e), "err med %.2f" % np.median(e), "|rel| med %.1f cm" % np.median(n))

# ---- timeline of selected episodes (full vs noM), every 5 steps, plus a summary of grasp / release events over all episodes
def timeline(arm, i, stride=5, lo=0, hi=300):
    Z = np.load(f"{D}/{arm}_{TID}_track.npz")
    names = list(Z["names"]); b, p = names.index("akita_black_bowl_1"), names.index("plate_1")
    pos, ee, a6, hid = Z[f"e{i}_pos"], Z[f"e{i}_ee"], Z[f"e{i}_a6"], Z[f"e{i}_hidden"]
    print(f"--- timeline {arm} episode index {i}: t | ee xyz cm | grip cmd | bowl-ee cm | bowl-plate xy cm | bowl z-plate z cm | covered")
    for t in range(lo, min(hi, len(ee)), stride):
        print(t, np.round(100*ee[t], 1).tolist(), round(float(a6[t]), 1), np.round(100*(pos[t][b]-ee[t]), 1).tolist(),
              round(100*float(np.linalg.norm(pos[t][b][:2]-pos[t][p][:2])), 1), round(100*float(pos[t][b][2]-pos[t][p][2]), 1), bool(hid[t]))
def events(arm):
    Z = np.load(f"{D}/{arm}_{TID}_track.npz"); names = list(Z["names"]); b, p = names.index("akita_black_bowl_1"), names.index("plate_1")
    n = int(Z["n"]); rows = []
    for i in range(n):
        a6, pos, ee = Z[f"e{i}_a6"], Z[f"e{i}_pos"], Z[f"e{i}_ee"]
        close = a6 > 0
        ch = np.nonzero(np.diff(close.astype(int)))[0] + 1
        lift = np.nonzero(pos[:, b, 2] > pos[0, b, 2] + 0.03)[0]
        first_lift = int(lift[0]) if len(lift) else -1
        # time and bowl-plate xy / height at the first opening after the first lift
        opens = [c for c in ch if not close[c] and c > first_lift] if first_lift >= 0 else []
        o = opens[0] if opens else -1
        rows.append((first_lift, o, None if o < 0 else (100*np.linalg.norm(pos[o][b][:2]-pos[o][p][:2]), 100*(pos[o][b][2]-pos[o][p][2])), len(ch)))
    print(f"=== events {arm}: (first lift step, first release step after lift, (bowl-plate xy cm, bowl-plate z cm) at release, #gripper toggles)")
    fl = [r[0] for r in rows if r[0] >= 0]; rl = [r[1] for r in rows if r[1] >= 0]
    print(" lift step median", np.median(fl) if fl else None, "n", len(fl), "| release step median", np.median(rl) if rl else None, "n", len(rl))
    xy = [r[2][0] for r in rows if r[2]]; zz = [r[2][1] for r in rows if r[2]]
    print(" at first release: bowl-plate xy med %.1f cm, z med %.1f cm; toggles med %s" % (np.median(xy), np.median(zz), np.median([r[3] for r in rows])))
    return rows
if __name__ == "__main__":
    for arm in ("full", "noM"):
        events(arm)
    timeline("full", 0, 5, 0, 120); timeline("full", 3, 5, 0, 120); timeline("noM", 0, 5, 0, 130)

def release_table():
    for arm in ("full", "noM"):
        J = json.load(open(f"{D}/{arm}_{TID}.json"))["tasks"][str(TID)]["episodes"]
        W = json.load(open(f"{D}/{arm}_{TID}_wa.json"))["eps"]
        C = chan(arm)
        Z = np.load(f"{D}/{arm}_{TID}_track.npz"); names = list(Z["names"]); b, p = names.index("akita_black_bowl_1"), names.index("plate_1")
        rows = []
        for i, w in enumerate(W):
            s = bool([e for e in J if e["trial"] == w["trial"]][0]["success"])
            a6, pos, ee = Z[f"e{i}_a6"], Z[f"e{i}_pos"], Z[f"e{i}_ee"]
            close = a6 > 0
            lift = np.nonzero(pos[:, b, 2] > pos[0, b, 2] + 0.03)[0]
            if not len(lift):
                continue
            ch = [c for c in np.nonzero(np.diff(close.astype(int)))[0] + 1 if not close[c] and c > lift[0]]
            if not ch:
                continue
            o = int(ch[0])
            xy = 100 * np.linalg.norm(pos[o][b][:2] - pos[o][p][:2])
            tq = [r for r in C.get(w["trial"], []) if r["role"] == "target" and r["t"] <= o]
            rx = 100 * np.linalg.norm(np.array(tq[-1]["rel_cm"][:2])) if tq else float("nan")
            rows.append((s, o, xy, rx, 100 * (pos[o][b][2] - pos[o][p][2]), len(tq)))
        R = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], float)
        print(f"=== {arm}: release-table over {len(R)} episodes with a lift+release. xy = bowl-plate xy at first release (cm); rel_xy = |rel xy| of the last target query before release (cm)")
        for lo, hi in ((0, 3), (3, 5), (5, 7), (7, 99)):
            m = (R[:, 2] >= lo) & (R[:, 2] < hi)
            print(f"  xy in [{lo},{hi}): n {int(m.sum())}  success {R[m,0].mean() if m.any() else float('nan'):.2f}")
        print("  median: release step %.0f, xy %.1f, rel_xy %.1f, n target queries before release %.1f ; success eps xy med %s, fail eps xy med %s" % (
            np.median(R[:, 1]), np.median(R[:, 2]), np.nanmedian(R[:, 3]), np.median(R[:, 5]),
            None if not (R[:, 0] == 1).any() else round(float(np.median(R[R[:, 0] == 1, 2])), 1), round(float(np.median(R[R[:, 0] == 0, 2])), 1)))
        # speed: ee xy displacement per 5 steps during the first 20 steps after the first target query (approach rate)
if __name__ == "__main__":
    release_table()

def f6a():
    """Same tables for F6_a (tid 9, full only): plate visible, same servo and channel."""
    global TID
    TID = 9
    J = json.load(open(f"{D}/full_9.json"))["tasks"]["9"]["episodes"]
    W = json.load(open(f"{D}/full_9_wa.json"))["eps"]
    C = chan("full")
    Z = np.load(f"{D}/full_9_track.npz"); names = list(Z["names"]); b, p = names.index("akita_black_bowl_1"), names.index("plate_1")
    xs, ss, lt, rl, tgsrc, tge = [], [], [], [], collections.Counter(), []
    for i, w in enumerate(W):
        s = bool([e for e in J if e["trial"] == w["trial"]][0]["success"])
        a6, pos = Z[f"e{i}_a6"], Z[f"e{i}_pos"]
        close = a6 > 0
        lift = np.nonzero(pos[:, b, 2] > pos[0, b, 2] + 0.03)[0]
        for r in C.get(w["trial"], []):
            if r["role"] == "target":
                tgsrc[r["src"]] += 1; tge.append(r["err_cm"])
        if not len(lift):
            continue
        ch = [c for c in np.nonzero(np.diff(close.astype(int)))[0] + 1 if not close[c] and c > lift[0]]
        if not ch:
            continue
        o = int(ch[0]); xs.append(100 * np.linalg.norm(pos[o][b][:2] - pos[o][p][:2])); ss.append(s); lt.append(int(lift[0])); rl.append(o)
    xs, ss = np.array(xs), np.array(ss)
    print(f"=== F6_a full (tid 9): n {len(xs)}; lift step med {np.median(lt)}, release step med {np.median(rl)}, xy at first release med {np.median(xs):.1f} cm; success by xy: "
          + ", ".join(f"[{lo},{hi}) n {int(((xs>=lo)&(xs<hi)).sum())} succ {ss[(xs>=lo)&(xs<hi)].mean() if ((xs>=lo)&(xs<hi)).any() else float('nan'):.2f}" for lo, hi in ((0, 3), (3, 5), (5, 7), (7, 99))))
    print("    target src counts", dict(tgsrc), "target chan err med %.2f cm" % np.median(tge))
if __name__ == "__main__":
    f6a()

def lastrel():
    """|rel xy| and |rel z| (cm) of the LAST target-role query of each episode, by cell and outcome; and the number of target queries."""
    global TID
    for arm, tid in (("full", 8), ("full", 9), ("noM", 8)):
        TID = tid
        J = json.load(open(f"{D}/{arm}_{tid}.json"))["tasks"][str(tid)]["episodes"]
        succ = {e["trial"]: bool(e["success"]) for e in J}
        C = chan(arm)
        rows = []
        for tr, recs in C.items():
            tq = [r for r in recs if r["role"] == "target"]
            if tq:
                rows.append((succ.get(tr), np.linalg.norm(tq[-1]["rel_cm"][:2]), tq[-1]["rel_cm"][2], len(tq), tq[0]["t"], tq[-1]["t"]))
        for lab in (True, False):
            R = np.array([r[1:] for r in rows if r[0] == lab], float)
            if len(R):
                print(f"[lastrel] {arm} tid {tid} success={lab}: n {len(R)} last-target rel_xy med {np.median(R[:,0]):.1f} cm (frac <3cm {np.mean(R[:,0]<3):.2f}), rel_z med {np.median(R[:,1]):.1f}, n target queries med {np.median(R[:,2]):.0f}, first/last target t med {np.median(R[:,3]):.0f}/{np.median(R[:,4]):.0f}")
if __name__ == "__main__":
    lastrel()

def replay(path="/coc/flash10/ksingara3/.claude_tmp/pp/sr82_replay/servo_expert.jsonl"):
    """The diagnostic server's per-query log (experiments/sr82_diag_serve.py): during TARGET-role queries, servo vs expert (= action - servo) in raw action units (xyz, mean of first 5 rows)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    eps, cur, prev = [], [], -1
    for r in rows:
        if r["t"] <= prev and cur:
            eps.append(cur); cur = []
        cur.append(r); prev = r["t"]
    eps.append(cur)
    print(f"[replay] {len(rows)} queries, {len(eps)} episodes (first 4 = full, next 4 = noM)")
    for k, e in enumerate(eps):
        tg = [r for r in e if r["role"] > 1.5]
        if not tg:
            print(f"[replay] ep {k}: no target queries"); continue
        S_ = np.array([r["servo"] for r in tg]); X = np.array([r["expert"] for r in tg]); A = np.array([r["act"] for r in tg]); Rl = np.array([r["state"][:3] for r in tg]) * 100
        print(f"[replay] ep {k} ({'full' if k < 4 else 'noM'}): n target q {len(tg)}; per query (rel_cm xyz | servo xyz | expert xyz | action xyz):")
        for r, rl in zip(tg, Rl):
            print("   t %3d rel %s | servo %s | expert %s | act %s" % (r["t"], np.round(rl, 0).astype(int).tolist(), r["servo"], r["expert"], r["act"]))
        if k < 4:
            n = len(tg)
            h = slice(n // 2, n)
            print("   second half of target phase: mean servo_xy %s, mean expert_xy %s, mean action_xy %s; cos(servo_xy, expert_xy) %.2f" % (
                np.round(S_[h, :2].mean(0), 3).tolist(), np.round(X[h, :2].mean(0), 3).tolist(), np.round(A[h, :2].mean(0), 3).tolist(),
                float(np.dot(S_[h, :2].mean(0), X[h, :2].mean(0)) / (np.linalg.norm(S_[h, :2].mean(0)) * np.linalg.norm(X[h, :2].mean(0)) + 1e-9))))
if __name__ == "__main__":
    replay()

if __name__ == "__main__" and os.environ.get("SR82_ONLY_REPLAY_A"):
    print("[replay] ===== F6_a (tid 9) replay =====")
    replay("/coc/flash10/ksingara3/.claude_tmp/pp/sr82_replay_a/servo_expert.jsonl")

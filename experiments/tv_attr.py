"""TV-A: attribute the confirmed TV+P gain on frozen pi0.5 (SR 47) between R1's role phrases and the silhouette role views, and
separate the silhouettes from an AC-VLA-style whole-wrist mask at a matched prompt (m3_prereg tv_attr; docs/SCAFFOLD_RESULTS.md 48).

Two new arms, on the fresh familiar-layout states of SR 47 (k = 36-55 = harness trials 24-43), pi0.5 only:
    p    phrases only: the prompt is sc_t16.given(instruction)[held] ("object: the cream cheese" / "target: the plate"), switched by
         the same ProprioHeld detector tvp uses (sc_rv's lex threshold, stepped on every env step from WAIT).  No paint, no mask:
         both camera images are passed through untouched.
    wp   tv_frozen's w (AC-VLA-style: the whole wrist image is grey 128 on every observation where the detector is on; agentview
         untouched; no silhouettes) plus the same phrases, switched by the same detector.
I, tv and tvp are reused from SR 47: I = results/flawed/pp_atlas_fresh/pi05_<tid>, tv / tvp = results/flawed/tv_fresh/pi05_<arm>_<tid>.

Code reuse.  tv_frozen.py and tv_fresh.py are imported and never edited.  The front end is tv_frozen.Front: wp IS Front with arm "w"
(same detector, same grey fill, same diagnostics); p is Front with its detector stepped exactly as Front.step steps it and the
observation returned unchanged.  The prompt wrapper is tv_frozen's tvp wrapper (same assert, same switch on Front.on).  The only
difference from tv_frozen.client_main is the arm table: p / wp load no segmenter, and both use the phrases.

Modes (LIBERO venv):
    python experiments/tv_attr.py client --arm p|wp <pp_atlas_client.py's arguments>     one n2 task per process (GPU node)
        -> <out>.json (libero_client), <out>_track.npz (pp_atlas.Rec), <out>_tv.json (per episode: hold, prompts)
    python experiments/tv_attr.py summary DIR                                            per-episode table (smoke)
    python experiments/tv_attr.py read [--out results/flawed/tv_attr_read.json]           (CPU)

Registered mapping (m3_prereg tag tv_attr), pi0.5, familiar layout, trials 24-43, success matrices [task, state], paired bootstrap
B = 2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2), CI 2.5 / 97.5, eps = .10, on N1-N6:
    VIEWS       d = tvp - p:   VIEWS ADD iff d > .10 and lo > 0;  VIEWS NOT NEEDED iff hi < .10;  else UNRESOLVED.
    SEPARATION  d = tvp - wp:  SEPARATED iff d > .10 and lo > 0;  W+P EQUIVALENT iff hi < .10;  else UNRESOLVED.
Descriptive: p - I, wp - I, wp - p, p - tv, tvp - tv on N1-N6; K1-K4, K6 success per arm and arm - I there; the atlas's within /
cross-chunk excess per arm (pp_atlas estimator, via tv_frozen.atlas_est); stage shares; detector diagnostics.
Checks (reported): every arm's client args equal I's except out / port / videos, and per-episode trial / branch / variant / at equal
I's (tv_fresh.harness); I's excess reproduces pp_atlas_fresh_read.json; tv / tvp reproduce tv_fresh_read.json's success and cross E;
the state / branch files match tv_fresh's prereg row; the reused tv_fresh rollouts match the hashes in tv_attr's prereg row.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import tv_frozen as TF                                                      # noqa: E402

ARMS = ("p", "wp")
EPS = 0.10
MODEL = "pi05"
READ_ARMS = ("I", "tv", "tvp", "p", "wp")


class AttrFront(TF.Front):
    """tv_frozen.Front with arm 'w' underneath.  wp = Front('w') unchanged.  p = the same detector, no change to the observation."""

    def __init__(self, env, arm, obj, tgt, X, seg, sheet=False):
        super().__init__(env, "w", obj, tgt, X, seg, sheet=sheet and arm == "wp")
        self.attr = arm

    def step(self, obs, n, cmd):
        if self.attr == "wp":
            return super().step(obs, n, cmd)
        if n < TF.WAIT:                                                     # p: Front.step's sequencer lines, nothing painted
            return obs
        t0 = time.time()
        was = self.on
        self.on = bool(self.det.step(np.asarray(obs["robot0_gripper_qpos"], np.float64), cmd, float(obs["robot0_eef_pos"][2])))
        row = n - TF.WAIT
        if self.on and not was:
            self.onsets.append(row)
        elif was and not self.on:
            self.offsets.append(row)
        if row % TF.REPLAN == 0:
            self.queries += 1
            self.held_queries += int(self.on)
        self.secs += time.time() - t0
        return obs

    def record(self, held):
        r = super().record(held)
        r["arm"] = self.attr
        return r


# ---------------------------------------------------------------- client (LIBERO venv)
def client_main():
    """tv_frozen.client_main with the arm table of p / wp (no segmenter; phrases in both)."""
    def take(name, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return v
        return default
    sys.argv.pop(1)                                                         # "client"
    arm = take("--arm")
    assert arm in ARMS, arm
    import fc_n2_client as FN
    import pp_atlas as PA
    import sc_rv as RV
    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID and TASK["layout"] == "familiar", TASK
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    seg = None
    P = RV.t16().given(TASK["language"])
    assert P is not None, TASK
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], {}
    SIDEF = OUTF.replace(".json", "_tv.json")
    sheet = os.environ.get("TV_SHEET") == "1"

    class TVEnv(PA.Rec):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.front = None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.front = AttrFront(self._e, arm, TASK["object"], TASK["target"], X, seg, sheet=sheet and len(SIDE) < 2)
            self.front.prompts = dict(object=0, target=0)
            return obs

        def step(self, a):
            r = super().step(a)
            if self.front is None:
                return r
            obs = self.front.step(r[0], self.n, float(a[6]))
            return (obs,) + tuple(r[1:])

        def close(self):
            if self.n > 0 and self.front is not None:
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = self.front.record(self.rows["held"])
                rec.update(trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"],
                           prompts=self.front.prompts)
                SIDE.append(rec)
                if self.front.frames:
                    import imageio
                    imageio.imwrite(OUTF.replace(".json", "_sheet_%s.png" % rec["trial"]), np.concatenate([f[1] for f in self.front.frames], 0))
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X,
                                   seg_ckpt=None, eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[tvA] %s %s trial %s: onsets %s (oracle first held %s) held queries %d/%d painted %s prompts %s" % (
                    arm, TASK["key"], rec["trial"], rec["onsets"][:4], rec["oracle_first_held"], rec["held_queries"], rec["queries"],
                    rec["painted_queries"], rec["prompts"]), flush=True)
                self.front = None
            self._e.close()

    Base = LC._wcp.WebsocketClientPolicy

    class PClient(Base):                                                    # tv_frozen's tvp wrapper, unchanged
        def infer(self, el):
            env = CUR.get("env")
            f = None if env is None else env.front
            assert str(el["prompt"]) == TASK["language"], el["prompt"]
            el = dict(el)
            on = bool(f is not None and f.on)
            el["prompt"] = P[int(on)]
            if f is not None:
                f.prompts["target" if on else "object"] += 1
            return super().infer(el)
    LC._wcp.WebsocketClientPolicy = PClient

    LC.OffScreenRenderEnv = lambda *a, **k: TVEnv(ORIG(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    print("[tvA] arm %s task %s %r object %s target %s lex tw %.6f lift %s phrases %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], X["tw"], X["lift"], P), flush=True)
    LC.main()


# ---------------------------------------------------------------- read (CPU)
def label(d, win, null):
    if d[0] > EPS and d[1] > 0:
        return win
    if d[2] < EPS:
        return null
    return "UNRESOLVED"


MAPPING = ("pi0.5, familiar layout, trials 24-43 (fresh k 36-55); success matrices [task, state]; paired bootstrap B=2000 rng 0, tasks "
           "and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5; on N1-N6. VIEWS d = tvp - p: VIEWS ADD iff d > .10 and lo > 0; "
           "VIEWS NOT NEEDED iff hi < .10; else UNRESOLVED. SEPARATION d = tvp - wp: SEPARATED iff d > .10 and lo > 0; W+P EQUIVALENT "
           "iff hi < .10; else UNRESOLVED. Descriptive: p - I, wp - I, wp - p, p - tv, tvp - tv; K cost per arm; atlas excess per arm.")


def read(a):
    import pp_atlas as PA
    import tv_fresh as TFR
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    br = json.load(open(TFR.BR, encoding="utf-8"))
    assert [int(b["trial"]) for b in br] == TFR.FRESH_TRIALS
    rows = {t: [r for r in ledger.load("m3_prereg") if r.get("tag") == t] for t in ("tv_fresh", "tv_attr")}
    assert len(rows["tv_fresh"]) == 1 and len(rows["tv_attr"]) == 1, {k: len(v) for k, v in rows.items()}
    reg_f = rows["tv_fresh"][0]["metrics"]["files_sha256"]
    reg_a = rows["tv_attr"][0]["metrics"]["files_sha256"]
    files_ok = {TFR.rel(p): reg_f.get(TFR.rel(p)) == TFR.sha(p) for p in TFR.state_files()}
    reused_ok = {k: TFR.sha(os.path.join(ROOT, k)) == v for k, v in reg_a.items() if k.startswith("results/flawed/tv_fresh/")}
    n2 = PA.tasks()
    ks = TFR.FRESH_TRIALS
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN
    D_ATTR = os.path.abspath(a.dir)
    where = dict(I=(TFR.I_DIR, MODEL + "_{tid}"), tv=(TFR.DIR, MODEL + "_tv_{tid}"), tvp=(TFR.DIR, MODEL + "_tvp_{tid}"),
                 p=(D_ATTR, MODEL + "_p_{tid}"), wp=(D_ATTR, MODEL + "_wp_{tid}"))
    A, H = {}, {}
    for arm in READ_ARMS:
        d, st = where[arm]
        files = [os.path.join(d, st.format(tid=T[c]["tid"])) for c in comps]
        miss = [f for f in files if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
        if miss:
            A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
            continue
        if arm != "I":
            H[arm] = {}
            for c, f in zip(comps, files):
                diff, ep_ok = TFR.harness(MODEL, os.path.join(TFR.I_DIR, "%s_%d" % (MODEL, T[c]["tid"])), f)
                H[arm][c] = dict(args_diff=diff, episodes_match=ep_ok)
        RATE, succ, per = {}, {}, {}
        for c, f in zip(comps, files):
            S, X, G, SU = PA.stage_table(f, T[c], ks)
            RATE[c] = {s: np.array([float(x == s) for x in S]) for s in PA.STAGES}
            succ[c] = np.array(SU, float)
            per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
        atl = TF.atlas_est(RATE, NEW, PAIR, ks, PA)
        new_m = np.stack([succ[c] for c in NEW])
        kn_m = np.stack([succ[c] for c in KNOWN])
        side = None
        if arm != "I":
            side = TF.side_summary([os.path.join(d, st.format(tid=T[c]["tid"])) + "_tv.json" for c in comps])
            if arm in ARMS:
                pr = [e["prompts"] for f in files for e in json.load(open(f + "_tv.json", encoding="utf-8"))["eps"]]
                side["target_prompt_share"] = _ratio(sum(x["target"] for x in pr), sum(x["target"] + x["object"] for x in pr))
        A[arm] = dict(new=new_m, known=kn_m, row=dict(
            success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), per_task_success={c: float(succ[c].mean()) for c in comps},
            stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
            grasp_ok_new=1.0 - float(np.mean([per[c]["GRASP"] for c in NEW])),
            cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per, atlas=atl,
            side=side))
    row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

    def dl(x, y, key="new"):
        if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
            return TF.boot2(A[x][key], A[y][key])
        return None
    D = dict(tvp_minus_p=dl("tvp", "p"), tvp_minus_wp=dl("tvp", "wp"), p_minus_I=dl("p", "I"), wp_minus_I=dl("wp", "I"),
             wp_minus_p=dl("wp", "p"), p_minus_tv=dl("p", "tv"), tvp_minus_tv=dl("tvp", "tv"), tvp_minus_I=dl("tvp", "I"),
             tv_minus_I=dl("tv", "I"))
    D["known"] = {"%s_minus_I" % x: dl(x, "I", "known") for x in READ_ARMS[1:]}
    V = dict(views=None if D["tvp_minus_p"] is None else label(D["tvp_minus_p"], "VIEWS ADD", "VIEWS NOT NEEDED"),
             separation=None if D["tvp_minus_wp"] is None else label(D["tvp_minus_wp"], "SEPARATED", "W+P EQUIVALENT"))
    ref = json.load(open(TFR.ATLAS_READ, encoding="utf-8"))
    chk = None
    if isinstance(A.get("I"), dict):
        g0, g1 = ref[MODEL]["familiar"]["groups"], A["I"]["row"]["atlas"]["groups"]
        chk = all(abs(g0[g]["E"] - g1[g]["E"]) < 1e-9 and all(abs(u - v) < 1e-9 for u, v in zip(g0[g]["ci"], g1[g]["ci"])) for g in g0)
    fr = json.load(open(TFR.READ, encoding="utf-8"))["models"][MODEL]["arms"]
    rep = {}
    for arm in ("I", "tv", "tvp"):
        if isinstance(row.get(arm), dict) and isinstance(fr.get(arm), dict):
            rep[arm] = (abs(row[arm]["success_new"][0] - fr[arm]["success_new"][0]) < 1e-9
                        and abs(row[arm]["success_known"][0] - fr[arm]["success_known"][0]) < 1e-9
                        and abs(row[arm]["atlas"]["groups"]["cross"]["E"] - fr[arm]["atlas"]["groups"]["cross"]["E"]) < 1e-9)
    hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
    res = dict(model=MODEL, states=ks, dir=D_ATTR, B=PA.B, eps=EPS, mapping=MAPPING, arms=row, deltas=D, verdicts=V, harness=H,
               harness_matches_I=hok, I_reproduces_pp_atlas_fresh_read=chk, reused_reproduce_tv_fresh_read=rep,
               state_files_match_tv_fresh_prereg=files_ok, reused_rollouts_match_tv_attr_prereg=reused_ok)
    print("== %s: verdicts %s | harness matches I: %s | I reproduces pp_atlas_fresh_read: %s | reused reproduce tv_fresh_read: %s" % (
        MODEL, V, hok, chk, rep), flush=True)
    print("   state files match tv_fresh prereg: %s | reused rollouts match tv_attr prereg: %s of %s" % (
        all(files_ok.values()), sum(reused_ok.values()), len(reused_ok)), flush=True)
    if H and not hok:
        for arm, hh in H.items():
            for c, h in hh.items():
                if h["args_diff"] or not h["episodes_match"]:
                    print("   HARNESS %s %s %s" % (arm, c, h), flush=True)
    for arm in READ_ARMS:
        r = row.get(arm)
        if isinstance(r, str) or r is None:
            print("   %-4s %s" % (arm, r), flush=True)
            continue
        g = r["atlas"]["groups"]
        print("   %-4s new %s known %s | grasp ok %.3f cross-stage %.3f | within E %s cross E %s -> %s" % (
            arm, TF._f(r["success_new"]), TF._f(r["success_known"]), r["grasp_ok_new"], r["cross_stage_share_new"],
            TF._f([g["within"]["E"]] + g["within"]["ci"]), TF._f([g["cross"]["E"]] + g["cross"]["ci"]), r["atlas"]["verdict"]), flush=True)
        print("        per task %s" % json.dumps({c: round(v, 3) for c, v in r["per_task_success"].items()}), flush=True)
        if r.get("side"):
            print("        side %s" % json.dumps({k: (TF._r(v) if isinstance(v, float) else v) for k, v in r["side"].items()
                                                  if k not in ("segmenter_iou", "paint")}), flush=True)
    for k, v in D.items():
        if k != "known":
            print("   %s %s" % (k, TF._f(v)), flush=True)
    print("   known %s" % {k: TF._f(v) for k, v in D["known"].items()}, flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


def _ratio(a, b):
    return None if not b else a / float(b)


def summary(d):
    """Smoke table: tv_frozen.summary's per-episode lines (it reads _tv.json side files of any arm)."""
    TF.summary(d)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        return summary(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read",))
    ap.add_argument("--dir", default=os.path.join(TF.OUT, "tv_attr"))
    ap.add_argument("--out", default=os.path.join(TF.OUT, "tv_attr_read.json"))
    read(ap.parse_args())


if __name__ == "__main__":
    main()

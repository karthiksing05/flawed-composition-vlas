"""V1 (docs/RIVAL_TESTS.md 1; m3_prereg pp_rival1): N2's client, unchanged, with the prompt following the hold state.  Only the prompt each query carries differs by arm;
images, state, seeds and episode protocol are N2's.  The oracle contact signal times the switch.  Also writes <out>_track.json: per episode, the hold's onset, the prompts
used and the final positions of every slot.  LIBERO venv (py3.8).  V1_ARM: intact | A | S | W.

    V1_ARM=A python experiments/pp_v1_client.py --port P --suite n2_goal --tasks 16 --trials 1 --fresh results/flawed/n2_states \
        --branches results/flawed/n2_branches_main.json --videos 0 --bind track --out X.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_client as FN                                                   # noqa: E402
import fa1_common as FC                                                     # noqa: E402
import fa1b_common as FB                                                    # noqa: E402

LC = FN.LC
ARM = os.environ["V1_ARM"]
assert ARM in ("intact", "A", "S", "W")
WAIT = 10
OBJ_PARTNER = {"cream_cheese_1": "K6", "wine_bottle_1": "K4"}
TGT_PARTNER = {"plate_1": "K1", "flat_stove_1_cook_region": "K2", "wooden_cabinet_1_top_side": "K3", "akita_black_bowl_1": "K6"}
TID = int(sys.argv[sys.argv.index("--tasks") + 1])
OUTF = sys.argv[sys.argv.index("--out") + 1]
TASK = FN.N2["tasks"][TID]
assert TASK["tid"] == TID and TASK["compound"] in ("N1", "N2", "N3", "N4", "N5", "N6")
TEXT = {t["compound"]: t for t in FN.N2["tasks"] if t["layout"] == "familiar"}
OWN = TASK["language"]
PRE = TEXT[OBJ_PARTNER[TASK["object"]]]["language"] if ARM == "S" else OWN
WRONG = "K2" if TASK["target"] == "plate_1" else "K1"
POST = {"intact": OWN, "A": TEXT[TGT_PARTNER[TASK["target"]]]["language"], "S": TEXT[TGT_PARTNER[TASK["target"]]]["language"], "W": TEXT[WRONG]["language"]}[ARM]
assert TEXT[TGT_PARTNER[TASK["target"]]]["target"] == TASK["target"] and TEXT[WRONG]["target"] != TASK["target"]
assert ARM != "S" or TEXT[OBJ_PARTNER[TASK["object"]]]["object"] == TASK["object"]
CUR, RECS = {"env": None, "episode": None}, []


class TrackEnv(LC.OffScreenRenderEnv):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._n, self._hm, self._con, self._held, self._on_step, self._obs, self._q = 0, FB.Held(), None, False, None, None, dict(pre=0, post=0)
        CUR["env"] = self

    def step(self, a):
        r = super().step(a)
        self._n += 1
        self._obs = r[0]
        if self._n >= WAIT:
            if self._con is None:
                self._con = FB.Contact(self, [TASK["object"]])
            h, _ = self._hm.step(*self._con.touching(TASK["object"]), self._con.pos(TASK["object"]), r[0]["robot0_eef_pos"])
            if h and self._on_step is None:
                self._on_step = self._n - WAIT
            self._held = bool(h)
        return r

    def close(self):
        if self._n > 0:
            sc = FC.scene(self, self._obs, FC.drawer_joint(self))
            RECS.append(dict(episode=CUR["episode"], hold_step=self._on_step, queries=self._q, held_end=self._held,
                             final={e: sc["pos"][i].tolist() for i, e in enumerate(FC.ENT)}))
            with open(OUTF.replace(".json", "_track.json"), "w") as fh:
                json.dump(dict(arm=ARM, tid=TID, compound=TASK["compound"], layout=TASK["layout"], pre=PRE, post=POST, wrong=TEXT[WRONG]["target"], episodes=RECS), fh, indent=1)
        super().close()


class SwitchClient(LC._wcp.WebsocketClientPolicy):
    def infer(self, el):
        env = CUR["env"]
        el = dict(el)
        el["prompt"] = POST if env._held else PRE
        env._q["post" if env._held else "pre"] += 1
        CUR["episode"] = el.get("meta/episode")
        return super().infer(el)


LC.OffScreenRenderEnv = TrackEnv
LC._wcp.WebsocketClientPolicy = SwitchClient

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()

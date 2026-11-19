"""MT (SR 81, docs/notes/mt_plan.md): one client for the four arms on the moved-target states (results/flawed/mt).  LIBERO venv (py3.8).
Every arm runs through fc_n2_client / libero_client (the N2 harness, suite n2_goal, --fresh results/flawed/mt/states, --branches a cell's file,
--bind track) and writes <out>_track.npz (pp_atlas.Rec: held flag, object / target / eef positions per step, final entity positions).  Nothing a
policy sees changes.
    I     frozen pi0.5 (server: pp_serve_purify.py --model pi05 --mode intact --comp= --seed 13), pp_atlas_client's recording.
    r2    R2 (server: sc_rv.py serve --arm rview2; client --seg: exact masks), the same recording.
    raa   RAA (server: raa.py serve --arm raa; client: raa.client_main, unchanged), with pp_atlas.Rec installed UNDER raa's wrappers.
    rgp   ReGuide-perc (rg.py client --arm rgp, unchanged; it records its own track), on the I server.
    python experiments/mt_client.py --arm I|r2|raa|rgp <libero_client.py's arguments>
raa.py and rg.py are imported, not edited.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def take(name):
    i = sys.argv.index(name)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


def install_rec(FN):
    import pp_atlas as PA
    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID
    ORIG, RECS = LC.OffScreenRenderEnv, []

    class ClientRec(PA.Rec):
        def close(self):
            if self.n > 0:
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
            self._e.close()
    LC.OffScreenRenderEnv = lambda *a, **k: ClientRec(ORIG(*a, **k), TASK["object"], TASK["target"])


def main():
    arm = take("--arm")
    assert arm in ("I", "r2", "raa", "rgp"), arm
    assert "--fresh" in sys.argv and "--branches" in sys.argv and "--bind" in sys.argv
    if arm == "rgp":
        import rg
        sys.argv[1:1] = ["client", "--arm", "rgp"]
        return rg.client_main()
    import fc_n2_client as FN
    install_rec(FN)
    if arm == "raa":
        import raa
        sys.argv[1:1] = ["client", "--arm", "raa"]
        return raa.client_main()
    if arm == "r2":
        assert "--seg" in sys.argv
        sys.argv.remove("--seg")                                            # fc_n2_client's handling (_seg_px), not libero_client's own --seg
        FN._seg_px()
    FN.LC.main()


if __name__ == "__main__":
    main()

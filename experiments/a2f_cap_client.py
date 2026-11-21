"""RAA-F probe (docs/notes/raaf_probe_runbook.md): the capture client for the servo-free state-token models.  LIBERO venv.

a2_cap_client.py (unchanged, imported) with two additions for Route B (env A2F_RAA=raa_b):
  * `import raa as RAA` inside a2_cap_client.client_main resolves to raa_b.py (sys.modules alias), so the real queries run through raa_b's client, which sends
    meta/raa_crop_uv_{obj,tgt};
  * a counterfactual side query is built by a2_cap_client.ra_side, which knows nothing about the crop.  The wrapper below adds meta/raa_crop_uv_{obj,tgt} for the
    swapped-in entities (the same raafb_crop.client_uv projection raa_b's channel uses), so the cf run's crop is centred on the swapped-in entity, as the model
    would see it for that instruction.  Without it the server would give the cf run an all-zero crop.
Routes A / A0 / A+ (A2F_RAA unset or raa): behaves exactly as a2_cap_client.py.

    [A2F_RAA=raa_b] python experiments/a2f_cap_client.py client --arm RA --ra-arm raac --scene 1 ... (a2_cap_client.py's arguments)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import a2_cap_client as AC                                                 # noqa: E402

if os.environ.get("A2F_RAA") == "raa_b":
    _orig_side = AC.ra_side

    def ra_side_b(P, arm, kd, text, el, q, ph, tap, w):
        out = _orig_side(P, arm, kd, text, el, q, ph, tap, w)
        if out.get("meta/raa_ok") and "meta/raa_obj" in out:
            import numpy as np
            import raafb_crop as CR
            roles = P["PF"].role_parse(P["lex"], text) if kd == "cf" else P["roles"]
            uv = CR.client_uv(tap._e, np.asarray(out["meta/raa_obj"]), np.asarray(out["meta/raa_tgt"]), None, None, roles[0], roles[1])
            out.update(uv)
        return out
    AC.ra_side = ra_side_b
    import raa_b                                                           # noqa: E402
    sys.modules["raa"] = raa_b                                             # a2_cap_client.client_main: `import raa as RAA` -> raa_b

if __name__ == "__main__":
    assert len(sys.argv) > 1 and sys.argv[1] == "client"
    AC.client_main()

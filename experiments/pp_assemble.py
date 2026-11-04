"""PARTS_RESULTS 52 (consolidation): assemble one checkpoint from two DECOUPLED builds of the same seed -- the object role (its
network, its role buffers and its flags) from --obj, the target role (network, role buffers, flags) from --tgt.  No training, no
new parameter.  CPU.

Under decoupled training (PARTS_RESULTS 35-36) a role's term is a function of its own design, its own rows and the seed only, so
the assembled checkpoint is what a build with `--reuse <role>:<donor>` training the other role would give (36 (d) checked that
reuse equals joint training bitwise; this consolidation re-checks it once, `check`).  A created part trains only its own role's
rows (the other role's rows give it an exactly-zero gradient through Phi's 0/1 gate), so a created checkpoint's own role can be
assembled with the other role of a different base the same way.

  build    --obj <ckpt rel> --tgt <ckpt rel> --out <ckpt rel> [--set-rtclip X]
           Shared buffers (center, size, opart, tpart) must be equal.  The interface (gm, tmode: fitted from the build's demos by
           a fixed rule, not trained) comes from --obj; a created target part's m comes from --tgt, whose inventory must then be
           equal.  Meta: seed, steps, tasks, decouple must agree.
           --set-rtclip: write this scalar into the target role's library-wide carry radius (the rtclip leave-one-out; only valid
           where pp_cons_check.py rtclip has shown the scalar inert).  The output must not exist.
           -> <out>, results/flawed/pp_assemble_<out tag>.json
  check    --a <ckpt rel> --b <ckpt rel> --role tgt|obj : the role's parameters and buffers bitwise equal (assembly validity).

    python experiments/pp_assemble.py build --obj pp_full_nsz_s10/P.npz --tgt pp_full_pdc_s10/P.npz --out pp_full_nts_s10/P.npz
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
from pp_p_train import ROLE_BUFFERS, ROLE_FLAGS                             # noqa: E402
from lifelong_ood.policy import parts_flow as PF                            # noqa: E402

CK = os.path.join(FC.DATA, "ckpt")
SHARED = ("center", "size", "opart", "tpart")
FLAG_ARGS = dict(tsize="tsize", aframe="aframe", psupport="psupport", tscale="tscale", nmode="nmode", rtransport="rtransport",
                 bbnom="bbnom", bbnosize="bbnosize", tnear="tnear")


def load(rel):
    m, meta = PF.PartsFlow.load(os.path.join(CK, rel))
    return m, meta


def build(a):
    out = os.path.join(CK, a.out)
    assert not os.path.exists(out), f"{out} exists"
    mo, mo_meta = load(a.obj)
    mt, mt_meta = load(a.tgt)
    for k in ("seed", "steps", "decouple"):
        assert mo_meta.get(k) == mt_meta.get(k), ("meta differs", k, mo_meta.get(k), mt_meta.get(k))
    assert mo_meta.get("decouple"), "assembly needs decoupled builds"
    assert list(mo_meta["tasks"]) == list(mt_meta["tasks"]), ("tasks differ", mo_meta["tasks"], mt_meta["tasks"])
    assert mo.rank == mt.rank and mo.H == mt.H, "rank or horizon differs"
    assert mo.psupport == mt.psupport, "psupport differs"                   # the flag is read by both roles' radius rule
    so, st = mo.state_dict(), mt.state_dict()
    for k in SHARED:
        assert torch.equal(so[k], st[k]), ("shared buffer differs", k)
    created_t = mt_meta.get("created") if mt_meta.get("role") == "tgt" else None
    created_o = mo_meta.get("created") if mo_meta.get("role") == "obj" else None
    # The interface (inventory gm, each target part's m) is fitted from the build's demo rows by a fixed rule, not trained; it
    # comes from the object donor (which reads it).  A created target part's m comes from the target donor, which must then
    # carry the same inventory.  A target donor built without --imode (nmode 0) carries no interface and contributes none.
    tmode = so["tmode"].clone()
    if mt.nmode > 0:
        assert mt.nmode == mo.nmode and torch.equal(so["gm"], st["gm"]), "inventory differs"
        keep = torch.ones(len(PF.TGT_ATOMS), dtype=torch.bool)
        if created_t:
            keep[PF.TGT_ATOMS.index(created_t)] = False
            tmode[~keep] = st["tmode"][~keep]
        assert torch.equal(so["tmode"][keep], st["tmode"][keep]), "tmode differs outside a created target part"
    else:
        assert not created_t, "a created target part needs its donor's interface"
    kw = dict(tsize=mt.tsize, aframe=mo.aframe, psupport=mo.psupport, tscale=mt.tscale, nmode=mo.nmode, rtransport=mo.rtransport,
              bbnom=mo.bbnom, bbnosize=mo.bbnosize, tnear=mt.tnear)
    m = PF.PartsFlow(mo.H, rank=mo.rank, **kw)
    sd = m.state_dict()
    def donor(k):
        if k.startswith("obj.") or k in ROLE_BUFFERS["obj"] or k == "gm":
            return so
        if k.startswith("tgt.") or k in ROLE_BUFFERS["tgt"]:
            return st
        return None
    for k in sd:
        if k == "tmode":
            sd[k] = tmode.clone()
        elif donor(k) is not None:
            sd[k] = donor(k)[k].clone()
        else:
            assert torch.equal(so[k], st[k]), ("unassigned buffer differs", k)
            sd[k] = so[k].clone()
    if a.set_rtclip is not None:
        sd["rtclip"] = torch.tensor(float(a.set_rtclip), dtype=torch.float32)
    m.load_state_dict(sd)
    for k, v in m.state_dict().items():                                      # every tensor comes from exactly the intended donor
        if (k == "rtclip" and a.set_rtclip is not None) or k == "tmode":
            continue
        assert torch.equal(v, (donor(k) or so)[k]), ("copy failed", k)
    assert torch.equal(m.state_dict()["tmode"], tmode), "copy failed: tmode"
    meta = dict(mo_meta)
    for k in ("created", "role", "task", "k", "demos"):                     # a created part's meta comes with its own role
        meta.pop(k, None)
    if created_o:
        meta.update({k: mo_meta[k] for k in ("created", "role", "task", "k", "demos") if k in mo_meta})
    if created_t:
        meta.update({k: mt_meta[k] for k in ("created", "role", "task", "k", "demos") if k in mt_meta})
    tag = os.path.basename(os.path.dirname(a.out))
    meta.update(tag=tag[3:].rsplit("_s", 1)[0], assembled=dict(obj=a.obj, tgt=a.tgt, set_rtclip=a.set_rtclip))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m.save(out, meta)
    rep = dict(out=a.out, obj=a.obj, tgt=a.tgt, set_rtclip=a.set_rtclip, flags=kw, created_obj=created_o, created_tgt=created_t,
               seed=mo_meta["seed"], tasks=mo_meta["tasks"])
    fn = os.path.join(ROOT, "results", "flawed", f"pp_assemble_{tag}_{os.path.basename(a.out)[:-4]}.json")
    assert not os.path.exists(fn), f"{fn} exists"
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("[assemble]", json.dumps(rep), flush=True)


def check(a):
    ma, _ = load(a.a)
    mb, _ = load(a.b)
    sa, sb = ma.state_dict(), mb.state_dict()
    keys = [k for k in sa if k.startswith(a.role + ".") or k in ROLE_BUFFERS[a.role]]
    diff = {k: float((sa[k].double() - sb[k].double()).abs().max()) for k in keys}
    flags = {f: (getattr(ma, f), getattr(mb, f)) for f in ROLE_FLAGS[a.role]}
    ok = all(torch.equal(sa[k], sb[k]) for k in keys) and all(x == y for x, y in flags.values())
    print("[check]", a.a, a.b, a.role, "n_tensors", len(keys), "max_abs_diff", max(diff.values()), "flags", flags, "BITWISE_EQUAL", ok,
          flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "check"))
    ap.add_argument("--obj")
    ap.add_argument("--tgt")
    ap.add_argument("--out")
    ap.add_argument("--set-rtclip", type=float, default=None)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--role", choices=("obj", "tgt"))
    a = ap.parse_args()
    if a.mode == "build":
        build(a)
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())

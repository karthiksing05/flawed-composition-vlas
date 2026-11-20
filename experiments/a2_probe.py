"""SR 83 (A2 distributedness probe, docs/notes/a2_plan.md): open-loop activation patching and ablation on recorded model inputs.  GPU, openpi venv.

The recorded inputs are what experiments/a2_capture.py stored at the GuidedPolicy (after the role views, paint and channel of the RAA family):
images, state, prompt, the noise seed.  For each observation the clean run (the instruction as given) and the counterfactual run (the swapped
word; for RAA-sL the whole front end's output for the swapped instruction) share one noise draw.  Everything is done in PyTorch hooks on the served
model (lifelong_ood/policy/backbone.PolicyBackbone, not edited): the VLM prefix pass (18 layers) and the action expert (18 layers x 10 denoising steps).

Components (a "write" = a sublayer's output before the residual add; per-row patches in one batch, each batch carries a clean reference row):
  vlm.L.attn|mlp.all|ins    replace layer L's attention / MLP write by the counterfactual run's, at all prefix positions / at the instruction tokens
  vlm.L.resid.all|ins       replace the layer's output hidden state (cumulative profile; descriptive)
  exp.L.attn|mlp|resid      the same in the action expert (the counterfactual run's activation at the SAME denoising step)
  vlm|exp.L.head.H          one attention head's output (the o_proj input slice), layers HEAD_LAYERS
  in.prefix|tok|servo|chan  (RAA-sL) the prefix (images + prompt) / the 4-dim state token / the servo input / token + servo, from the counterfactual run
  abl.*                     zero-ablation of layer bands' writes on the clean run; abl.chan|tok|servo: the role channel removed (raw state zeros = the unrouted convention)
Outputs one npz per observation under --out: the unnormalised chunk of every row, and the index of each row's in-batch clean reference.

    python experiments/a2_probe.py --model pi05|raasl --cap DIR[+DIR..] --out DIR --sets F7+N+K [--limit N] [--ident]
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
FL = os.path.join(ROOT, "results", "flawed")
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_main6000"
NL = 18
HEAD_LAYERS = (3, 9, 15)
NHEAD, HD = 8, 256
VLM_BANDS = ((0, 3), (4, 8), (9, 13), (14, 17))
EXP_BANDS = ((0, 8), (9, 17))
PER_TASK = 12                                            # registered: first 12 captured observations (trial order) per N / K task
NK_TIDS = {"K": (0, 2, 6, 10), "N": (16, 22, 24, 26)}
F7_TRIALS = range(4, 24)
B1, B2 = "akita_black_bowl_1", "akita_black_bowl_2"


# ----------------------------------------------------------------------------------------------------------------- patching machinery
class Patcher:
    def __init__(self, bk):
        self.bk, self.m = bk, bk.model
        pw = bk.model.paligemma_with_expert
        self.stacks = {"vlm": list(pw.paligemma.language_model.layers), "exp": list(pw.gemma_expert.model.layers)}
        self.plan, self.cf, self.cl, self.rec, self.step = {}, {}, {}, None, None
        self.servo = float(getattr(bk.model.config, "servo_k", 0.0) or 0.0) != 0.0
        for where, layers in self.stacks.items():
            for li, ly in enumerate(layers):
                ly.self_attn.register_forward_hook(self._hook(where, li, "attn"))
                ly.mlp.register_forward_hook(self._hook(where, li, "mlp"))
                ly.register_forward_hook(self._hook(where, li, "resid"))
                ly.self_attn.o_proj.register_forward_pre_hook(self._phook(where, li))

    def _hook(self, where, li, kind):
        def h(mod, args, out):
            if kind in ("attn", "resid") and isinstance(out, tuple):
                y = self.apply((where, li, kind), out[0])
                return out if y is out[0] else (y,) + tuple(out[1:])
            y = self.apply((where, li, kind), out)
            return out if y is out else y
        return h

    def _phook(self, where, li):
        def h(mod, args):
            y = self.apply((where, li, "head"), args[0])
            return None if y is args[0] else (y,) + tuple(args[1:])
        return h

    def apply(self, key, y):
        sk = (key, self.step if key[0] == "exp" else None)
        if self.rec is not None:
            self.rec[sk] = y.detach().clone()
        specs = self.plan.get(key)
        if not specs:
            return y
        y = y.clone()
        for (r, pos, hd, op, src, ins) in specs:
            sl = slice(None) if pos == "all" else slice(ins[0], ins[0] + ins[1])
            if op == "zero":
                y[r, sl] = 0
                continue
            S = (self.cf if src == "cf" else self.cl)[sk]
            if key[2] == "head":
                a, b = hd * HD, (hd + 1) * HD
                y[r, :, a:b] = S[0, :, a:b].to(y.dtype)
            else:
                y[r, sl] = S[0, sl].to(y.dtype)
        return y

    def state(self, dicts):
        obs = self.bk.observations(dicts)
        return self.m._preprocess_observation(obs, train=False)[4]

    @torch.no_grad()
    def run(self, rows, seed, rec=None):
        """rows: dicts(pre, tok, srv: observation dicts; comps: [(where, li, kind, pos, head, op, src)]).  -> unnormalised chunks [B, H, 7]."""
        bk, B = self.bk, len(rows)
        observation = bk.observations([r["pre"] for r in rows])
        off, n = bk.text_span(observation)
        plan = {}
        for i, r in enumerate(rows):
            for (where, li, kind, pos, hd, op, src) in r["comps"]:
                plan.setdefault((where, li, kind), []).append((i, pos, hd, op, src, (int(off), int(n[i]))))
        self.plan, self.rec, self.step = plan, rec, None
        pre = bk.prefix(observation)
        pre.state = self.state([r["tok"] for r in rows]).to(pre.state.dtype)
        sv = self.state([r["srv"] for r in rows])
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn((1, bk.dim_full), generator=g).to(bk.device).float().expand(B, -1).contiguous()
        K = bk.num_steps
        dt = torch.tensor(-1.0 / K, dtype=torch.float32, device=bk.device)
        t = torch.tensor(1.0, dtype=torch.float32, device=bk.device)
        k = 0
        while t >= -dt / 2:
            self.step = k
            x = x + dt * bk.v_full(x, t.expand(B), pre)
            t = t + dt
            k += 1
        self.step = None
        if self.servo:
            from lifelong_ood.policy.raas_model import servo_norm_torch
            s = servo_norm_torch(bk.model.config, sv, bk.A)[..., :3]
            x = x.reshape(B, bk.H, bk.A).clone()
            x[:, :, :3] = x[:, :, :3] + s.to(x.dtype)
            x = x.reshape(B, bk.dim_full)
        real = x[:, bk.real].float().cpu().numpy()
        out = [bk.unnormalize(real[i:i + 1], state=pre.state[i].float().cpu().numpy())[0] for i in range(B)]
        self.plan, self.rec = {}, None
        return np.stack(out)


# ----------------------------------------------------------------------------------------------------------------- row construction
def comp(where, li, kind, pos="all", hd=-1, op="patch", src="cf"):
    return (where, li, kind, pos, hd, op, src)


def build_rows(model, clean, cf, ident):
    """-> list of (name, row).  Every row is a counterfactual / ablation of the clean observation."""
    R = []

    def add(name, comps=(), pre=clean, tok=None, srv=None):
        R.append((name, dict(pre=pre, tok=clean if tok is None else tok, srv=clean if srv is None else srv, comps=list(comps))))
    add("total", pre=cf, tok=cf, srv=cf)
    if ident:
        add("ident.vlm5.attn.all", [comp("vlm", 5, "attn", src="cl")])
        add("ident.vlm9.mlp.ins", [comp("vlm", 9, "mlp", "ins", src="cl")])
        add("ident.exp9.resid", [comp("exp", 9, "resid", src="cl")])
        add("ident.vlm9.head3", [comp("vlm", 9, "head", hd=3, src="cl")])
    for li in range(NL):
        for kind in ("attn", "mlp"):
            add("vlm.%d.%s.all" % (li, kind), [comp("vlm", li, kind)])
            add("vlm.%d.%s.ins" % (li, kind), [comp("vlm", li, kind, "ins")])
            add("exp.%d.%s.all" % (li, kind), [comp("exp", li, kind)])
        add("vlm.%d.resid.all" % li, [comp("vlm", li, "resid")])
        add("vlm.%d.resid.ins" % li, [comp("vlm", li, "resid", "ins")])
        add("exp.%d.resid.all" % li, [comp("exp", li, "resid")])
    for where in ("vlm", "exp"):
        for li in HEAD_LAYERS:
            for h in range(NHEAD):
                add("%s.%d.head.%d" % (where, li, h), [comp(where, li, "head", hd=h)])
    if model == "raasl":
        add("in.prefix", pre=cf, tok=clean, srv=clean)
        add("in.tok", pre=clean, tok=cf, srv=clean)
        add("in.servo", pre=clean, tok=clean, srv=cf)
        add("in.chan", pre=clean, tok=cf, srv=cf)
    for bi, (a, b) in enumerate(VLM_BANDS):
        for pos in ("all", "ins"):
            add("abl.vlm.b%d.%s" % (bi, pos), [comp("vlm", li, k, pos, op="zero") for li in range(a, b + 1) for k in ("attn", "mlp")])
    for bi, (a, b) in enumerate(EXP_BANDS):
        add("abl.exp.b%d" % bi, [comp("exp", li, k, op="zero") for li in range(a, b + 1) for k in ("attn", "mlp")])
    if model == "raasl":
        z = dict(clean)
        z["observation/state"] = np.zeros_like(np.asarray(clean["observation/state"]))
        add("abl.chan", pre=clean, tok=z, srv=z)
        add("abl.tok", pre=clean, tok=z, srv=clean)
        add("abl.servo", pre=clean, tok=clean, srv=z)
    return R


# ----------------------------------------------------------------------------------------------------------------- recorded inputs
def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {"observation/image": z["image"], "observation/wrist_image": z["wrist"], "observation/state": z["state"], "prompt": str(z["prompt"])}
    return d, int(z["seed"]), z["actions"]


def ntok(P, d):
    return int(P.bk.observation(d).tokenized_prompt_mask.sum())


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def obs_nk(P, caps, model, sset):
    """N / K: the 3rd held query of each captured episode; the first swap candidate (steer_text.cands order) whose prompt has as many tokens."""
    out, skipped = [], []
    for tid in NK_TIDS[sset]:
        recs = []
        for c in caps:
            recs += jl(os.path.join(c, "%s_nk_%d_cap.jsonl" % (model, tid)))
        recs.sort(key=lambda r: r["trial"])
        got = 0
        for r in recs:
            if got >= PER_TASK:
                break
            od = None
            for c in caps:
                p = os.path.join(c, "obs", re.sub(r"[^A-Za-z0-9_.+-]", "_", r["base"] + "_true.npz"))
                if os.path.exists(p):
                    od = os.path.join(c, "obs")
            if od is None:
                skipped.append((r["base"], "no true obs"))
                continue
            clean, seed, act = load_npz(os.path.join(od, re.sub(r"[^A-Za-z0-9_.+-]", "_", r["base"] + "_true.npz")))
            pick = None
            for cand in r["cands"]:
                p = os.path.join(od, re.sub(r"[^A-Za-z0-9_.+-]", "_", "%s_cf_%s.npz" % (r["base"], cand)))
                if not os.path.exists(p):
                    continue
                cf, seed2, _ = load_npz(p)
                if seed2 == seed and ntok(P, cf) == ntok(P, clean):
                    pick = (cand, cf)
                    break
            if pick is None:
                skipped.append((r["base"], "no equal-length candidate"))
                continue
            got += 1
            out.append(dict(id="%s_%s" % (model, r["base"]), set=sset, task=r["key"], trial=r["trial"], cluster="%s_%d_%d" % (sset, tid, r["trial"]),
                            clean=clean, cf=pick[1], seed=seed, cap_actions=act, eef=r["eef"], p_cor=r["p_tgt"], p_cf=r["p_cand"][pick[0]],
                            cand=pick[0], phase="place"))
    return out, skipped


def obs_f7(P, caps, model, geom_caps):
    """F7: row 0 of rgc tids 0 (relL) and 1 (relR); the counterfactual of a state is the other task's instruction on the same layout (SP_R is SP_L with
    the bowls swapped, identical images).  pi0.5: the clean images with the partner's prompt; RAA-sL: the partner task's recorded model input."""
    cache = pickle.load(open(os.path.join(FL, "rg_comp", "cache_f7.pkl"), "rb"))
    acc = cache["acc"]
    out, skipped = [], []

    def f(c, tag, tr):
        return os.path.join(c, "obs", "row0_%s_t%d.npz" % (tag, tr))
    for tr in F7_TRIALS:
        pair = {}
        for tid in (0, 1):
            pair[tid] = None
            for c in caps:
                if os.path.exists(f(c, "tid%d" % tid, tr)):
                    pair[tid] = load_npz(f(c, "tid%d" % tid, tr))
        if pair[0] is None or pair[1] is None:
            skipped.append((tr, "missing row0 obs"))
            continue
        eef = None
        for c in geom_caps:
            if os.path.exists(f(c, "tid0", tr)):
                eef = [float(v) for v in np.load(f(c, "tid0", tr))["state"][:3]]
        sp = acc[tr]["lay"]["SP"]
        pos = {0: (sp[B1], sp[B2]), 1: (sp[B2], sp[B1])}                  # (B1 = the object of the task, the other bowl)
        for tid in (0, 1):
            clean, seed, act = pair[tid]
            o = pair[1 - tid][0]
            if model == "pi05":
                cf = dict(clean)
                cf["prompt"] = o["prompt"]
            else:
                cf = o
            if ntok(P, cf) != ntok(P, clean):
                skipped.append((tr, tid, "token counts differ"))
                continue
            out.append(dict(id="%s_f7_tid%d_t%d" % (model, tid, tr), set="F7", task="F7_%s" % ("relL" if tid == 0 else "relR"), trial=tr,
                            cluster="F7_%d" % tr, clean=clean, cf=cf, seed=seed, cap_actions=act, eef=eef, p_cor=[float(v) for v in pos[tid][0]],
                            p_cf=[float(v) for v in pos[tid][1]], cand="other_bowl", phase="grasp"))
    return out, skipped


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("pi05", "raasl"))
    ap.add_argument("--cap", required=True, help="capture dirs (+-separated: sbatch --export splits on commas) holding <model>_nk_<tid>_cap.jsonl / obs/")
    ap.add_argument("--geom", default="", help="pi05 F7 capture dirs (the eef of row 0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="F7+N+K")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--ident", action="store_true", help="plumbing identity rows on every observation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from lifelong_ood.policy.backbone import PolicyBackbone
    if a.model == "pi05":
        bk = PolicyBackbone("pi05_libero", device="cuda")
    else:
        bk = PolicyBackbone("pi05_g8v_raas_l_ae", checkpoint_dir=CK, device="cuda")
    print("[a2] %s" % bk.describe(), flush=True)
    P = Patcher(bk)
    caps = [c for c in a.cap.split("+") if c]
    geom = [c for c in a.geom.split("+") if c]
    todo, skipped = [], {}
    for s in a.sets.split("+"):
        o, sk = obs_f7(P, caps, a.model, geom or caps) if s == "F7" else obs_nk(P, caps, a.model, s)
        todo += o
        skipped[s] = sk
        print("[a2] set %s: %d observations, %d skipped %s" % (s, len(o), len(sk), sk[:6]), flush=True)
    if a.limit:
        todo = todo[:a.limit]
    json.dump(dict(skipped={k: [list(map(str, x)) for x in v] for k, v in skipped.items()}, n=len(todo)), open(os.path.join(a.out, "selection_%s.json" % a.sets), "w"))
    for oi, ob in enumerate(todo):
        path = os.path.join(a.out, ob["id"] + ".npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        # the counterfactual run's activations (and the clean run's, for the identity rows)
        P.cf, P.cl = {}, {}
        rec = {}
        P.run([dict(pre=ob["cf"], tok=ob["cf"], srv=ob["cf"], comps=[])], ob["seed"], rec=rec)
        P.cf = rec
        if a.ident:
            rec2 = {}
            P.run([dict(pre=ob["clean"], tok=ob["clean"], srv=ob["clean"], comps=[])], ob["seed"], rec=rec2)
            P.cl = rec2
        R = build_rows(a.model, ob["clean"], ob["cf"], a.ident)
        names, chunks, ref = [], [], []
        per = a.batch - 1
        for i in range(0, len(R), per):
            part = R[i:i + per]
            rows = [dict(pre=ob["clean"], tok=ob["clean"], srv=ob["clean"], comps=[])] + [r for _, r in part]
            ch = P.run(rows, ob["seed"])
            r0 = len(names)
            names += ["ref"] + [n for n, _ in part]
            chunks += list(ch)
            ref += [r0] * len(rows)
        np.savez(path, names=np.array(names), chunks=np.stack(chunks).astype(np.float32), ref=np.array(ref), cap_actions=ob["cap_actions"],
                 meta=np.array(json.dumps({k: ob[k] for k in ("id", "set", "task", "trial", "cluster", "seed", "eef", "p_cor", "p_cf", "cand", "phase")})))
        c0 = np.stack(chunks)[0]
        print("[a2] %d/%d %s rows %d %.0fs replay-vs-server max|diff| %.4g" % (oi + 1, len(todo), ob["id"], len(names), time.time() - t0,
                                                                          float(np.abs(c0[:, :7] - ob["cap_actions"][:, :7]).max())), flush=True)
    print("[a2] done", flush=True)


if __name__ == "__main__":
    main()

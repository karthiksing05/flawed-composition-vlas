"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): the comparison arms on the generated test units (and S1's dev units).
openpi venv, GPU.  Every hyper-parameter is S1's / S1c's (no test- or dev-based choice).

  --holistic  the T1 comparator: S1's B1b (s1_train.build: one head per role, phrase in the policy's own word embeddings, image
              tokens + a learned 16 x 16 position; per-token BCE + softmax CE against the SIM target mask), RETRAINED on the
              generated training pool's labels (units.json "train"; 40 epochs, AdamW 3e-4, seed 0); training layouts with
              hash % 8 == 0 are a within-training check (reported only) -> results/bind/b1_3/b1b/pred_{test,dev}.npz
  --gdino     B1a (s1_b1a.query: role phrase minus its goal preposition; best box; centre token) on the test units, plus
              S1b's body boxes (s1b_detect: body phrase, threshold 0.05, NMS 0.5, at most 6) for S1c -> b1a/pred_test.npz, det_test.json
  --s1c       S1c AS TRAINED in S1 (results/bind/s1c/{object,goal}.pt; LIBERO training cells), applied to the test units with the
              GroundingDINO body boxes -> s1c/pred_test.npz
  --smoke     holistic: 2 epochs, within-training check only (no test or dev unit)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
B13 = os.path.join(ROOT, "results", "bind", "b1_3")
S1 = os.path.join(ROOT, "results", "bind", "s1")
FE = os.path.join(B13, "feats")


def need_prereg():
    from lifelong_ood import ledger
    assert any(r.get("tag") == "pb1_3" for r in ledger.load("m3_prereg")), "pre-register first (experiments/prereg/pb1_3.py)"


def val_layout(lay):
    return int(hashlib.sha256(("b13-b1b-val" + lay).encode()).hexdigest(), 16) % 8 == 0


def holistic(torch, U, lab, smoke):
    import s1_train as ST
    torch.manual_seed(ST.SEED)
    np.random.seed(ST.SEED)
    pz = np.load(os.path.join(FE, "phrases.npz"), allow_pickle=True)
    pidx = {str(p): i for i, p in enumerate(pz["phrases"])}
    PH, PL = torch.tensor(pz["emb"].astype(np.float32)), torch.tensor(pz["len"])
    dev = torch.device("cuda")
    feats = {"train": np.load(os.path.join(FE, "siglip_train.npy")), "test": np.load(os.path.join(FE, "siglip_test.npy")),
             "dev": np.load(os.path.join(S1, "feats", "dev_img.npy"))}
    fidx = {"train": {f: i for i, f in enumerate(json.load(open(os.path.join(FE, "siglip_train_frames.json"))))},
            "test": {f: i for i, f in enumerate(json.load(open(os.path.join(FE, "siglip_test_frames.json"))))},
            "dev": {f: i for i, f in enumerate(json.load(open(os.path.join(S1, "feats", "dev_frames.json"))))}}
    FT = {s: torch.tensor(v, device=dev) for s, v in feats.items()}
    cache = {}

    def zc(f):
        if f not in cache:
            cache[f] = np.load(os.path.join(ROOT, f), allow_pickle=True)
        return cache[f]
    Head = ST.build(torch)
    od = os.path.join(B13, "b1b")
    os.makedirs(od, exist_ok=True)
    log, preds = {}, {"test": {}, "dev": {}}
    for role in ("object", "goal"):
        ex = [e for e in U["train"] if e["role"] == role]
        tr = [e for e in ex if not val_layout(e["layout"])]
        va = [e for e in ex if val_layout(e["layout"])]
        T = lambda es: torch.tensor(np.stack([ST.target_of(e, zc) for e in es]))             # noqa: E731
        Xi = lambda es, s: torch.tensor([fidx[s][e["frame"]] for e in es])                  # noqa: E731
        Pi = lambda es: torch.tensor([pidx[e["phrase"]] for e in es])                       # noqa: E731
        trI, trT, trP = Xi(tr, "train"), T(tr), Pi(tr)
        vaI, vaT, vaP = Xi(va, "train"), T(va), Pi(va)
        net = Head().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=ST.LR, weight_decay=ST.WD)
        t0, hist = time.time(), []
        for epoch in range(2 if smoke else ST.EPOCHS):
            net.train()
            perm = torch.randperm(len(tr))
            tot = 0.0
            for g in range(0, len(tr), ST.BATCH):
                b = perm[g:g + ST.BATCH]
                img, tgt, pi = FT["train"][trI[b].to(dev)].float(), trT[b].to(dev), trP[b]
                lo = net(img, PH[pi].to(dev), PL[pi].to(dev))
                bce = torch.nn.functional.binary_cross_entropy_with_logits(lo, tgt.clamp(0, 1))
                m = (tgt > 0).float()
                ce = -(torch.log_softmax(lo, -1) * (m / m.sum(-1, keepdim=True).clamp(min=1))).sum(-1).mean()
                loss = bce + ce
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            net.eval()
            with torch.no_grad():
                hits = []
                for g in range(0, len(va), 256):
                    lo = net(FT["train"][vaI[g:g + 256].to(dev)].float(), PH[vaP[g:g + 256]].to(dev), PL[vaP[g:g + 256]].to(dev))
                    pk = lo.argmax(-1).cpu()
                    hits.append((vaT[g:g + 256][torch.arange(len(pk)), pk] > 0).float())
                vh = float(torch.cat(hits).mean()) if hits else float("nan")
            hist.append(dict(epoch=epoch, loss=tot / len(tr), val_hit=vh))
            print(f"[b1.3 b1b] {role} epoch {epoch}: loss {tot / len(tr):.4f} within-train val hit {vh:.3f} ({time.time() - t0:.0f}s)", flush=True)
        log[role] = dict(n_train=len(tr), n_val=len(va), hist=hist, secs=time.time() - t0)
        if smoke:
            continue
        torch.save(net.state_dict(), os.path.join(od, f"{role}.pt"))
        for s, units in (("test", U["test"]), ("dev", lab["dev"])):
            us = [e for e in units if e["role"] == role]
            with torch.no_grad():
                dI, dP = Xi(us, s), Pi(us)
                lo = torch.cat([net(FT[s][dI[g:g + 256].to(dev)].float(), PH[dP[g:g + 256]].to(dev), PL[dP[g:g + 256]].to(dev)).cpu()
                                for g in range(0, len(us), 256)])
            for e, l in zip(us, lo.numpy()):
                preds[s][f"{role}|{e['frame']}"] = l.astype(np.float32)
    for s, p in preds.items():
        if p:
            keys = sorted(p)
            np.savez(os.path.join(od, f"pred_{s}.npz"), keys=np.array(keys), logits=np.stack([p[k] for k in keys]))
    json.dump(log, open(os.path.join(od, "train_log_smoke.json" if smoke else "train_log.json"), "w"), indent=1)


def gdino(torch, U):
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import s1_b1a as BA
    import s1b_detect as SD
    dev = torch.device("cuda")
    proc = AutoProcessor.from_pretrained(BA.MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(BA.MODEL).to(dev).eval()
    lex = json.load(open(os.path.join(ROOT, "results", "bind", "s1b", "det.json")))["lexicon"]
    keys, covs, peaks, scores, qs, det = [], [], [], [], [], {}

    def run(im, text, thr):
        inp = proc(images=im, text=text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        return proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=thr, target_sizes=[im.size[::-1]])[0]
    for e in U["test"]:
        im = Image.fromarray(np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)["img"])
        q = BA.query(e["phrase"])
        r = run(im, q, 0.0)
        cov = np.zeros((224, 224), np.float32)
        if len(r["scores"]):
            i = int(r["scores"].argmax())
            x0, y0, x1, y1 = [float(v) for v in r["boxes"][i]]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(223, x1), min(223, y1)
            cov[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = 1.0
            pk = int(min(15, ((y0 + y1) / 2) // 14)) * 16 + int(min(15, ((x0 + x1) / 2) // 14))
            sc = float(r["scores"][i])
        else:
            pk, sc = -1, 0.0
        keys.append(f"{e['role']}|{e['frame']}")
        covs.append(cov.reshape(16, 14, 16, 14).mean((1, 3)).reshape(-1))
        peaks.append(pk)
        scores.append(sc)
        qs.append(q)
        body = SD.split(e["phrase"], lex)[0]
        k = f"{e['frame']}|{body}"
        if k not in det:
            rb = run(im, body + ".", SD.THR)
            bx = sorted(([*map(float, bb), float(s)] for bb, s in zip(rb["boxes"], rb["scores"])), key=lambda x: -x[4])
            keep = []
            for x in bx:
                if all(SD.iou(x, kk) < SD.NMS for kk in keep):
                    keep.append(x)
                if len(keep) == SD.K_MAX:
                    break
            det[k] = keep
    od = os.path.join(B13, "b1a")
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, "pred_test.npz"), keys=np.array(keys), cov=np.stack(covs).astype(np.float32), peak=np.array(peaks),
             score=np.array(scores), query=np.array(qs))
    json.dump(dict(lexicon=lex, det=det), open(os.path.join(B13, "det_test.json"), "w"))
    print(f"[b1.3 gdino] {len(keys)} test units; {len(det)} body queries", flush=True)


def s1c(torch, U):
    import s1b_detect as SD
    import s1b_train as SB
    dev = torch.device("cuda")
    D = json.load(open(os.path.join(B13, "det_test.json")))
    lex, det = D["lexicon"], D["det"]
    pz = np.load(os.path.join(FE, "phrases.npz"), allow_pickle=True)
    pidx = {str(p): i for i, p in enumerate(pz["phrases"])}
    PH, PL = torch.tensor(pz["emb"][:, :12].astype(np.float32)), torch.tensor(pz["len"]).clamp(max=12)
    FT = torch.tensor(np.load(os.path.join(FE, "siglip_test.npy")), device=dev)
    fidx = {f: i for i, f in enumerate(json.load(open(os.path.join(FE, "siglip_test_frames.json"))))}
    Head = SB.build(torch)
    preds = {}
    for role in ("object", "goal"):
        net = Head().to(dev)
        net.load_state_dict(torch.load(os.path.join(ROOT, "results", "bind", "s1c", f"{role}.pt"), map_location=dev))
        net.eval()
        for e in [e for e in U["test"] if e["role"] == role]:
            key = f"{role}|{e['frame']}"
            body = SD.split(e["phrase"], lex)[0]
            boxes = det.get(f"{e['frame']}|{body}", [])[:SB.KMAX]
            if not boxes:
                preds[key] = dict(peak=-1, mask=np.zeros(256, bool))
                continue
            cl = np.stack([SB.cells(b) for b in boxes])
            gm = SB.geom(boxes)
            K, G2 = len(boxes), SB.G * SB.G
            fm = FT[torch.tensor([fidx[e["frame"]]], device=dev)].float().transpose(1, 2).reshape(1, 2048, 16, 16)
            grid = torch.tensor(cl[None] / 224.0 * 2 - 1, device=dev, dtype=torch.float32).reshape(1, K * G2, 1, 2)
            crop = torch.nn.functional.grid_sample(fm, grid, align_corners=False).squeeze(-1).transpose(1, 2).reshape(1, K, G2, 2048)
            ri = torch.tensor([pidx[e["phrase"]]])
            with torch.no_grad():
                il, ml = net(crop, torch.tensor(gm[None], device=dev, dtype=torch.float32), PH[ri].to(dev), PL[ri].to(dev))
            c = int(il[0].argmax())
            mm = ml[0, c].cpu().numpy()
            msk = np.zeros(256, bool)
            msk[SB.tok_of(cl[c][1 / (1 + np.exp(-mm)) >= 0.5])] = True
            preds[key] = dict(peak=int(SB.tok_of(cl[c][int(mm.argmax())])), mask=msk)
    od = os.path.join(B13, "s1c")
    os.makedirs(od, exist_ok=True)
    keys = sorted(preds)
    np.savez(os.path.join(od, "pred_test.npz"), keys=np.array(keys), peak=np.array([preds[k]["peak"] for k in keys]),
             mask=np.stack([preds[k]["mask"] for k in keys]))
    print(f"[b1.3 s1c] {len(keys)} test units", flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--holistic", action="store_true")
    ap.add_argument("--gdino", action="store_true")
    ap.add_argument("--s1c", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    U = json.load(open(os.path.join(B13, "units.json"), encoding="utf-8"))
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    if not a.smoke:
        need_prereg()
    if a.holistic:
        holistic(torch, U, lab, a.smoke)
    if a.gdino:
        gdino(torch, U)
    if a.s1c:
        s1c(torch, U)


if __name__ == "__main__":
    main()

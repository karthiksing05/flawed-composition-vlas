"""RAA-F Route B dataset checker (openpi venv, CPU): lol/libero_g8v_raacb (or --repo) against lol/libero_g8v_raac.  Row-for-row equality of state, actions and prompt
(the crop is the ONLY change), crop_image present at 224 x 224 x 3, crop non-zero on every routed (K1-K6, state role != 0) frame and all-zero on every unrouted one,
episode / frame counts, and the crop's feature in info.json.  Prints PASS / FAIL, exits 1 on FAIL."""
import argparse, glob, io, json, os, sys
import numpy as np
HOME = "/coc/flash10/ksingara3/robot/data/hf/lerobot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lol/libero_g8v_raacb")
    ap.add_argument("--ref", default="lol/libero_g8v_raac")
    ap.add_argument("--root", default="")
    ap.add_argument("--nimg", type=int, default=40, help="episodes whose crop images are decoded (evenly spaced)")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    root = a.root or os.path.join(HOME, a.repo)
    ref = os.path.join(HOME, a.ref)
    fb, fr = sorted(glob.glob(root + "/data/*/episode_*.parquet")), sorted(glob.glob(ref + "/data/*/episode_*.parquet"))
    tb = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(root + "/meta/tasks.jsonl")}
    tr = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(ref + "/meta/tasks.jsonl")}
    info = json.load(open(root + "/meta/info.json"))
    bad = []
    nb = sum(len(pq.read_table(f, columns=["actions"])) for f in fb)
    nr = sum(len(pq.read_table(f, columns=["actions"])) for f in fr)
    print("[chk] episodes %d vs ref %d; frames %d vs ref %d" % (len(fb), len(fr), nb, nr))
    if len(fb) != len(fr) or nb != nr:
        bad.append("counts")
    ft = info["features"].get("crop_image")
    print("[chk] crop_image feature %s" % ft)
    if not ft or list(ft["shape"]) != [224, 224, 3]:
        bad.append("crop feature")
    pick = set(np.unique(np.linspace(0, len(fb) - 1, a.nimg).astype(int)).tolist())
    nz_routed = z_routed = nz_unrouted = n_unrouted = 0
    for i, (f1, f2) in enumerate(zip(fb, fr)):
        c1 = pq.read_table(f1).to_pydict()
        c2 = pq.read_table(f2, columns=["state", "actions", "task_index"]).to_pydict()
        s1, s2 = np.asarray(c1["state"], np.float32), np.asarray(c2["state"], np.float32)
        if not (np.array_equal(s1, s2) and np.array_equal(np.asarray(c1["actions"], np.float32), np.asarray(c2["actions"], np.float32))):
            bad.append("state/actions differ in " + os.path.basename(f1))
        if [tb[t] for t in c1["task_index"]] != [tr[t] for t in c2["task_index"]]:
            bad.append("prompt differs in " + os.path.basename(f1))
        if i in pick:
            routed = s1[:, 3] > 0.5
            for k, x in enumerate(c1["crop_image"]):
                im = np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB"))
                assert im.shape == (224, 224, 3), im.shape
                if routed[k]:
                    nz_routed += 1
                    z_routed += int(not im.any())
                else:
                    n_unrouted += 1
                    nz_unrouted += int(im.any())
    print("[chk] decoded %d episodes: routed frames %d (all-zero crops: %d), unrouted %d (non-zero crops: %d)" % (len(pick), nz_routed, z_routed, n_unrouted, nz_unrouted))
    if z_routed or nz_unrouted:
        bad.append("crop zero/non-zero pattern")
    print("[chk] " + ("FAIL: " + "; ".join(bad[:8]) if bad else "PASS"))
    sys.exit(1 if bad else 0)


main()

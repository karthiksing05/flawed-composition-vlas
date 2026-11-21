"""RAA-F: dataset sanity (openpi venv, sbatch only).  Episode order is original, copy, copy per eligible demo: a copy's rows must equal the original's rows h0..e-1 with actions xyz + K delta, state rel + delta."""
import json, sys, pathlib
import numpy as np, pyarrow.parquet as pq
repo, delta = sys.argv[1:3]
K = 2.851209548740752
D = {(d["task"], d["demo"]): d for d in json.load(open(delta))["demos"]}
fs = sorted((pathlib.Path("/coc/flash10/ksingara3/robot/data/hf/lerobot") / repo / "data").rglob("episode_*.parquet"))
E = [pq.read_table(f, columns=["state", "actions"]).to_pydict() for f in fs]
print("episodes", len(E), "frames", sum(len(e["actions"]) for e in E))
n, bad = 0, 0
for i, e in enumerate(E):
    st = np.asarray(e["state"]); ac = np.asarray(e["actions"])
    if i and st[:, 3].min() == 2 and len(ac) < len(np.asarray(E[i - 1]["actions"])) or False:
        pass
# find copies: whole episode has role 2 on every row and previous original-or-copy source
last = None
for i, e in enumerate(E):
    st = np.asarray(e["state"]); ac = np.asarray(e["actions"])
    if np.all(st[:, 3] == 2) and last is not None and len(ac) <= len(last[1]):
        so, ao = last
        found = False
        for h0 in range(len(ao) - len(ac) + 1):
            dl = (ac[0, :3] - ao[h0, :3]) / K
            if np.allclose(ac[:, :3], ao[h0:h0 + len(ac), :3] + K * dl, atol=1e-5) and np.allclose(ac[:, 3:], ao[h0:h0 + len(ac), 3:]) and np.allclose(st[:, :3], so[h0:h0 + len(ac), :3] + dl, atol=1e-5) and np.allclose(st[:, 3], 2):
                found = True; n += 1; break
        bad += int(not found)
    else:
        last = (st, ac)
print("copies verified", n, "bad", bad)

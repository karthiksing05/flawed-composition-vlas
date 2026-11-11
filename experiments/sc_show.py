"""Print a results JSON compactly, for reading through a SLURM job (result files are not read on the login node).

    python experiments/sc_show.py results/flawed/sc_q0_read.json [--keys a b.c ...] [--depth 3] [--width 4000]

Floats are rounded to 3 places.  Lists longer than 12 items and dicts deeper than --depth are summarised by their size.
"""
import argparse
import json


def trim(x, depth):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        if depth <= 0:
            return f"<dict {len(x)}>"
        return {k: trim(v, depth - 1) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) > 12:
            return f"<list {len(x)}>"
        return [trim(v, depth - 1) for v in x]
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=4000)
    a = ap.parse_args()
    d = json.load(open(a.path, encoding="utf-8"))
    if not a.keys:
        print("[show] keys:", list(d) if isinstance(d, dict) else type(d).__name__, flush=True)
        a.keys = list(d) if isinstance(d, dict) else []
    for k in a.keys:
        x = d
        for part in k.split("."):
            x = x[part] if isinstance(x, dict) else x[int(part)]
        print(f"[show] {k} = {json.dumps(trim(x, a.depth))[:a.width]}", flush=True)

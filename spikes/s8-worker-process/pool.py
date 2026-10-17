"""Pool per-round latency CSVs and print p50/p99 and the overhead against the direct baseline."""

import glob
import re
import sys
from collections import defaultdict

out = sys.argv[1]
us = sys.argv[2]
depth = sys.argv[3] if len(sys.argv) > 3 else "1"
runs = defaultdict(list)
for f in sorted(glob.glob(f"{out}/lat_*_{us}us_d{depth}_r*.csv")):
    mode = re.search(r"lat_(\w+?)_", f.split("/")[-1]).group(1)
    with open(f) as fh:
        runs[mode].append([int(x) for x in fh])


def q(v, p):
    v = sorted(v)
    return v[int(p / 100 * (len(v) - 1) + 0.5)] / 1000


T = float(us)
print(f"gpu_us={us} depth={depth}  (2% of step = {0.02 * T:.0f} us)")
base = {}
for mode in ("direct", "inproc", "proc"):
    if mode not in runs:
        continue
    pooled = [x for r in runs[mode] for x in r]
    per_round = ", ".join(f"{q(r, 50):.1f}/{q(r, 99):.1f}" for r in runs[mode])
    base[mode] = (q(pooled, 50), q(pooled, 99), q(pooled, 99.9))
    stats = f"p50={base[mode][0]:9.1f} p99={base[mode][1]:9.1f} p99.9={base[mode][2]:9.1f} us"
    print(f"{mode:7s} n={len(pooled):6d} {stats}   per-round p50/p99: {per_round}")
for a, b in (("direct", "inproc"), ("direct", "proc"), ("inproc", "proc")):
    if a in base and b in base:
        d50 = base[b][0] - base[a][0]
        d99 = base[b][1] - base[a][1]
        print(
            f"{b} - {a}: p50 {d50:+.1f} us ({d50 / T * 100:+.2f}% of step)  p99 {d99:+.1f} us ({d99 / T * 100:+.2f}%)"
        )

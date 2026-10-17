#!/usr/bin/env python3
"""Runs a command and prints wall seconds and peak RSS of the child (no /usr/bin/time on this host)."""

import resource
import subprocess
import sys
import time

t = time.time()
rc = subprocess.call(sys.argv[1:])
print(
    f"MEASURE wall_s={time.time() - t:.1f} peak_rss_gib={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576:.2f} exit={rc} cmd={' '.join(sys.argv[1:3])} ..."
)
sys.exit(rc)

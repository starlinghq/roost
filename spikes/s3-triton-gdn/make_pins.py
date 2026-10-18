"""Pick one config per kernel from an autotuned capture: the winner of the serving-relevant launch
(varlen with initial state for the chunk kernels, varlen for the recurrent kernel)."""

import json
import sys
from pathlib import Path

specs = json.loads(Path(sys.argv[1]).read_text())
TUNABLE = {
    "l2norm_fwd_kernel": ["BT"],
    "chunk_local_cumsum_scalar_kernel": [],
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": ["BK"],
    "recompute_w_u_fwd_kernel": [],  # BK, BV are fixed by the caller
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ["BV"],
    "chunk_fwd_kernel_o": ["BK", "BV"],
}
pins = {}
for r in specs:
    n = r["kernel"]
    if n not in TUNABLE:
        continue
    c = r["constexprs"]
    good = c.get("IS_VARLEN") is True and c.get("USE_INITIAL_STATE", True) is not False
    if n in pins and not good:
        continue
    if n in pins and pins[n]["_good"]:
        continue
    pins[n] = dict(
        kwargs={k: c[k] for k in TUNABLE[n]}, num_warps=r["num_warps"], num_stages=r["num_stages"], _good=good
    )
for p in pins.values():
    p.pop("_good")
Path(sys.argv[2]).write_text(json.dumps(pins, indent=1))
print(json.dumps(pins, indent=1))

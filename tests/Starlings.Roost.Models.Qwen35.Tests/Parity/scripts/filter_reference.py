"""Keep only the named cases of a reference.jsonl, in the file's own order, byte for byte.

usage: filter_reference.py REFERENCE.jsonl OUT.jsonl CASE[,CASE...]

The R-022 negative runs cover two of the five cases (capital, long-context). `evaluate-layers` reads the reference's
case list, so the two-case reference must be a line-for-line subset of reference.jsonl; this copies the matching lines
unchanged.
"""

import json
import sys
from pathlib import Path


def main(reference: Path, out: Path, cases: list[str]) -> int:
    kept = [line for line in reference.read_text().splitlines(keepends=True) if json.loads(line)["id"] in cases]
    found = {json.loads(line)["id"] for line in kept}
    if found != set(cases):
        print(f"missing cases: {sorted(set(cases) - found)}", file=sys.stderr)
        return 1
    out.write_text("".join(kept))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")))

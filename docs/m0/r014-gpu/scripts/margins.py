import json
import sys

for bound in (0.2, 0.25):
    n = 0
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                n += sum(1 for m in d["step_margin"] if m < bound)
    print(bound, n)

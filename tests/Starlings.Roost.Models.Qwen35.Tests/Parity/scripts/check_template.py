"""R-022: compare the prompt set rendered by Starling with the checkpoint's own template.

Starling's side is Qwen35ChatTemplate and Starlings.Tokenizers (Qwen35CpuParityRun.WritesPromptSet); the other side
is the checkpoint's Jinja template and the transformers tokenizer. Test-time only; run with the evidence
pipeline's `reference` extra.

    python -I check_template.py <snapshot directory> <run directory with rendered.jsonl>

Prints one line per case and exits 1 if any text or id sequence differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(snapshot: Path, run: Path) -> int:
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot))
    failed = 0
    for line in (run / "rendered.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        jinja = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=case["enable_thinking"]
        )
        ids = list(tokenizer(case["rendered"], add_special_tokens=False)["input_ids"])
        jinja_ids = list(tokenizer(jinja, add_special_tokens=False)["input_ids"])
        same_text = jinja == case["rendered"]
        same_ids = ids == case["prompt_ids"] and jinja_ids == case["prompt_ids"]
        failed += not (same_text and same_ids)
        print(f"{case['id']}: {len(case['prompt_ids'])} tokens; text equal {same_text}; ids equal {same_ids}")
    print(f"transformers {transformers.__version__}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))

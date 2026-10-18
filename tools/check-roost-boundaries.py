#!/usr/bin/env python3
"""Roost's repository boundaries (build plan R-003). Exit 1 with one line per violation.

    python3 -I tools/check-roost-boundaries.py            # every check; run `dotnet restore Roost.slnx` first
    python3 -I tools/check-roost-boundaries.py --no-closure   # skip the restore-dependent closure check

Checks:
  python     No Python or libtorch under src/: no Python source files, and no project, C# or native file that
             references Python bindings or libtorch (pythonnet, TorchSharp, libtorch, torch extension headers).
  closure    No Microsoft.ML.OnnxRuntime* package in the restored dependency closure (obj/project.assets.json)
             of a worker-side package: one that runs in or is compiled into the GPU worker process, which pins
             its own CUDA runtime and allocator (build plan decision 30). Host-process packages (Engine,
             Generation, Generation.Roost, Server) may use ONNX Runtime.
  binaries   No kernel or native binaries tracked in Git (by extension and by ELF, PE, Mach-O or fatbin magic).
             Kernel binaries are release assets only.
  solution   No *.Cuda project in Roost.slnx. CUDA test projects are local-only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Worker-side packages (build plan decision 30). Starlings.Roost.Models.* matches every model package.
WORKER_SIDE = re.compile(
    r"^Starlings\.Roost\.(Abstractions|Ops|Graph|Worker|WorkerHost|Kernels|Cuda|Native|Distributed|Models\..+)$"
)
ONNX = re.compile(r"^microsoft\.ml\.onnxruntime", re.IGNORECASE)

PYTHON_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd", ".ipynb"}
PYTHON_REFERENCE = re.compile(
    r"pythonnet|Python\.Runtime|Python\.Included|TorchSharp|libtorch|torch/extension\.h|torch/all\.h"
    r"|ATen/|c10/|TORCH_CHECK|TORCH_LIBRARY|\bat::Tensor\b|python3\d*\.(dll|so)|libpython",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".cs", ".csproj", ".props", ".targets", ".h", ".hpp", ".c", ".cc", ".cpp", ".cu", ".cuh", ".json"}

BINARY_SUFFIXES = {
    ".so",
    ".dll",
    ".dylib",
    ".a",
    ".lib",
    ".o",
    ".obj",
    ".exe",
    ".cubin",
    ".fatbin",
    ".ptx",
    ".pdb",
    ".nupkg",
}
BINARY_MAGIC = (
    b"\x7fELF",  # ELF (Linux .so, .cubin)
    b"MZ",  # PE (Windows .dll, .exe)
    b"\xcf\xfa\xed\xfe",  # Mach-O 64
    b"\xfe\xed\xfa\xcf",
    b"\x50\xed\x55\xba",  # CUDA fatbin
)


def tracked(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    # Submodules appear as one gitlink path; their contents are another repository's.
    return [root / p for p in out.decode().split("\0") if p and (root / p).is_file()]


def check_python(root: Path, files: list[Path]) -> list[str]:
    errors = []
    src = root / "src"
    for path in files:
        if src not in path.parents:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in PYTHON_SUFFIXES:
            errors.append(f"python: {rel}: Python source under src/")
        elif path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in PYTHON_REFERENCE.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"python: {rel}:{line}: references {match.group()!r}")
    return errors


def check_closure(root: Path) -> list[str]:
    errors = []
    for project in sorted((root / "src").glob("*/*.csproj")):
        if not WORKER_SIDE.match(project.stem):
            continue
        assets = project.parent / "obj" / "project.assets.json"
        rel = project.relative_to(root).as_posix()
        if not assets.is_file():
            errors.append(f"closure: {rel}: not restored (run `dotnet restore Roost.slnx`)")
            continue
        libraries = json.loads(assets.read_text(encoding="utf-8")).get("libraries", {})
        for key in sorted(libraries):
            name = key.split("/", 1)[0]
            if ONNX.match(name):
                errors.append(f"closure: {rel}: worker-side package reaches {key}")
    return errors


def check_binaries(root: Path, files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"binaries: {rel}: native binary tracked in Git")
            continue
        with path.open("rb") as handle:
            head = handle.read(4)
        if any(head.startswith(magic) for magic in BINARY_MAGIC):
            errors.append(f"binaries: {rel}: native binary tracked in Git (magic {head.hex()})")
    return errors


def check_solution(root: Path) -> list[str]:
    solution = root / "Roost.slnx"
    if not solution.is_file():
        return ["solution: Roost.slnx missing"]
    paths = re.findall(r'Path="([^"]+\.csproj)"', solution.read_text(encoding="utf-8"))
    return [
        f"solution: Roost.slnx lists {p}; CUDA projects stay out of the solution"
        for p in paths
        if re.search(r"\.Cuda(\.|$)", Path(p).stem)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-closure", action="store_true", help="skip the restore-dependent closure check")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    files = tracked(root)
    errors = check_python(root, files) + check_binaries(root, files) + check_solution(root)
    if not args.no_closure:
        errors += check_closure(root)
    for error in errors:
        print(error)
    if errors:
        print(f"{len(errors)} boundary violation(s)", file=sys.stderr)
        return 1
    print("roost boundaries: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

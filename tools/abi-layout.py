#!/usr/bin/env python3
"""Golden layout of the C contract structs (R-004): C1 in roost_kernels.h, C2 and C8 in roost_worker_shm.h.

    python3 -I tools/abi-layout.py program            # print the C program that prints the layout
    python3 -I tools/abi-layout.py check              # build it with gcc, run it, diff against the golden file
    python3 -I tools/abi-layout.py check --update     # rewrite the golden file (a contract change; stream A only)
    python3 -I tools/abi-layout.py headers            # compile the headers: gcc -std=c11 -pedantic, g++ -std=c++17
    python3 -I tools/abi-layout.py msvc-asserts       # compile the golden layout as static asserts with clang in
                                                      # MSVC mode (x86_64-pc-windows-msvc); not MSVC itself
    python -I tools/abi-layout.py msvc                # MSVC itself (cl.exe on PATH, a VS developer shell): compile
                                                      # the headers as C11 and C++17 with /W4 /WX and the golden
                                                      # layout as static asserts; CI's abi-msvc job on windows-latest

The golden file is read by the C# layout test (tests/Starlings.Roost.Abstractions.Tests), which compares it with
Marshal.OffsetOf and Unsafe.SizeOf on every OS, so the C# test never needs a C compiler. Build output goes to a
temporary directory, never into the repository.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ROOT / "kernels" / "include"
HEADERS = ("roost_kernels.h", "roost_worker_shm.h")
GOLDEN = ROOT / "tests" / "Starlings.Roost.Abstractions.Tests" / "Layout" / "abi-layout.golden.txt"

STRUCT = re.compile(r"typedef struct (\w+) \{(.*?)\} \1;", re.DOTALL)
FIELD = re.compile(r"^\s*(?P<type>(?:const\s+)?[A-Za-z_][\w ]*?[\s\*]+)(?P<name>\w+);")
FN_TYPEDEF = re.compile(r"typedef [^;]*?\(\*(\w+)\)\(")

# Field type classes recorded in the golden file, so that a change of type that keeps the size (uint64_t to void*,
# float to uint32_t) still changes the golden file and fails the C# layout test. Struct-typed fields record the
# struct name; everything with a '*' and every function-pointer typedef is "ptr".
SCALARS = {
    "uint8_t": "u8",
    "int8_t": "i8",
    "uint16_t": "u16",
    "int16_t": "i16",
    "uint32_t": "u32",
    "int32_t": "i32",
    "uint64_t": "u64",
    "int64_t": "i64",
    "float": "f32",
    "double": "f64",
}
COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

GCC_FLAGS = ("-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic")
GXX_FLAGS = ("-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic")


def type_class(ctype: str, fn_typedefs: set[str], struct_names: set[str]) -> str:
    """The golden type class of a C field type: u32, f32, ptr, a struct name and so on."""
    spelled = " ".join(ctype.replace("*", " * ").split())
    if "*" in spelled:
        return "ptr"
    base = spelled.removeprefix("const ")
    if base in fn_typedefs:
        return "ptr"
    if base in SCALARS:
        return SCALARS[base]
    if base in struct_names:
        return base
    raise ValueError(f"field type {ctype.strip()!r} has no golden type class; use a fixed-width type")


def parse_typed_structs(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Struct names and (field, C type) pairs, in declaration order, from a header in this repository's style."""
    text = COMMENT.sub("", text)
    structs = []
    for match in STRUCT.finditer(text):
        fields = []
        for line in match.group(2).splitlines():
            if not line.strip():
                continue
            field = FIELD.match(line)
            if field is None:
                raise ValueError(f"{match.group(1)}: cannot parse field line {line.strip()!r}")
            fields.append((field.group("name"), field.group("type")))
        structs.append((match.group(1), fields))
    return structs


def parse_structs(text: str) -> list[tuple[str, list[str]]]:
    """Struct names and field names, in declaration order."""
    return [(name, [field for field, _ in fields]) for name, fields in parse_typed_structs(text)]


def all_typed_structs(include: Path = INCLUDE) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """(header, struct, [(field, type class)]) for every contract struct, in header order."""
    texts = {header: (include / header).read_text(encoding="utf-8") for header in HEADERS}
    fn_typedefs = {m for text in texts.values() for m in FN_TYPEDEF.findall(COMMENT.sub("", text))}
    parsed = [(header, parse_typed_structs(text)) for header, text in texts.items()]
    struct_names = {name for _, structs in parsed for name, _ in structs}
    return [
        (header, name, [(field, type_class(ctype, fn_typedefs, struct_names)) for field, ctype in fields])
        for header, structs in parsed
        for name, fields in structs
    ]


def all_structs(include: Path = INCLUDE) -> list[tuple[str, str, list[str]]]:
    return [(header, name, [field for field, _ in fields]) for header, name, fields in all_typed_structs(include)]


def program(include: Path = INCLUDE) -> str:
    lines = [
        "/* Generated by tools/abi-layout.py; prints the layout of every contract struct. */",
        "#include <stddef.h>",
        "#include <stdio.h>",
        *(f'#include "{header}"' for header in HEADERS),
        "",
        '#define S(T) printf("struct %s size=%zu align=%zu\\n", #T, sizeof(T), _Alignof(T))',
        '#define F(T, f, k) printf("  %s %zu %zu %s\\n", #f, offsetof(T, f), sizeof(((T*)0)->f), k)',
        "/* The compiler confirms each scalar type class the parser read from the header. */",
        "#define K(T, f, ctype) \\",
        '  _Static_assert(_Generic(((T*)0)->f, ctype: 1, default: 0), #T "." #f " is " #ctype)',
        "",
        "int main(void) {",
        '  printf("# Generated by tools/abi-layout.py from kernels/include; do not edit by hand.\\n");',
        '  printf("# struct <name> size=<bytes> align=<bytes>, then <field> <offset> <size> <type>, x86-64 LP64.\\n");',
        '  printf("# <type> is u8..u64, i8..i64, f32, f64, ptr (any pointer) or the name of a contract struct.\\n");',
    ]
    ctypes = {v: k for k, v in SCALARS.items()}
    for _, name, fields in all_typed_structs(include):
        lines.append(f"  S({name});")
        for field, kind in fields:
            if kind in ctypes:
                lines.append(f"  K({name}, {field}, {ctypes[kind]});")
            elif kind != "ptr":
                lines.append(f"  K({name}, {field}, {kind});")
            lines.append(f'  F({name}, {field}, "{kind}");')
    lines += ["  return 0;", "}", ""]
    return "\n".join(lines)


def compiler(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} not found on PATH")
    return path


def run_layout(include: Path = INCLUDE) -> str:
    with tempfile.TemporaryDirectory(prefix="roost-abi-layout-") as tmp:
        source = Path(tmp) / "print_layout.c"
        source.write_text(program(include), encoding="utf-8")
        exe = Path(tmp) / "print_layout"
        subprocess.run([compiler("gcc"), *GCC_FLAGS, f"-I{include}", str(source), "-o", str(exe)], check=True)
        return subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout


def padding_errors(layout: str) -> list[str]:
    """Implicit padding in a layout listing: the contract rule is explicit reserved fields, never compiler padding."""
    errors: list[str] = []
    name, size, end = None, 0, 0

    def finish() -> None:
        if name is not None and end != size:
            errors.append(f"{name}: {size - end} bytes of tail padding")

    for row in layout.splitlines():
        parts = row.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "struct":
            finish()
            name, size, end = parts[1], int(parts[2].split("=")[1]), 0
            continue
        offset, width = int(parts[1]), int(parts[2])
        if offset != end:
            errors.append(f"{name}.{parts[0]}: {offset - end} bytes of padding before it")
        end = offset + width
    finish()
    return errors


def check(update: bool) -> int:
    actual = run_layout()
    padding = padding_errors(actual)
    if padding:
        print("\n".join(padding))
        print("add explicit reserved fields instead of relying on compiler padding")
        return 1
    if update:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(actual, encoding="utf-8", newline="\n")
        print(f"wrote {GOLDEN.relative_to(ROOT)}")
        return 0
    expected = GOLDEN.read_text(encoding="utf-8").replace("\r\n", "\n") if GOLDEN.exists() else ""
    if actual == expected:
        print(f"{GOLDEN.relative_to(ROOT)} matches gcc")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(expected.splitlines(keepends=True), actual.splitlines(keepends=True), "golden", "gcc", n=1)
    )
    print("layout differs from the golden file; a layout change is a contract change (README.md, Contracts)")
    return 1


def headers() -> int:
    with tempfile.TemporaryDirectory(prefix="roost-abi-headers-") as tmp:
        for header in HEADERS:
            for tool, flags, suffix in (("gcc", GCC_FLAGS, ".c"), ("g++", GXX_FLAGS, ".cpp")):
                source = Path(tmp) / f"use_{Path(header).stem}{suffix}"
                source.write_text(f'#include "{header}"\nint main(void) {{ return 0; }}\n', encoding="utf-8")
                cmd = [compiler(tool), *flags, f"-I{INCLUDE}", "-c", str(source), "-o", str(source) + ".o"]
                subprocess.run(cmd, check=True)
                print(f"{tool} {' '.join(flags)}: {header} ok")
    return 0


def golden_asserts() -> str:
    """The golden layout as C11 static asserts on sizeof and offsetof."""
    golden = GOLDEN.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = ["#include <stddef.h>", *(f'#include "{h}"' for h in HEADERS)]
    current = None
    for row in golden.splitlines():
        if not row or row.startswith("#"):
            continue
        parts = row.split()
        if parts[0] == "struct":
            current = parts[1]
            size = parts[2].split("=")[1]
            lines.append(f'_Static_assert(sizeof({current}) == {size}, "{current} size");')
        else:
            field, offset = parts[0], parts[1]
            lines.append(f'_Static_assert(offsetof({current}, {field}) == {offset}, "{current}.{field}");')
    return "\n".join(lines) + "\n"


def msvc_asserts() -> int:
    """Static-assert the golden layout under clang's MSVC target (LLP64 and MSVC struct layout rules)."""
    clang = os.environ.get("CLANG", "clang")
    with tempfile.TemporaryDirectory(prefix="roost-abi-msvc-") as tmp:
        source = Path(tmp) / "msvc_asserts.c"
        source.write_text(golden_asserts(), encoding="utf-8")
        cmd = [
            compiler(clang),
            "--target=x86_64-pc-windows-msvc",
            "-fms-compatibility",
            "-ffreestanding",
            "-std=c11",
            "-Wall",
            "-Werror",
            "-fsyntax-only",
            f"-I{INCLUDE}",
            str(source),
        ]
        subprocess.run(cmd, check=True)
    print("clang --target=x86_64-pc-windows-msvc: golden layout holds")
    return 0


MSVC_C_FLAGS = ("/nologo", "/W4", "/WX", "/std:c11", "/TC")
MSVC_CXX_FLAGS = ("/nologo", "/W4", "/WX", "/std:c++17", "/TP", "/EHsc")


def msvc() -> int:
    """MSVC itself: both headers as C11 and C++17 with warnings as errors, and the golden layout as static asserts."""
    cl = compiler("cl")
    with tempfile.TemporaryDirectory(prefix="roost-abi-cl-") as tmp:
        jobs = []
        for header in HEADERS:
            for flags, suffix in ((MSVC_C_FLAGS, ".c"), (MSVC_CXX_FLAGS, ".cpp")):
                source = Path(tmp) / f"use_{Path(header).stem}{suffix}"
                source.write_text(f'#include "{header}"\nint main(void) {{ return 0; }}\n', encoding="utf-8")
                jobs.append((flags, source, f"{header} ({suffix[1:]})"))
        asserts = Path(tmp) / "msvc_asserts.c"
        asserts.write_text(golden_asserts(), encoding="utf-8")
        jobs.append((MSVC_C_FLAGS, asserts, "golden layout static asserts"))
        for flags, source, label in jobs:
            cmd = [cl, *flags, f"/I{INCLUDE}", "/c", str(source), f"/Fo{source}.obj"]
            subprocess.run(cmd, check=True)
            print(f"cl {' '.join(flags)}: {label} ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("program")
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--update", action="store_true")
    sub.add_parser("headers")
    sub.add_parser("msvc-asserts")
    sub.add_parser("msvc")
    args = parser.parse_args(argv)
    if args.command == "program":
        sys.stdout.write(program())
        return 0
    if args.command == "check":
        return check(args.update)
    if args.command == "headers":
        return headers()
    if args.command == "msvc":
        return msvc()
    return msvc_asserts()


if __name__ == "__main__":
    sys.exit(main())

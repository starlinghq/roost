#!/usr/bin/env bash
# Release drill for Roost (R-060, docs/releasing.md). The release workflow runs it, and it runs on a CPU-only host.
#
# From a clean clone of the committed HEAD, with an empty private NuGet cache, it
#   1. restores Roost.slnx in locked mode, runs the CPU test suite and the Roost boundary check,
#   2. packs every project marked IsPackable, plus every Starlings.Roost.* project a packed package depends on (a
#      ProjectReference becomes a package dependency, so the set must be closed or a consumer cannot restore),
#      at the given version (VersionPrefix and VersionSuffix are global properties; no file is edited),
#   3. checks every nuspec: version, the MIT OR Apache-2.0 license expression, repository URL and commit, every
#      Starlings.* dependency an exact [x.y.z] pin, every Starlings.Roost.* dependency one of the packed packages,
#   4. with --kernel-assets, checks the kernel release assets: one library per SM of kernels/manifest.toml, digests equal
#      to the manifest's (kernels/tools/record_digests.py --check), and the manifest asset equal to the committed one,
#   5. pushes the packages to a local folder feed (never nuget.org),
#   6. builds a consumer outside the checkout whose NuGet.config knows only that feed and nuget.org (Starlings.Roost.*
#      mapped to the feed), with its own empty package cache; checks that the restored packages are the packed bytes and
#      that no worker-side package reaches Microsoft.ML.OnnxRuntime* (decision 30), then compiles and runs it. The
#      consumer prints RESULT: PASS.
#
#   tools/release-drill.sh --version 0.0.1-drill.1 --no-kernels       # local, CPU only
#   tools/release-drill.sh --version 0.1.0-preview.1 --kernel-assets <dir>   # what the release workflow runs
#   DRILL_KEEP=1 ...      keep the temporary directory
#   DRILL_MSBUILD="-m:2"  extra MSBuild arguments (the machine's memory budget asks for -m:2 when tight)
#
# Uncommitted changes are not part of the drill: it builds exactly the committed HEAD. The packages, the kernel digest
# check and the logs are copied to --out (default <repo>/artifacts/release-drill/, ignored by Git).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
upstream_url="https://github.com/starlinghq/roost"
version=""
kernels=""
kernel_assets=""
skip_tests=0
out="$root/artifacts/release-drill"

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    exit 2
}

while (($#)); do
    case "$1" in
        --version) version="${2:?--version needs a value}"; shift 2 ;;
        --no-kernels) kernels="no"; shift ;;
        --kernel-assets) kernels="yes"; kernel_assets="$(cd "${2:?--kernel-assets needs a directory}" && pwd)"; shift 2 ;;
        --skip-tests) skip_tests=1; shift ;;
        --out) out="${2:?--out needs a directory}"; shift 2 ;;
        -h | --help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

step() { printf '\n== %s\n' "$*"; }
fail() {
    echo "DRILL FAIL: $*" >&2
    exit 1
}

[[ -n "$version" ]] || fail "--version is required (the release workflow passes the tag's version)"
[[ -n "$kernels" ]] || fail "pass --kernel-assets <dir> (a release) or --no-kernels (packages only)"
semver='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-([0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*))?$'
[[ "$version" =~ $semver ]] || fail "version '$version' is not SemVer 2 without build metadata"
prefix="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
suffix="${BASH_REMATCH[5]}"
[[ "$prefix" != "0.0.0" ]] || fail "0.0.0 is the development version (Directory.Build.props); give a release version"

dotnet="${DOTNET:-$(command -v dotnet || true)}"
if [[ -z "$dotnet" || "$("$dotnet" --list-sdks 2>/dev/null | wc -l)" -eq 0 ]]; then
    dotnet="$HOME/.dotnet/dotnet"
fi
[[ -x "$dotnet" ]] || fail "no .NET SDK found (set DOTNET)"
export DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
read -ra msbuild_extra <<<"${DRILL_MSBUILD:-}"
# No shared compiler server: a long-lived server from another build on this host must not load the model compiler's
# analyzers for the drill (it failed to load Starlings.Roost.Graph from a fresh clone path once).
msbuild_extra+=("-p:UseSharedCompilation=false")
version_props=("-p:VersionPrefix=$prefix" "-p:VersionSuffix=$suffix")

work="$(mktemp -d "${TMPDIR:-/tmp}/roost-release-drill.XXXXXX")"
cleanup() {
    if [[ "${DRILL_KEEP:-0}" == "1" ]]; then
        echo "kept: $work"
    else
        rm -rf -- "$work"
    fi
}
trap cleanup EXIT

step "0. Inputs"
sha="$(git -C "$root" rev-parse HEAD)"
echo "commit:    $sha"
echo "version:   $version (VersionPrefix $prefix, VersionSuffix '${suffix}')"
if [[ -n "$(git -C "$root" status --porcelain)" ]]; then
    echo "note:      working tree has uncommitted changes; the drill ignores them and builds $sha"
fi
echo "dotnet:    $dotnet ($("$dotnet" --version))"
echo "kernels:   ${kernel_assets:-not part of this drill (--no-kernels)}"
echo "work dir:  $work"

step "1. Clean checkout of $sha"
src="$work/src"
git clone --quiet --no-hardlinks "$root" "$src"
git -C "$src" -c advice.detachedHead=false checkout --quiet --detach "$sha"
# SourceLink and the nuspec repository element name the public repository, as in a GitHub checkout.
git -C "$src" remote set-url origin "$upstream_url"
[[ -z "$(git -C "$src" status --porcelain --ignored)" ]] || fail "clone is not clean"
echo "ok:   no build outputs, caches or untracked files in the checkout"

# Every NuGet operation from here on uses an empty, private package cache: nothing comes from ~/.nuget.
export NUGET_PACKAGES="$work/nuget-build"

step "2. Locked restore, tests, Roost boundaries"
# Restore without the version: project-reference ranges in packages.lock.json are recorded at the development version,
# and locked mode refuses any change. Pack applies the release version below (--no-restore).
(cd "$src" && CI=true "$dotnet" restore Roost.slnx "${msbuild_extra[@]}") >"$work/restore.log" 2>&1 \
    || { tail -40 "$work/restore.log"; fail "locked restore failed"; }
echo "ok:   locked restore of Roost.slnx"
if ((skip_tests)); then
    echo "note: tests skipped (--skip-tests); a release never passes this flag"
else
    (cd "$src" && CI=true "$dotnet" build Roost.slnx --configuration Release --no-restore "${msbuild_extra[@]}") \
        >"$work/build.log" 2>&1 || { tail -40 "$work/build.log"; fail "build failed"; }
    # The xunit v3 test projects are executables; run each one directly. (`dotnet test` through Microsoft.Testing.Platform
    # reports "Zero tests ran" on some hosts, which would hide whether anything was tested.)
    : >"$work/test.log"
    tests=0
    while read -r rel; do
        name="$(basename "$rel" .csproj)"
        exe="$src/$(dirname "$rel")/bin/Release/net10.0/$name"
        [[ -x "$exe" ]] || fail "test executable missing: $exe"
        "$exe" >"$work/test-$name.log" 2>&1 || { tail -60 "$work/test-$name.log"; fail "tests failed: $name"; }
        cat "$work/test-$name.log" >>"$work/test.log"
        summary="$(sed 's/\x1b\[[0-9;]*m//g' "$work/test-$name.log" | grep -a "  $name  Total:" | tail -1)"
        if [[ ! "$summary" =~ Total:\ ([0-9]+),\ Errors:\ 0,\ Failed:\ 0 ]] || ((BASH_REMATCH[1] == 0)); then
            fail "no passing test summary for $name: '$summary'"
        fi
        echo "ok:  $summary"
        tests=$((tests + 1))
    done < <(grep -o 'Path="tests/[^"]*\.csproj"' "$src/Roost.slnx" | sed 's/^Path="//; s/"$//')
    ((tests > 0)) || fail "Roost.slnx lists no test projects"
    echo "ok:   $tests test projects passed"
fi
(cd "$src" && python3 -I tools/check-roost-boundaries.py) || fail "Roost boundary check failed"

step "3. Pack (IsPackable projects and their Starlings.Roost.* dependency closure)"
pack="$work/pack"
mkdir -p "$pack"
declare -A packed=()
pack_project() { # <csproj> <force-packable 0|1>
    local project="$1" force="$2" extra=()
    ((force)) && extra=("-p:IsPackable=true")
    (cd "$src" && CI=true "$dotnet" pack "$project" --configuration Release --no-restore --output "$pack" \
        "${version_props[@]}" "${extra[@]}" "${msbuild_extra[@]}") >>"$work/pack.log" 2>&1 \
        || { tail -40 "$work/pack.log"; fail "pack failed: $project"; }
}
for project in "$src"/src/*/*.csproj; do
    name="$(basename "$project" .csproj)"
    if [[ "$(cd "$src" && "$dotnet" msbuild "$project" -getProperty:IsPackable)" == "true" ]]; then
        echo "pack: $name (IsPackable)"
        pack_project "$project" 0
        packed[$name]=marked
    fi
done
((${#packed[@]})) || fail "no project is marked IsPackable"
# Close the set: a packed package that depends on an unpacked Starlings.Roost.* project gets that project packed too.
while :; do
    missing="$(python3 -I - "$pack" <<'EOF'
import re, sys, zipfile
from pathlib import Path
ids, deps = set(), set()
for nupkg in Path(sys.argv[1]).glob("*.nupkg"):
    with zipfile.ZipFile(nupkg) as z:
        nuspec = z.read(next(n for n in z.namelist() if n.endswith(".nuspec") and "/" not in n)).decode("utf-8-sig")
    ids.add(re.search(r"<id>([^<]+)</id>", nuspec).group(1))
    deps.update(re.findall(r'<dependency id="(Starlings\.Roost\.[^"]+)"', nuspec))
print("\n".join(sorted(deps - ids)))
EOF
)"
    [[ -n "$missing" ]] || break
    while read -r id; do
        project="$src/src/$id/$id.csproj"
        [[ -f "$project" ]] || fail "a packed package depends on $id, and there is no src/$id/$id.csproj to pack"
        [[ -z "${packed[$id]:-}" ]] || fail "$id was packed but its package id differs from its project name"
        echo "pack: $id (dependency closure; not marked IsPackable)"
        pack_project "$project" 1
        packed[$id]=closure
    done <<<"$missing"
done
ls -l "$pack"

step "4. Nuspec checks"
python3 -I - "$pack" "$version" "$sha" "$upstream_url" <<'EOF' | tee "$work/nuspec-check.log" || fail "nuspec checks failed"
import hashlib, re, sys, zipfile
from pathlib import Path

pack, version, sha, url = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
errors, ids = [], {}
for nupkg in sorted(pack.glob("*.nupkg")):
    with zipfile.ZipFile(nupkg) as z:
        names = z.namelist()
        nuspec = z.read(next(n for n in names if n.endswith(".nuspec") and "/" not in n)).decode("utf-8-sig")
    pid = re.search(r"<id>([^<]+)</id>", nuspec).group(1)
    ids[pid] = (nupkg, nuspec, names)
for pid, (nupkg, nuspec, names) in ids.items():
    def need(cond, what):
        if not cond:
            errors.append(f"{pid}: {what}")
    need(nupkg.name == f"{pid}.{version}.nupkg", f"file name {nupkg.name}")
    need(f"<version>{version}</version>" in nuspec, "nuspec version")
    need('<license type="expression">MIT OR Apache-2.0</license>' in nuspec, "license expression MIT OR Apache-2.0")
    repo = re.search(r"<repository ([^>]*)/>", nuspec)
    need(repo is not None and f'url="{url}"' in repo.group(1), f"repository url {url}")
    need(repo is not None and f'commit="{sha}"' in repo.group(1), f"repository commit {sha}")
    need(not any(n.lower().endswith((".so", ".dll.so", ".cubin", ".onnx", ".safetensors")) for n in names), "native binaries or model assets packed")
    for dep, rng in re.findall(r'<dependency id="(Starlings\.[^"]+)" version="([^"]+)"', nuspec):
        need(re.fullmatch(r"\[[^,\]]+\]", rng) is not None, f"dependency {dep} {rng} is not an exact [x.y.z] pin")
        if dep.startswith("Starlings.Roost."):
            need(dep in ids, f"dependency {dep} is not one of the packed packages")
            need(rng == f"[{version}]", f"dependency {dep} {rng} is not this release's version")
    if re.search(r'<dependency id="Microsoft\.ML\.OnnxRuntime', nuspec, re.IGNORECASE):
        print(f"note: {pid} depends on ONNX Runtime directly (allowed only for host-process packages; checked in step 6)")
    print(f"ok:   {nupkg.name} sha256 {hashlib.sha256(nupkg.read_bytes()).hexdigest()}")
for e in errors:
    print("FAIL: " + e)
sys.exit(1 if errors else 0)
EOF
echo "ok:   versions, MIT OR Apache-2.0, repository $upstream_url at $sha, exact Starlings.* pins, closed Starlings.Roost.* set"

step "5. Kernel release assets"
if [[ "$kernels" == "yes" ]]; then
    cmp -s "$kernel_assets/manifest.toml" "$src/kernels/manifest.toml" || fail "the manifest asset differs from the committed kernels/manifest.toml"
    sms="$(python3 -I "$src/kernels/tools/manifest_cmake.py" "$src/kernels/manifest.toml" --sm-list)"
    IFS=';' read -ra sm_list <<<"$sms"
    for sm in "${sm_list[@]}"; do
        [[ -f "$kernel_assets/libroost_kernels_sm$sm.so" ]] || fail "no kernel library asset for SM $sm"
    done
    # A release requires the built libraries to equal the manifest's digests (CI only reports a difference).
    python3 -I "$src/kernels/tools/record_digests.py" "$kernel_assets" --check | tee "$work/kernel-digests-check.log" \
        || fail "kernel library digests differ from kernels/manifest.toml (see docs/releasing.md, 'Kernel digests')"
    echo "ok:   ${#sm_list[@]} SM libraries, digests equal to the manifest"
else
    echo "skipped (--no-kernels): this drill proves the packages only"
fi

step "6. Local feed and a clean consumer outside the checkout"
feed="$work/feed"
mkdir -p "$feed"
for nupkg in "$pack"/*.nupkg; do
    "$dotnet" nuget push "$nupkg" --source "$feed" >/dev/null
done
ls "$feed"
consumer="$work/consumer"
mkdir -p "$consumer"
cp "$root/tools/release-drill/consumer/Program.cs" "$consumer/"
references="" constants=""
for nupkg in "$pack"/*.nupkg; do
    id="$(basename "$nupkg" ".$version.nupkg")"
    references+="    <PackageReference Include=\"$id\" Version=\"[$version]\" />"$'\n'
    constants+="HAS_$(tr '.a-z' '_A-Z' <<<"$id");"
done
python3 -I - "$root/tools/release-drill/consumer/Consumer.csproj.in" "$consumer/Consumer.csproj" "$references" "$constants" "$pack" "$version" "$sha" <<'EOF'
import sys, zipfile
from pathlib import Path
assemblies = set()
for nupkg in Path(sys.argv[5]).glob("*.nupkg"):
    with zipfile.ZipFile(nupkg) as z:
        assemblies.update(Path(n).stem for n in z.namelist() if n.startswith("lib/") and n.endswith(".dll"))
template = open(sys.argv[1], encoding="utf-8").read()
text = template.replace("@REFERENCES@", sys.argv[3].rstrip("\n")).replace("@CONSTANTS@", sys.argv[4])
open(sys.argv[2], "w", encoding="utf-8").write(text.replace("@ASSEMBLIES@", ";".join(sorted(assemblies))).replace("@VERSION@", sys.argv[6]).replace("@COMMIT@", sys.argv[7]))
EOF
cat >"$consumer/NuGet.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="drill-local" value="$feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="drill-local">
      <package pattern="Starlings.Roost.*" />
    </packageSource>
    <packageSource key="nuget.org">
      <package pattern="*" />
    </packageSource>
  </packageSourceMapping>
</configuration>
EOF
export NUGET_PACKAGES="$work/nuget-consumer"
(cd "$consumer" && "$dotnet" restore "${msbuild_extra[@]}") >"$work/consumer-restore.log" 2>&1 \
    || { tail -40 "$work/consumer-restore.log"; fail "consumer restore failed"; }
(cd "$consumer" && "$dotnet" list package --include-transitive) | tee "$work/consumer-packages.txt"
python3 -I - "$consumer/obj/project.assets.json" "$pack" "$version" <<'EOF' || fail "consumer restore graph check failed"
import base64, hashlib, json, re, sys
from pathlib import Path

assets = json.load(open(sys.argv[1], encoding="utf-8"))
pack, version = Path(sys.argv[2]), sys.argv[3]
libraries = assets["libraries"]
assert all(lib["type"] == "package" for lib in libraries.values()), "a non-package library (project reference) in the consumer"
for nupkg in sorted(pack.glob("*.nupkg")):
    pid = nupkg.name[: -len(f".{version}.nupkg")]
    lib = libraries.get(f"{pid}/{version}")
    assert lib is not None, f"{pid} {version} was not restored"
    packed = base64.b64encode(hashlib.sha512(nupkg.read_bytes()).digest()).decode()
    assert lib["sha512"] == packed, f"restored {pid} differs from the packed .nupkg"
print("ok:   every restored Starlings.Roost.* package is the packed .nupkg (sha512)")

# Decision 30: no worker-side package may reach ONNX Runtime, directly or transitively.
worker_side = re.compile(r"^Starlings\.Roost\.(Abstractions|Ops|Graph|Worker|WorkerHost|Kernels|Cuda|Native|Distributed|Models\..+)$")
onnx = re.compile(r"^microsoft\.ml\.onnxruntime", re.IGNORECASE)
(target,) = assets["targets"].values()
graph = {key.split("/")[0].lower(): [d.lower() for d in entry.get("dependencies", {})] for key, entry in target.items()}
names = {key.split("/")[0].lower(): key.split("/")[0] for key in target}
bad = []
for key in target:
    pid = key.split("/")[0]
    if not worker_side.match(pid):
        continue
    seen, stack = set(), [pid.lower()]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph.get(cur, []))
    bad += [f"{pid} reaches {names.get(n, n)}" for n in sorted(seen) if onnx.match(n)]
assert not bad, "worker-side package closure reaches ONNX Runtime: " + "; ".join(bad)
print("ok:   no worker-side package reaches Microsoft.ML.OnnxRuntime* (decision 30)")
EOF
(cd "$consumer" && "$dotnet" build --configuration Release --no-restore "${msbuild_extra[@]}") >"$work/consumer-build.log" 2>&1 \
    || { tail -40 "$work/consumer-build.log"; fail "consumer build failed"; }
if grep -rIl --exclude='*.log' -F "$root" "$consumer"; then
    fail "consumer files mention the repository path"
fi
echo "ok:   consumer compiled against the packages; no consumer file names the repository path"
(cd "$consumer" && "$dotnet" run --configuration Release --no-build) | tee "$work/consumer.log" || fail "consumer run failed"
grep -qx 'RESULT: PASS' "$work/consumer.log" || fail "consumer did not pass"

mkdir -p "$out"
rm -f "$out"/*.nupkg "$out"/*.snupkg
cp "$pack"/*.nupkg "$out/"
cp "$pack"/*.snupkg "$out/" 2>/dev/null || true
cp "$work/nuspec-check.log" "$work/consumer.log" "$work/consumer-packages.txt" "$out/"
[[ -f "$work/kernel-digests-check.log" ]] && cp "$work/kernel-digests-check.log" "$out/"
{
    echo "commit $sha"
    echo "version $version"
    echo "kernels ${kernels}"
    for id in "${!packed[@]}"; do echo "package $id ${packed[$id]}"; done | sort
    (cd "$out" && sha256sum ./*.nupkg ./*.snupkg 2>/dev/null) | sed 's|  \./|  |'
} >"$out/drill.txt"
echo "RESULT: PASS" >>"$out/drill.txt"
step "DRILL PASS: Roost $version from $sha (packages and logs in $out)"
echo "RESULT: PASS"

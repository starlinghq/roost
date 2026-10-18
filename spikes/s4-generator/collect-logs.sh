#!/usr/bin/env bash
# Re-run every measurement of the S4 report and save the raw output under results/. CPU only, two build jobs.
# Usage: spikes/s4-generator/collect-logs.sh   (run from anywhere; each command's output goes to results/<name>.log)
set -uo pipefail
cd "$(dirname "$0")"
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2 CARGO_BUILD_JOBS=2
mkdir -p results
{ echo "date: $(date -u +%FT%TZ)"; echo "uptime: $(uptime)"; echo "nproc: $(nproc)"; dotnet --version; git rev-parse HEAD; } > results/environment.log 2>&1
(cd Toy && dotnet build -m:2 -nologo) > results/toy-build-debug.log 2>&1
(cd Toy && dotnet ../out/Toy/bin/Debug/net10.0/Toy.dll) > results/toy-jit-default.log 2>&1
(cd Toy && DOTNET_TieredCompilation=0 dotnet ../out/Toy/bin/Debug/net10.0/Toy.dll) > results/toy-jit-tc0.log 2>&1
(cd Toy && dotnet publish -c Release -r linux-x64 -m:2 -nologo -o ../out/Toy/publish) > results/toy-publish-aot.log 2>&1
(cd Toy && ../out/Toy/publish/Toy) > results/toy-aot.log 2>&1
ls -l out/Toy/publish/Toy > results/toy-aot-size.log 2>&1
(cd MissingOp && dotnet build -m:2 -nologo) > results/missing-op.log 2>&1
(cd MissingMember && dotnet build -m:2 -nologo) > results/missing-member.log 2>&1
(cd IncrementalCheck && dotnet build -m:2 -nologo && dotnet ../out/IncrementalCheck/bin/Debug/net10.0/IncrementalCheck.dll) > results/incremental.log 2>&1
./measure-build.sh 5 > results/measure-build.log 2>&1

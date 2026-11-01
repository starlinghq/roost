# Build-only CI step for MSVC (R-114): compile the kernel library for every SM in kernels/manifest.toml with cl.exe as
# nvcc's host compiler, load each DLL and check its registration table (stub_check --no-gpu), and write the per-SM
# SHA-256 digests as JSON. Needs no GPU. Run it from a shell where vcvars64.bat has been called and the CUDA toolkit is
# installed (.github/workflows/ci.yml, job kernels-msvc). Binaries are never uploaded or committed.
#
#   pwsh kernels/ci/build-all-sm-msvc.ps1 <work-dir> [sm-list]     # sm-list defaults to the manifest's, as 80;86;...
param(
    [Parameter(Mandatory = $true)][string]$Work,
    [string]$Sms = ''
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Sms) {
    $Sms = (& python -I "$root\kernels\tools\manifest_cmake.py" "$root\kernels\manifest.toml" --sm-list).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'cannot read the manifest SM list' }
}
New-Item -ItemType Directory -Force -Path $Work | Out-Null
Write-Host "SM list: $Sms"
& nvcc --version
& cl 2>&1 | Select-Object -First 1

$build = Join-Path $Work 'build'
& cmake -S "$root\kernels" -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DROOST_SM_LIST=$Sms" "-DROOST_UPSTREAM_DIR=$Work\upstream"
if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
& cmake --build $build
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

$digests = [ordered]@{}
foreach ($sm in $Sms.Split(';')) {
    $dll = Join-Path $build "roost_kernels_sm$sm.dll"
    if (-not (Test-Path $dll)) { throw "missing $dll" }
    & (Join-Path $build 'roost_stub_check.exe') $dll --no-gpu
    if ($LASTEXITCODE -ne 0) { throw "registration check failed for sm_$sm" }
    $digests["sm$sm"] = (Get-FileHash -Algorithm SHA256 $dll).Hash.ToLowerInvariant()
}
$out = if ($env:ROOST_DIGESTS_OUT) { $env:ROOST_DIGESTS_OUT } else { Join-Path $Work 'digests-msvc.json' }
$digests | ConvertTo-Json | Set-Content -Encoding utf8 $out
Get-Content $out

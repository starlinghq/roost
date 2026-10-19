using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.Kernels;

namespace Starlings.Roost.Kernels.Tests;

internal static class TestManifest
{
    /// <summary>The repository root: the directory holding Roost.slnx.</summary>
    public static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Roost.slnx")))
        {
            dir = dir.Parent;
        }

        return dir?.FullName ?? throw new InvalidOperationException("Roost.slnx not found above " + AppContext.BaseDirectory);
    }

    public static KernelManifest Repository() => KernelManifest.Load(Path.Combine(RepoRoot(), "kernels", "manifest.toml"));

    /// <summary>A small manifest for the ABI the host is built against (<see cref="KernelAbi"/>) unless told otherwise: Residual (residual-stub) and Gemm (gemm-x) for the given SMs, with one digest per SM.</summary>
    public static KernelManifest Build(
        string sm86Status = "experimental",
        string digest86 = "00",
        long bytes86 = 0,
        bool gemmFor86 = false,
        int abiMinor = KernelAbi.Minor,
        int abiMajor = KernelAbi.Major,
        string cuda = "13.3",
        string? gemmDigest86 = null)
    {
        var gemmBinary86 = gemmFor86
            ? $"\n[[ops.variants.binaries]]\narch = 86\nsha256 = \"{gemmDigest86 ?? digest86}\"\nbytes = {bytes86}\n"
            : string.Empty;
        var toml = $$"""
            schema_version = 1
            contract_version = 1
            [abi]
            major = {{abiMajor}}
            minor = {{abiMinor}}
            [toolkit]
            cuda = "{{cuda}}"
            min_driver = "580"
            [[sm]]
            arch = 80
            status = "untested"
            [[sm]]
            arch = 86
            status = "{{sm86Status}}"
            [[sm]]
            arch = 90
            status = "unsupported"
            [[ops]]
            op = "Residual"
            [[ops.variants]]
            id = "residual-stub"
            flags = ["graph_capturable", "deterministic", "batch_invariant"]
            [ops.variants.constraints]
            sm_min = 80
            [[ops.variants.binaries]]
            arch = 86
            sha256 = "{{digest86}}"
            bytes = {{bytes86}}
            [[ops.variants.binaries]]
            arch = 80
            sha256 = "{{new string('a', 64)}}"
            [[ops]]
            op = "Gemm"
            [[ops.variants]]
            id = "gemm-x"
            flags = ["deterministic"]
            [ops.variants.constraints]
            sm_min = 89
            [[ops.variants.binaries]]
            arch = 89
            sha256 = "{{new string('b', 64)}}"
            {{gemmBinary86}}
            """;
        return KernelManifest.Parse(Encoding.UTF8.GetBytes(toml));
    }
}

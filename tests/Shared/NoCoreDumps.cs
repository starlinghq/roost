// Linked into every test project by Directory.Build.props (projects whose name ends in .Tests or .Tests.Cuda).
#pragma warning disable SYSLIB1054 // A blittable DllImport needs no unsafe code; LibraryImport would require AllowUnsafeBlocks in every test project.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Starlings.Roost.Testing;

/// <summary>
/// Marks every Roost test process, and every worker process a test re-enters through the same assembly, as non-dumpable on
/// Linux, so a process that dies by a signal leaves no core and no systemd-coredump record on a developer machine. The
/// process still dies and the test still fails. GitHub's runners (GITHUB_ACTIONS=true) keep normal crash behaviour, so real
/// fatal-signal tests run there unchanged.
/// </summary>
internal static class NoCoreDumps
{
    private const int PrSetDumpable = 4;

    /// <summary>Runs when the test assembly loads, before any test or re-entered worker code.</summary>
    [ModuleInitializer]
    internal static void Apply()
    {
        if (!OperatingSystem.IsLinux() || Environment.GetEnvironmentVariable("GITHUB_ACTIONS") == "true")
        {
            return;
        }

        _ = Prctl(PrSetDumpable, 0, 0, 0, 0);
    }

    [DllImport("libc", EntryPoint = "prctl")]
    private static extern int Prctl(int option, nuint arg2, nuint arg3, nuint arg4, nuint arg5);
}

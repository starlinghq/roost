#pragma warning disable SYSLIB1054 // Blittable DllImport; see tests/Shared/NoCoreDumps.cs.

using System.Runtime.InteropServices;

namespace Starlings.Roost.Tests;

/// <summary>Checks that tests/Shared/NoCoreDumps.cs made this test process non-dumpable (Linux, off GitHub's runners).</summary>
public sealed class NoCoreDumpsTests
{
    private const int PrGetDumpable = 3;

    [Fact]
    public void TestProcessIsNotDumpableOnALinuxDeveloperMachine()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "PR_SET_DUMPABLE is Linux only.");
        Assert.SkipWhen(
            (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") == "true" && Environment.GetEnvironmentVariable("RUNNER_ENVIRONMENT") == "github-hosted"),
            "GitHub's runners keep normal crash behaviour so real fatal-signal tests run there.");

        Assert.Equal(0, Prctl(PrGetDumpable, 0, 0, 0, 0));
    }

    [DllImport("libc", EntryPoint = "prctl")]
    private static extern int Prctl(int option, nuint arg2, nuint arg3, nuint arg4, nuint arg5);
}

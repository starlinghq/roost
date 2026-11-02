using System.Diagnostics;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary>
/// The official openai Python package against the same server: <c>tests/tools/api_compat/openai_compat.py</c>, run with
/// <c>uv</c> at the pinned package version. Opt-in (<c>ROOST_API_COMPAT_PYTHON=1</c>), because it needs <c>uv</c> and the
/// network to fetch the package; it is skipped, not passed, otherwise.
/// </summary>
public sealed class PythonClientCompatTests
{
    private const string OptIn = "ROOST_API_COMPAT_PYTHON";
    private const string PinnedOpenAi = "3.35.1"; // keep equal to PINNED_OPENAI in the script

    [Fact]
    public async Task TheOfficialPythonClientPassesTheScript()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable(OptIn) == "1", $"Set {OptIn}=1 to run the openai Python package (needs uv and the network).");

        var script = Path.Combine(RepositoryRoot(), "tests", "tools", "api_compat", "openai_compat.py");
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("Hello there.") });

        var start = new ProcessStartInfo("uv")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in new[] { "run", "--no-project", "--with", "openai==" + PinnedOpenAi, "--", "python", "-I", script, "--base-url", new Uri(rig.Endpoint, "v1").ToString() })
        {
            start.ArgumentList.Add(arg);
        }

        using var process = Process.Start(start)!;
        var stdout = process.StandardOutput.ReadToEndAsync(TestContext.Current.CancellationToken);
        var stderr = process.StandardError.ReadToEndAsync(TestContext.Current.CancellationToken);
        await process.WaitForExitAsync(TestContext.Current.CancellationToken);

        Assert.True(process.ExitCode == 0, $"exit {process.ExitCode}\n{await stdout}\n{await stderr}");
        Assert.Contains("every check passed", await stdout, StringComparison.Ordinal);
    }

    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Roost.slnx")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Roost.slnx not found above " + AppContext.BaseDirectory);
    }
}

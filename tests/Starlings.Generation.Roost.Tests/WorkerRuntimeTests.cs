using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests;

public sealed class WorkerRuntimeTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("roost-worker-runtime-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static RoostWorkerOptions Options(string executable) => new()
    {
        DeviceOrdinal = 1,
        Launch = new RoostWorkerLaunch
        {
            Executable = executable,
            WeightsPath = "/models/qwen/model.safetensors",
            WeightsSize = 1_234_567,
            KernelDirectory = "/opt/kernels",
            ManifestPath = "/opt/kernels/manifest.toml",
            AllowUntestedSm = true,
        },
    };

    private string FakeExecutable(string relative, UnixFileMode mode)
    {
        string path = Path.Combine(_dir, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "#!/bin/sh\n");
        if (!OperatingSystem.IsWindows())
        {
            File.SetUnixFileMode(path, mode);
        }

        return path;
    }

    [Fact]
    public void TheCommandCarriesTheProfilesCapacitiesAndTheLockedWeights()
    {
        var profile = TestRig.Profile() with { MaxKvBlocksPerStep = 12 };
        string exe = FakeExecutable("roost-worker", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);

        var command = RoostWorkerRuntime.CommandFor(profile, Options(exe));

        Assert.Equal(exe, command.FileName);
        var parsed = WorkerArguments.Parse(command.Arguments);
        Assert.Equal(profile.ModelId, parsed.Model);
        Assert.Equal("/models/qwen/model.safetensors", parsed.WeightsPath);
        Assert.Equal(1_234_567, parsed.WeightsSize);
        Assert.Equal(Convert.ToHexStringLower(profile.WeightsDigest.AsSpan()), parsed.WeightsSha256);
        Assert.Equal("/opt/kernels", parsed.KernelDirectory);
        Assert.Equal("/opt/kernels/manifest.toml", parsed.ManifestPath);
        Assert.True(parsed.AllowUntested);
        Assert.Equal(1, parsed.Device);
        Assert.Equal(profile.MaxTokensPerStep, parsed.MaxTokens);
        Assert.Equal(profile.MaxSequencesPerStep, parsed.MaxSequences);
        Assert.Equal(12, parsed.MaxKvBlocks);
        Assert.Equal(profile.AttentionBlocks, parsed.AttentionBlocks);
        Assert.Equal(profile.RecurrentSlots, parsed.RecurrentSlots);
    }

    [Fact]
    public void MaxKvBlocksDefaultsToTheAttentionPool()
    {
        var profile = TestRig.Profile() with { MaxKvBlocksPerStep = null };
        string exe = FakeExecutable("roost-worker", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);

        var parsed = WorkerArguments.Parse(RoostWorkerRuntime.CommandFor(profile, Options(exe)).Arguments);

        Assert.Equal(profile.AttentionBlocks, parsed.MaxKvBlocks);
    }

    [Fact]
    public void AMissingLaunchIsRefusedNamingBothOptions()
    {
        var e = Assert.Throws<ArgumentException>(() => RoostWorkerRuntime.CommandFor(TestRig.Profile(), new RoostWorkerOptions()));
        Assert.Contains("Launch", e.Message, StringComparison.Ordinal);
        Assert.Contains("Command", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingExecutableIsRefusedNamingThePath()
    {
        string missing = Path.Combine(_dir, "nowhere", "roost-worker");
        var e = Assert.Throws<FileNotFoundException>(() => RoostWorkerRuntime.CommandFor(TestRig.Profile(), Options(missing)));
        Assert.Contains(missing, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePackagedWorkerIsFoundBesideTheAppOrUnderRuntimes()
    {
        Assert.Null(RoostWorkerRuntime.FindExecutable(_dir));
        string file = OperatingSystem.IsWindows() ? "roost-worker.exe" : "roost-worker";
        string nested = FakeExecutable(Path.Combine("runtimes", RuntimeInformation.RuntimeIdentifier, "native", file), UnixFileMode.UserRead | UnixFileMode.UserExecute);
        Assert.Equal(nested, RoostWorkerRuntime.FindExecutable(_dir));

        string beside = FakeExecutable(file, UnixFileMode.UserRead | UnixFileMode.UserExecute);
        Assert.Equal(beside, RoostWorkerRuntime.FindExecutable(_dir));
    }

    [Fact]
    public void AnExecutableThatLostItsExecuteBitGetsItBackWhereReadIsGranted()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }

        string exe = FakeExecutable("roost-worker", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead);

        RoostWorkerRuntime.CommandFor(TestRig.Profile(), Options(exe));

        var mode = File.GetUnixFileMode(exe);
        Assert.True(mode.HasFlag(UnixFileMode.UserExecute));
        Assert.True(mode.HasFlag(UnixFileMode.GroupExecute));
        Assert.False(mode.HasFlag(UnixFileMode.OtherExecute));
    }

    [Fact]
    public void TheManifestDigestIsTheFilesSha256()
    {
        string manifest = Path.Combine(_dir, RoostWorkerRuntime.ManifestFileName);
        File.WriteAllText(manifest, "abi_major = 1\n");
        Assert.Equal(SHA256.HashData(File.ReadAllBytes(manifest)), RoostWorkerRuntime.ManifestDigest(manifest));
    }
}

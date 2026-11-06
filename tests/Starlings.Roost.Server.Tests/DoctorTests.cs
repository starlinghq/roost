using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.Server.Doctor;
using Starlings.Roost.Server.Tests.Support;

namespace Starlings.Roost.Server.Tests;

/// <summary><c>roost doctor</c> (R-046): each check's typed result over fake probes, the command line, and the real probes that need no GPU.</summary>
public sealed class DoctorTests
{
    private static DoctorCommand Full(FakeDoctorProbes probes) => new()
    {
        Weights = FakeDoctorProbes.WeightsPath,
        WeightsSha256 = FakeDoctorProbes.WeightsSha256,
        Tokenizer = "tokenizer.json",
        Worker = probes.Worker,
    };

    [Fact]
    public void AHealthyMachinePassesEveryCheckInOrder()
    {
        var probes = new FakeDoctorProbes();

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorChecks.All, report.Checks.Select(c => c.Name));
        Assert.All(report.Checks, c => Assert.Equal(DoctorStatus.Pass, c.Status));
        Assert.True(report.Healthy);
        Assert.Contains("sha256 " + FakeDoctorProbes.Sha(probes.Files[probes.ManifestPath]), report[DoctorChecks.Manifest].Detail, StringComparison.Ordinal);
        Assert.Contains("SM 90 supported", report[DoctorChecks.Sm].Detail, StringComparison.Ordinal);
        Assert.Contains("matches the manifest", report[DoctorChecks.Kernels].Detail, StringComparison.Ordinal);
        Assert.Equal(["tokenizer.json: <|im_end|> <|endoftext|>"], probes.TokenizerChecks);
    }

    [Fact]
    public void InputsNotGivenAreSkippedNotPassed()
    {
        var probes = new FakeDoctorProbes();

        var report = RoostDoctor.Run(new DoctorCommand(), probes);

        Assert.Equal(DoctorStatus.Skipped, report[DoctorChecks.Weights].Status);
        Assert.Equal(DoctorStatus.Skipped, report[DoctorChecks.Tokenizer].Status);
        Assert.Equal(DoctorStatus.Pass, report[DoctorChecks.Manifest].Status); // found beside the packaged worker
        Assert.True(report.Healthy);
    }

    [Fact]
    public void NoDriverFails()
    {
        var probes = new FakeDoctorProbes { Driver = null };

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Driver].Status);
        Assert.False(report.Healthy);
    }

    [Theory]
    [InlineData("570.124.06", "fail")]
    [InlineData("580.65.06", "pass")]
    [InlineData("610.57", "pass")]
    [InlineData("unknown", "warn")]
    public void TheDriverIsComparedWithTheManifestsMinimum(string driver, string expected)
    {
        var probes = new FakeDoctorProbes { Driver = driver };

        var check = RoostDoctor.Run(Full(probes), probes)[DoctorChecks.Driver];

        Assert.Equal(expected, DoctorReport.Label(check.Status));
        Assert.Contains(driver, check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingNvmlIsAWarning()
    {
        var probes = new FakeDoctorProbes { Nvml = null };

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Warn, report[DoctorChecks.Nvml].Status);
        Assert.True(report.Healthy);
    }

    [Fact]
    public void NoVisibleGpuFailsAndSkipsTheSmCheck()
    {
        var probes = new FakeDoctorProbes { Gpus = new GpuListing([], "nvidia-smi could not run (No such file or directory)") };

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Gpu].Status);
        Assert.Contains("nvidia-smi could not run", report[DoctorChecks.Gpu].Detail, StringComparison.Ordinal);
        Assert.Equal(DoctorStatus.Skipped, report[DoctorChecks.Sm].Status);
    }

    [Theory]
    [InlineData(90, false, "pass")]
    [InlineData(86, false, "warn")] // experimental: ran on hardware, not admitted
    [InlineData(89, false, "fail")] // untested needs the opt-in
    [InlineData(89, true, "warn")]
    [InlineData(75, true, "fail")] // not in the manifest
    public void TheSmStatusComesFromTheManifest(int sm, bool allowUntested, string expected)
    {
        var probes = new FakeDoctorProbes { Gpus = new GpuListing([new DoctorGpu(0, "GPU", sm, null)]) };

        var check = RoostDoctor.Run(Full(probes) with { AllowUntestedSm = allowUntested }, probes)[DoctorChecks.Sm];

        Assert.Equal(expected, DoctorReport.Label(check.Status));
        Assert.Contains($"SM {sm}", check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void TheWorstSmOfSeveralGpusDecides()
    {
        var probes = new FakeDoctorProbes { Gpus = new GpuListing([new DoctorGpu(0, "A", 90, null), new DoctorGpu(1, "B", 86, null)]) };

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Warn, report[DoctorChecks.Sm].Status);
        Assert.Equal(DoctorStatus.Pass, report[DoctorChecks.Kernels].Status); // both libraries are present and match
    }

    [Fact]
    public void ALibraryWhoseDigestDiffersFromTheManifestFails()
    {
        var probes = new FakeDoctorProbes();
        probes.Files[probes.LibraryPath(90)] = Encoding.UTF8.GetBytes("another build");

        var check = RoostDoctor.Run(Full(probes), probes)[DoctorChecks.Kernels];

        Assert.Equal(DoctorStatus.Fail, check.Status);
        Assert.Contains("the manifest pins " + FakeDoctorProbes.Sha(FakeDoctorProbes.Library90), check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingLibraryForTheGpusSmFails()
    {
        var probes = new FakeDoctorProbes();
        probes.Files.Remove(probes.LibraryPath(90));

        var check = RoostDoctor.Run(Full(probes), probes)[DoctorChecks.Kernels];

        Assert.Equal(DoctorStatus.Fail, check.Status);
        Assert.Contains("is missing", check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void AnSmWithoutAManifestDigestFails()
    {
        var probes = new FakeDoctorProbes { Gpus = new GpuListing([new DoctorGpu(0, "GPU", 89, null)]) };
        probes.Files[probes.LibraryPath(89)] = Encoding.UTF8.GetBytes("unrecorded");

        var check = RoostDoctor.Run(Full(probes) with { AllowUntestedSm = true }, probes)[DoctorChecks.Kernels];

        Assert.Equal(DoctorStatus.Fail, check.Status);
        Assert.Contains("records no library digest", check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void AKernelDirectoryWithoutLibrariesFails()
    {
        var probes = new FakeDoctorProbes();
        string empty = Path.GetFullPath("empty-kernels");

        var check = RoostDoctor.Run(Full(probes) with { KernelDirectory = empty }, probes)[DoctorChecks.Kernels];

        Assert.Equal(DoctorStatus.Fail, check.Status);
        Assert.Contains("no kernel library", check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void NoWorkerAndNoManifestFailsTheManifestAndKernelChecks()
    {
        var probes = new FakeDoctorProbes { HasPackagedWorker = false };

        var report = RoostDoctor.Run(new DoctorCommand(), probes);

        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Manifest].Status);
        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Kernels].Status);
        Assert.Equal(DoctorStatus.Skipped, report[DoctorChecks.Sm].Status);
        Assert.Contains("min", report[DoctorChecks.Driver].Detail, StringComparison.Ordinal); // not compared without a manifest
    }

    [Fact]
    public void AMalformedManifestFails()
    {
        var probes = new FakeDoctorProbes();
        probes.Files[probes.ManifestPath] = Encoding.UTF8.GetBytes("schema_version = 1\n[abi]\nmajor = 1\n");

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Manifest].Status);
        Assert.Contains("C5 schema", report[DoctorChecks.Manifest].Detail, StringComparison.Ordinal);
        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Kernels].Status);
    }

    [Fact]
    public void AManifestAtANewerMinorPassesAsItDoesInTheLoader()
    {
        var probes = new FakeDoctorProbes();
        probes.Files[probes.ManifestPath] = Encoding.UTF8.GetBytes(FakeDoctorProbes.Manifest(
            new Dictionary<int, string> { [90] = "supported" }, new Dictionary<int, byte[]> { [90] = FakeDoctorProbes.Library90 }, abiMajor: KernelAbi.Major, abiMinor: KernelAbi.Minor + 1));

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Pass, report[DoctorChecks.Manifest].Status);
    }

    [Theory]
    [InlineData(KernelAbi.Major, KernelAbi.Minor - 1)] // the case of #155: doctor passed, then roost serve failed
    [InlineData(KernelAbi.Major + 1, KernelAbi.Minor)]
    public void AManifestTheLoaderWouldRefuseFailsTheCheck(int major, int minor)
    {
        var probes = new FakeDoctorProbes();
        probes.Files[probes.ManifestPath] = Encoding.UTF8.GetBytes(FakeDoctorProbes.Manifest(
            new Dictionary<int, string> { [90] = "supported" }, new Dictionary<int, byte[]> { [90] = FakeDoctorProbes.Library90 }, abiMajor: major, abiMinor: minor));

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Manifest].Status);
        Assert.Contains($"ABI {major}.{minor}", report[DoctorChecks.Manifest].Detail, StringComparison.Ordinal);
        Assert.Contains($"needs ABI {KernelAbi.Major}.{KernelAbi.Minor}", report[DoctorChecks.Manifest].Detail, StringComparison.Ordinal);
        Assert.False(report.Healthy);
        Assert.Equal(DoctorStatus.Pass, report[DoctorChecks.Kernels].Status); // the digests are still reported
    }

    [Fact]
    public void AnExplicitManifestIsUsedInsteadOfTheOneBesideTheWorker()
    {
        var probes = new FakeDoctorProbes();
        string other = Path.GetFullPath("other.manifest.toml");
        probes.Files[other] = Encoding.UTF8.GetBytes(FakeDoctorProbes.Manifest(new Dictionary<int, string> { [90] = "untested" }, new Dictionary<int, byte[]> { [90] = FakeDoctorProbes.Library90 }));

        var report = RoostDoctor.Run(Full(probes) with { Manifest = other }, probes);

        Assert.Contains(other, report[DoctorChecks.Manifest].Detail, StringComparison.Ordinal);
        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Sm].Status);
    }

    [Fact]
    public void WeightsThatDoNotMatchTheLockFail()
    {
        var probes = new FakeDoctorProbes();
        string wrong = new('0', 64);

        var check = RoostDoctor.Run(Full(probes) with { WeightsSha256 = wrong }, probes)[DoctorChecks.Weights];

        Assert.Equal(DoctorStatus.Fail, check.Status);
        Assert.Contains("the lock pins " + wrong, check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingWeightsFail()
    {
        var probes = new FakeDoctorProbes();
        probes.Files.Remove(FakeDoctorProbes.WeightsPath);

        Assert.Equal(DoctorStatus.Fail, RoostDoctor.Run(Full(probes), probes)[DoctorChecks.Weights].Status);
    }

    [Fact]
    public void ATokenizerProblemFails()
    {
        var probes = new FakeDoctorProbes { TokenizerProblem = "tokenizer.json lacks the profile's stop token(s) <|im_end|>" };

        var check = RoostDoctor.Run(Full(probes), probes)[DoctorChecks.Tokenizer];

        Assert.Equal(DoctorStatus.Fail, check.Status);
        Assert.Equal(probes.TokenizerProblem, check.Detail);
    }

    [Theory]
    [InlineData(1L << 30, "warn")]
    [InlineData(3L << 30, "pass")]
    [InlineData(null, "warn")]
    public void LowOrUnreadableDiskAndMemoryAreWarnings(long? bytes, string expected)
    {
        var probes = new FakeDoctorProbes { FreeDisk = bytes, AvailableMemory = bytes };

        var report = RoostDoctor.Run(Full(probes), probes);

        Assert.Equal(expected, DoctorReport.Label(report[DoctorChecks.Disk].Status));
        Assert.Equal(expected, DoctorReport.Label(report[DoctorChecks.Memory].Status));
        Assert.True(report.Healthy);
    }

    [Fact]
    public void TheFloorsAreConfigurable()
    {
        var probes = new FakeDoctorProbes { FreeDisk = 50L << 30 };

        var check = RoostDoctor.Run(Full(probes) with { MinFreeDiskGib = 64 }, probes)[DoctorChecks.Disk];

        Assert.Equal(DoctorStatus.Warn, check.Status);
        Assert.Contains("floor 64 GiB", check.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public void NoGpuSkipsTheGpuChecksAndChecksEveryLibraryPresent()
    {
        var probes = new FakeDoctorProbes { Driver = null, Nvml = null, Gpus = new GpuListing([], "no GPU in an image build") };

        var report = RoostDoctor.Run(Full(probes) with { NoGpu = true }, probes);

        foreach (var name in new[] { DoctorChecks.Driver, DoctorChecks.Nvml, DoctorChecks.Gpu, DoctorChecks.Sm })
        {
            Assert.Equal(DoctorStatus.Skipped, report[name].Status);
        }

        var kernels = report[DoctorChecks.Kernels];
        Assert.Equal(DoctorStatus.Pass, kernels.Status);
        Assert.Contains("SM 86", kernels.Detail, StringComparison.Ordinal);
        Assert.Contains("SM 90", kernels.Detail, StringComparison.Ordinal);
        Assert.True(report.Healthy);
    }

    [Fact]
    public void NoGpuStillFailsAMismatchedLibrary()
    {
        var probes = new FakeDoctorProbes();
        probes.Files[probes.LibraryPath(86)] = Encoding.UTF8.GetBytes("tampered");

        var report = RoostDoctor.Run(Full(probes) with { NoGpu = true }, probes);

        Assert.Equal(DoctorStatus.Fail, report[DoctorChecks.Kernels].Status);
        Assert.False(report.Healthy);
    }

    [Fact]
    public void TheReportPrintsOneLinePerCheckThenASummary()
    {
        var probes = new FakeDoctorProbes { Nvml = null };
        using var output = new StringWriter();

        RoostDoctor.Run(Full(probes), probes).WriteTo(output);

        var lines = output.ToString().TrimEnd().Split(Environment.NewLine);
        Assert.Equal(DoctorChecks.All.Count + 1, lines.Length);
        Assert.StartsWith("[pass] driver: driver 610.57", lines[0], StringComparison.Ordinal);
        Assert.StartsWith("[warn] nvml: ", lines[1], StringComparison.Ordinal);
        Assert.Equal("roost doctor: no check failed (1 warning(s)).", lines[^1]);
    }

    [Fact]
    public async Task TheCommandExitsZeroWhenNothingFailsAndOneWhenACheckFails()
    {
        var healthy = new FakeDoctorProbes();
        using var output = new StringWriter();
        using var error = new StringWriter();

        Assert.Equal(0, await RoostCli.RunAsync(["doctor", "--worker", healthy.Worker], output, error, healthy));
        Assert.Contains("[pass] kernels: ", output.ToString(), StringComparison.Ordinal);
        Assert.Empty(error.ToString());

        var broken = new FakeDoctorProbes { Driver = null };
        using var output2 = new StringWriter();
        Assert.Equal(1, await RoostCli.RunAsync(["doctor"], output2, error, broken));
        Assert.Contains("[fail] driver: ", output2.ToString(), StringComparison.Ordinal);
        Assert.Contains("roost doctor: 1 check(s) failed", output2.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheCommandReportsAUsageErrorWithExitTwo()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();

        Assert.Equal(2, await RoostCli.RunAsync(["doctor", "--gpu-memory-utilization", "0.9"], output, error, new FakeDoctorProbes()));
        Assert.Contains("unknown option '--gpu-memory-utilization'", error.ToString(), StringComparison.Ordinal);
        Assert.Empty(output.ToString());
    }

    [Fact]
    public void DoctorParsesItsFlags()
    {
        string sha = new('a', 64);

        var c = RoostCli.ParseDoctor(["doctor", "--weights", "w.safetensors", "--weights-sha256", sha, "--tokenizer", "t.json", "--kernels", "k", "--manifest", "m.toml",
            "--allow-untested-sm", "--no-gpu", "--min-free-disk-gib", "20", "--min-free-memory-gib", "0.5"]);

        Assert.Equal(RoostCli.Qwen35Profile, c.Model);
        Assert.Equal("w.safetensors", c.Weights);
        Assert.Equal(sha, c.WeightsSha256);
        Assert.Equal("t.json", c.Tokenizer);
        Assert.Equal("k", c.KernelDirectory);
        Assert.Equal("m.toml", c.Manifest);
        Assert.Null(c.Worker);
        Assert.True(c.AllowUntestedSm);
        Assert.True(c.NoGpu);
        Assert.Equal(20, c.MinFreeDiskGib);
        Assert.Equal(0.5, c.MinFreeMemoryGib);
    }

    [Theory]
    [InlineData(new[] { "doctor", "--weights", "w" }, "--weights and --weights-sha256 go together")]
    [InlineData(new[] { "doctor", "--weights-sha256", "abc", "--weights", "w" }, "--weights-sha256 must be 64 hex digits")]
    [InlineData(new[] { "doctor", "--model", "llama" }, "unknown model profile 'llama'")]
    [InlineData(new[] { "doctor", "--min-free-disk-gib", "-1" }, "is not a non-negative number")]
    [InlineData(new[] { "doctor", "--min-free-disk-gib", "lots" }, "is not a non-negative number")]
    [InlineData(new[] { "doctor", "--host", "0.0.0.0" }, "unknown option '--host'")]
    public void DoctorRefusesBadArguments(string[] args, string message)
    {
        var e = Assert.Throws<ArgumentException>(() => RoostCli.ParseDoctor(args));
        Assert.Contains(message, e.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("8.6", 86)]
    [InlineData("12.0", 120)]
    [InlineData("10.0", 100)]
    [InlineData("8", null)]
    [InlineData("8.10", null)]
    [InlineData("[N/A]", null)]
    public void NvidiaSmiComputeCapabilityIsReadAsAnSm(string text, int? sm) =>
        Assert.Equal(sm, SystemDoctorProbes.ParseComputeCapability(text));

    [Fact]
    public void TheCommittedKernelManifestParsesWithTheSameSmStatuses()
    {
        var bytes = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "manifest.toml"));

        var manifest = DoctorManifest.Parse(bytes);

        Assert.Equal(FakeDoctorProbes.Sha(bytes), manifest.Sha256);
        Assert.Equal(KernelAbi.Major, manifest.AbiMajor);
        Assert.Equal(KernelAbi.Minor, manifest.AbiMinor); // a manifest behind the loader fails roost doctor and roost serve
        Assert.Equal("580", manifest.MinDriver);
        Assert.Equal([80, 86, 89, 90, 100, 120], manifest.SmStatuses.Keys);
        Assert.Equal("experimental", manifest.SmStatuses[86]);
        Assert.Equal("experimental", manifest.SmStatuses[89]);
        Assert.DoesNotContain("supported", manifest.SmStatuses.Values); // nothing is supported before R-026
        Assert.All(manifest.LibraryDigests.Values, digests => Assert.Single(digests)); // one library per SM holds every variant
    }

    [Fact]
    public void TheRealTokenizerProbeNamesMissingStopTokens()
    {
        // The tiny fixture holds <|im_end|> but not <|endoftext|>, the profile's second stop token.
        string tokenizer = Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json");
        var probes = new SystemDoctorProbes();

        Assert.Null(probes.CheckTokenizer(tokenizer, ["<|im_end|>"]));
        var problem = probes.CheckTokenizer(tokenizer, ["<|im_end|>", "<|endoftext|>"]);
        Assert.NotNull(problem);
        Assert.EndsWith("lacks the profile's stop token(s) <|endoftext|>", problem, StringComparison.Ordinal);
        Assert.Contains("does not exist", probes.CheckTokenizer(Path.Combine(AppContext.BaseDirectory, "missing.json"), []), StringComparison.Ordinal);
    }

    [Fact]
    public void TheRealFileProbesHashAndListKernelLibraries()
    {
        string directory = Directory.CreateTempSubdirectory("roost-doctor-").FullName;
        try
        {
            string library = Path.Combine(directory, "libroost_kernels_sm86.so");
            File.WriteAllBytes(library, FakeDoctorProbes.Library86);
            File.WriteAllText(Path.Combine(directory, "unrelated.so"), "x");
            var probes = new SystemDoctorProbes();

            Assert.Equal([library], probes.KernelLibraries(directory));
            Assert.Equal((FakeDoctorProbes.Sha(FakeDoctorProbes.Library86), (long)FakeDoctorProbes.Library86.Length), probes.HashFile(library));
            Assert.Null(probes.HashFile(Path.Combine(directory, "missing.so")));
            Assert.Empty(probes.KernelLibraries(Path.Combine(directory, "missing")));
            Assert.NotNull(probes.FreeDiskBytes(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}

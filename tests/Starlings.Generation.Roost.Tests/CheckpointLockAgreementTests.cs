using Starlings.Models.Qwen35;
using Starlings.Roost.Worker.Weights;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// Roost's lock for Qwen3.5-0.8B and the checkpoint lock Starling publishes describe one checkpoint (#144). Roost's stays
/// the registry's record: it covers the checkpoint's other files, and the admission records, fixture proposals and GPU
/// harnesses use it. This test makes any disagreement between the two a failure.
/// </summary>
public sealed class CheckpointLockAgreementTests
{
    [Fact]
    public void StarlingsLockAgreesWithRoostsOnEveryFileItCovers()
    {
        var roost = WeightLock.Load(Path.Combine(RepoRoot(), "evidence", "registry", "qwen3.5-0.8b.lock.toml"));
        var starling = Qwen35Checkpoint.Lock;

        Assert.Equal(starling.Repository, roost.Repo);
        Assert.Equal(starling.Revision, roost.Revision);
        Assert.NotEmpty(starling.Files);
        var byPath = roost.Files.ToDictionary(f => f.Path, StringComparer.Ordinal);
        foreach (var file in starling.Files)
        {
            Assert.True(byPath.TryGetValue(file.Path, out var locked), $"Roost's lock has no {file.Path}");
            Assert.Equal(file.Size, locked.Size);
            Assert.Equal(file.Sha256, locked.Sha256, ignoreCase: true);
        }

        Assert.Contains(starling.Files, f => f.Path.EndsWith(".safetensors", StringComparison.Ordinal));
    }

    private static string RepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Roost.slnx")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Roost.slnx not found above the test directory.");
    }
}

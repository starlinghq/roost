using System.Reflection;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.WorkerHost;
using Xunit.v3;

[assembly: Xunit.AssemblyFixture(typeof(Starlings.Roost.WorkerHost.Tests.Support.SegmentRunGuard))]
[assembly: Starlings.Roost.WorkerHost.Tests.Support.SweepSegmentsAfterTest]

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>
/// Where every segment of this test run lives: one directory, <c>roost-worker-test-&lt;run id&gt;</c>, under the default
/// segment directory (<c>/dev/shm</c> on Linux). Tests never use the shared default, so a segment a dead worker or a
/// failed test leaves behind is findable by the prefix, removed after the test, and counted at the end of the run.
/// </summary>
internal static class TestSegments
{
    public const string Prefix = "roost-worker-test-";

    private static readonly Lazy<string> _root = new(() =>
    {
        var path = Path.Combine(RingSegment.DefaultDirectory(), Prefix + Guid.NewGuid().ToString("N"));
        _ = Directory.CreateDirectory(path);
        return path;
    });

    public static bool Created => _root.IsValueCreated;

    public static string Root => _root.Value;

    /// <summary>Options that put the transport's segments under this run's prefix.</summary>
    public static WorkerTransportOptions Options(WorkerTransportOptions? options = null) =>
        (options ?? new WorkerTransportOptions()) with { SegmentDirectory = Root };

    /// <summary>A new empty directory under the run's root, for a test that inspects its own segments.</summary>
    public static string NewDirectory()
    {
        var path = Path.Combine(Root, Guid.NewGuid().ToString("N"));
        _ = Directory.CreateDirectory(path);
        return path;
    }

    /// <summary>Deletes everything under the run's root, whether or not its worker is alive; returns the names removed.</summary>
    public static string[] Sweep()
    {
        if (!Created || !Directory.Exists(Root))
        {
            return [];
        }

        var removed = new List<string>();
        foreach (var entry in Directory.GetFileSystemEntries(Root))
        {
            removed.Add(Path.GetFileName(entry));
            try
            {
                if (Directory.Exists(entry))
                {
                    Directory.Delete(entry, recursive: true);
                }
                else
                {
                    File.Delete(entry);
                }
            }
            catch (IOException)
            {
                // A file still being removed by another process; the end-of-run check reports what is left.
            }
        }

        return [.. removed];
    }
}

/// <summary>Teardown after every test: removes the segments the test left, also when its worker died.</summary>
[AttributeUsage(AttributeTargets.Assembly)]
internal sealed class SweepSegmentsAfterTestAttribute : BeforeAfterTestAttribute
{
    public override void After(MethodInfo methodUnderTest, IXunitTest test) => _ = TestSegments.Sweep();
}

/// <summary>
/// The end of the run: nothing with this run's prefix may remain. The per-test teardown has already removed what tests
/// left, so anything found here was created late, for example by a worker that outlived its test. It is removed and
/// the run fails.
/// </summary>
public sealed class SegmentRunGuard : IDisposable
{
    public void Dispose()
    {
        if (!TestSegments.Created)
        {
            return;
        }

        var left = Directory.Exists(TestSegments.Root) ? Directory.GetFileSystemEntries(TestSegments.Root) : [];
        _ = TestSegments.Sweep();
        if (Directory.Exists(TestSegments.Root))
        {
            Directory.Delete(TestSegments.Root, recursive: true);
        }

        Assert.True(left.Length == 0, $"segments with the prefix {TestSegments.Prefix} remain at the end of the run: {string.Join(", ", left)}");
    }
}

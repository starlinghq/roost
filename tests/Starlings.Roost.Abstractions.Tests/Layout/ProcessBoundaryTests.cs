using System.Reflection;
using Starlings.Roost.Abi;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Tests.Layout;

/// <summary>
/// C2 and the C8 segment records live in shared memory mapped at different addresses in the host and the worker,
/// so no field may be a host or device pointer (README.md, Contracts, check 2). DeviceSpan is worker-side only.
/// </summary>
public sealed class ProcessBoundaryTests
{
    public static TheoryData<string> SharedMemoryTypeNames() =>
        [.. NativeLayout.MirroredTypes
            .Where(t => t.Namespace is "Starlings.Roost.Plans" or "Starlings.Roost.Transport")
            .Select(t => t.FullName!)];

    [Fact]
    public void EveryRecordInTheSharedHeaderIsInAPlansOrTransportNamespace()
    {
        var shared = NativeLayout.MirroredTypes
            .Where(t => t.Namespace is "Starlings.Roost.Plans" or "Starlings.Roost.Transport")
            .Select(NativeLayout.NativeName)
            .Order(StringComparer.Ordinal);
        var fromHeader = NativeLayout.Golden.Keys
            .Where(IsSharedHeaderStruct)
            .Order(StringComparer.Ordinal);
        Assert.Equal(fromHeader, shared);
    }

    [Theory]
    [MemberData(nameof(SharedMemoryTypeNames))]
    public void HasNoPointerOrDeviceSpanField(string typeName)
    {
        var type = NativeLayout.MirroredTypes.Single(t => t.FullName == typeName);
        var offending = new List<string>();
        Walk(type, type.Name, offending);
        Assert.Empty(offending);
    }

    [Fact]
    public void TheCheckRejectsAPointer()
    {
        var offending = new List<string>();
        Walk(typeof(WithPointer), nameof(WithPointer), offending);
        Walk(typeof(WithSpan), nameof(WithSpan), offending);
        Assert.Equal(2, offending.Count);
    }

    private static void Walk(Type type, string path, List<string> offending)
    {
        foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
        {
            var t = field.FieldType;
            var where = $"{path}.{field.Name}";
            if (t == typeof(nint) || t == typeof(nuint) || t.IsPointer || t.IsFunctionPointer || t == typeof(DeviceSpan))
            {
                offending.Add(where);
            }
            else if (t.IsValueType && !t.IsPrimitive && !t.IsEnum)
            {
                Walk(t, where, offending);
            }
        }
    }

    private static bool IsSharedHeaderStruct(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Layout", "include", "roost_worker_shm.h"))
            .Contains($"typedef struct {name} {{", StringComparison.Ordinal);

#pragma warning disable CS0649 // Never assigned; only the field types matter.
    private struct WithPointer
    {
        public nint Address;
    }

    private struct WithSpan
    {
        public DeviceSpan Span;
    }
#pragma warning restore CS0649
}

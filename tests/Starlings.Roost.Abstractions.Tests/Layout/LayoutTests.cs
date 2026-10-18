using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Starlings.Roost.Tests.Layout;

/// <summary>
/// Every C# mirror of a C contract struct has the gcc layout recorded in the golden file, field for field, and is
/// blittable. Runs on every OS without a C compiler; CI's abi job regenerates the golden file with gcc.
/// </summary>
public sealed class LayoutTests
{
    public static TheoryData<string> MirroredTypeNames() =>
        [.. NativeLayout.MirroredTypes.Select(t => t.FullName!)];

    private static Type Mirrored(string fullName) => NativeLayout.MirroredTypes.Single(t => t.FullName == fullName);

    [Fact]
    public void EveryGoldenStructHasExactlyOneMirror()
    {
        var mirrored = NativeLayout.MirroredTypes.Select(NativeLayout.NativeName).ToList();
        Assert.Equal(mirrored.Count, mirrored.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(
            NativeLayout.Golden.Keys.Order(StringComparer.Ordinal),
            mirrored.Order(StringComparer.Ordinal));
    }

    [Theory]
    [MemberData(nameof(MirroredTypeNames))]
    public void SizeMatchesGolden(string typeName)
    {
        var type = Mirrored(typeName);
        var native = NativeLayout.Golden[NativeLayout.NativeName(type)];
        Assert.Equal(native.Size, Marshal.SizeOf(type));
        Assert.Equal(native.Size, UnsafeSizeOf(type));
    }

    [Theory]
    [MemberData(nameof(MirroredTypeNames))]
    public void FieldsMatchGoldenInOrder(string typeName)
    {
        var type = Mirrored(typeName);
        var native = NativeLayout.Golden[NativeLayout.NativeName(type)];
        var fields = InstanceFields(type);
        Assert.Equal(native.Fields.Select(f => NativeLayout.ToPascal(f.Name)), fields.Select(f => f.Name));
        foreach (var (field, expected) in fields.Zip(native.Fields))
        {
            Assert.True(
                expected.Offset == (int)Marshal.OffsetOf(type, field.Name),
                $"{type.Name}.{field.Name}: offset {Marshal.OffsetOf(type, field.Name)}, gcc {expected.Offset}");
            Assert.True(
                expected.Size == FieldSize(field.FieldType),
                $"{type.Name}.{field.Name}: size {FieldSize(field.FieldType)}, gcc {expected.Size}");
        }
    }

    [Theory]
    [MemberData(nameof(MirroredTypeNames))]
    public void IsBlittable(string typeName)
    {
        var type = Mirrored(typeName);
        Assert.True(type.IsValueType);
        Assert.True(type.IsLayoutSequential);
        Assert.False(IsReferenceOrContainsReferences(type));
        Assert.Equal(Marshal.SizeOf(type), UnsafeSizeOf(type));
        Assert.All(InstanceFields(type), f => Assert.NotEqual(typeof(bool), f.FieldType));
        Assert.All(InstanceFields(type), f => Assert.NotEqual(typeof(char), f.FieldType));
    }

    internal static FieldInfo[] InstanceFields(Type type) =>
        [.. type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .OrderBy(f => Marshal.OffsetOf(type, f.Name).ToInt64())];

    private static int FieldSize(Type type) => type.IsEnum ? Marshal.SizeOf(Enum.GetUnderlyingType(type)) : Marshal.SizeOf(type);

    private static int UnsafeSizeOf(Type type) =>
        (int)typeof(Unsafe).GetMethod(nameof(Unsafe.SizeOf))!.MakeGenericMethod(type).Invoke(null, null)!;

    private static bool IsReferenceOrContainsReferences(Type type) =>
        (bool)typeof(RuntimeHelpers).GetMethod(nameof(RuntimeHelpers.IsReferenceOrContainsReferences))!
            .MakeGenericMethod(type).Invoke(null, null)!;
}

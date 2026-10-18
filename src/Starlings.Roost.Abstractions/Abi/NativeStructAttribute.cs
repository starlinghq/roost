namespace Starlings.Roost.Abi;

/// <summary>
/// Marks a blittable struct as the field-for-field mirror of a C struct in <c>kernels/include/</c>.
/// </summary>
/// <remarks>
/// The layout tests compare every marked struct with the golden layout file that <c>tools/abi-layout.py</c> builds
/// with gcc. C field <c>snake_case</c> names map to C# <c>PascalCase</c> field names, in the same order.
/// </remarks>
/// <param name="name">The C struct name, for example <c>RoostRmsNormParams</c>.</param>
[AttributeUsage(AttributeTargets.Struct, Inherited = false)]
public sealed class NativeStructAttribute(string name) : Attribute
{
    /// <summary>The C struct name.</summary>
    public string Name { get; } = name;
}

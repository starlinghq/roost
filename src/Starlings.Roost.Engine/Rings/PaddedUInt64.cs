using System.Runtime.InteropServices;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// A 64-bit counter alone on its cache lines: 64 bytes of padding on each side of <see cref="Value"/>, so two of
/// them (or one and a neighbouring field) never share a line. Used for the managed MPSC queue cursors; the shared
/// segment's cursors get the same isolation from the C8 layout.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 128)]
internal struct PaddedInt64
{
    [FieldOffset(64)]
    public long Value;
}

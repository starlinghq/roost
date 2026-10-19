using System.Diagnostics;
using System.Security.Cryptography;
using Microsoft.Win32.SafeHandles;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// Streams a locked safetensors file into a <see cref="IWeightSink"/> and verifies it as it is read (R-024).
/// </summary>
/// <remarks>
/// <para>
/// The lock records one SHA-256 over the whole file, so the digest is an incremental hash over the bytes in file order,
/// header first. Every byte is read exactly once; the same buffer is hashed and then copied to the arena, so the bytes
/// that were verified are the bytes that were uploaded. The arena is laid out in slot order (<see cref="WeightLayout"/>),
/// which is the order the model's generated loader indexes <c>StepContext.Weights</c>; each tensor goes to its slot's
/// offset as its bytes arrive, so the order of arrival is the file's order (HF files are sorted by dtype and name, not
/// by layer) while the order of the arena is the model's. Seeking in slot order would make the whole-file digest
/// impossible to check.
/// </para>
/// <para>
/// A substituted file fails before the first forward, in three places, cheapest first: the size against the lock before
/// any read; the header (length, JSON, dtype, shape, a byte range per tensor with no gap) and every slot's tensor,
/// dtype and element count before the arena is allocated; and the digest at end of file, which decides
/// <see cref="IWeightSink.Commit"/> or <see cref="IWeightSink.Discard"/>. The digest cannot be known sooner, so a
/// same-size substitution is detected only after the last byte; nothing is handed out until then.
/// </para>
/// </remarks>
public static class WeightStreamer
{
    /// <summary>The default read size: large enough to keep NVMe and the PCIe copies busy, small enough to stay in cache.</summary>
    public const int DefaultChunkBytes = 8 * 1024 * 1024;

    /// <summary>Streams and verifies a checkpoint file.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <param name="locked">Its lock entry (size and SHA-256).</param>
    /// <param name="slots">The model's weight slots in loader order.</param>
    /// <param name="sink">The destination arena. On success it is committed; on any failure it is discarded.</param>
    /// <param name="chunkBytes">Read size; positive.</param>
    /// <param name="slotCopied">Called after all of a slot's bytes are copied, in file order, <b>before</b> the digest is known. For progress only: a slot is not usable until this method returns.</param>
    /// <returns>The layout of the committed arena.</returns>
    /// <exception cref="WeightLoadException">The file does not match the lock or the slots, or the sink cannot hold it.</exception>
    public static WeightLayout Load(
        string path,
        LockedFile locked,
        IReadOnlyList<WeightSlotSpec> slots,
        IWeightSink sink,
        int chunkBytes = DefaultChunkBytes,
        Action<SlotPlacement>? slotCopied = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(locked);
        ArgumentNullException.ThrowIfNull(slots);
        ArgumentNullException.ThrowIfNull(sink);
        ArgumentOutOfRangeException.ThrowIfLessThan(chunkBytes, 16);

        var clock = Stopwatch.StartNew();
        try
        {
            var layout = Stream(path, locked, slots, sink, chunkBytes, slotCopied, clock);
            sink.Commit();
            return layout;
        }
        catch
        {
            sink.Discard();
            throw;
        }
    }

    private static WeightLayout Stream(
        string path, LockedFile locked, IReadOnlyList<WeightSlotSpec> slots, IWeightSink sink, int chunkBytes, Action<SlotPlacement>? slotCopied, Stopwatch clock)
    {
        SafeFileHandle handle;
        try
        {
            handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.SequentialScan);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw Mismatch($"cannot open {Path.GetFileName(path)}: {e.Message}", e);
        }

        using (handle)
        {
            var size = RandomAccess.GetLength(handle);
            if (size != locked.Size)
            {
                throw Mismatch($"{locked.Path} is {size} bytes but the lock records {locked.Size}");
            }

            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var prefix = new byte[8];
            ReadExactly(handle, 0, prefix);
            hash.AppendData(prefix);
            var headerBytes = SafetensorsHeader.ReadLength(prefix, size);
            var json = new byte[headerBytes];
            ReadExactly(handle, 8, json);
            hash.AppendData(json);
            var header = SafetensorsHeader.Parse(json, size);

            var (placements, total) = WeightLayout.Plan(slots);
            var bySource = BindSlots(header, slots, placements);

            sink.Allocate(total);

            var buffer = new byte[chunkBytes];
            var remainingPerSlot = new long[placements.Length];
            for (var i = 0; i < placements.Length; i++)
            {
                remainingPerSlot[i] = placements[i].Bytes;
            }

            long position = header.DataStart;
            var next = 0; // index of the first tensor not yet finished
            while (position < size)
            {
                var want = (int)Math.Min(buffer.Length, size - position);
                var read = RandomAccess.Read(handle, buffer.AsSpan(0, want), position);
                if (read == 0)
                {
                    throw Mismatch($"{locked.Path} ended at {position} bytes; the lock records {locked.Size}");
                }

                var chunk = buffer.AsSpan(0, read);
                hash.AppendData(chunk);
                var chunkEnd = position + read;
                while (next < header.Tensors.Length)
                {
                    var t = header.Tensors[next];
                    var begin = header.DataStart + t.Begin;
                    var end = header.DataStart + t.End;
                    if (begin >= chunkEnd && end > begin)
                    {
                        break;
                    }

                    var from = Math.Max(begin, position);
                    var to = Math.Min(end, chunkEnd);
                    if (to > from && bySource.TryGetValue(t.Name, out var targets))
                    {
                        var piece = chunk[(int)(from - position)..(int)(to - position)];
                        foreach (var slot in targets)
                        {
                            sink.Write(placements[slot].Offset + (from - begin), piece);
                            remainingPerSlot[slot] -= piece.Length;
                            if (remainingPerSlot[slot] == 0)
                            {
                                slotCopied?.Invoke(placements[slot]);
                            }
                        }
                    }

                    if (end > chunkEnd)
                    {
                        break;
                    }

                    next++;
                }

                position = chunkEnd;
            }

            for (var i = 0; i < remainingPerSlot.Length; i++)
            {
                if (remainingPerSlot[i] != 0)
                {
                    throw new InvalidOperationException($"slot {placements[i].Name} received {placements[i].Bytes - remainingPerSlot[i]} of {placements[i].Bytes} bytes");
                }
            }

            // A file that grew after the size check carries bytes the lock does not cover.
            if (RandomAccess.GetLength(handle) != size || RandomAccess.Read(handle, buffer.AsSpan(0, 1), size) != 0)
            {
                throw Mismatch($"{locked.Path} changed size while it was being read");
            }

            var actual = Convert.ToHexStringLower(hash.GetHashAndReset());
            if (!string.Equals(actual, locked.Sha256, StringComparison.Ordinal))
            {
                throw Mismatch($"{locked.Path} has SHA-256 {actual} but the lock records {locked.Sha256}");
            }

            return new WeightLayout(placements, total, actual, size, clock.Elapsed);
        }
    }

    /// <summary>Checks every slot against the header and maps each source tensor to the slots it fills.</summary>
    private static Dictionary<string, List<int>> BindSlots(SafetensorsHeader header, IReadOnlyList<WeightSlotSpec> slots, SlotPlacement[] placements)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        var bySource = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (var i = 0; i < slots.Count; i++)
        {
            var s = slots[i];
            if (!names.Add(s.Name))
            {
                throw new ArgumentException($"slot name {s.Name} is listed twice", nameof(slots));
            }

            if (SafetensorsDType.ElementSize(s.Dtype) == 0 || s.Elements < 0)
            {
                throw new ArgumentException($"slot {s.Name} has an unsupported dtype or a negative element count", nameof(slots));
            }

            if (!header.ByName.TryGetValue(s.Source, out var t))
            {
                throw Mismatch($"the checkpoint has no tensor {s.Source} for slot {s.Name}");
            }

            if (t.Dtype != s.Dtype || t.Elements != s.Elements)
            {
                throw Mismatch($"slot {s.Name} expects {s.Dtype} x {s.Elements} but the checkpoint has {SafetensorsHeader.Describe(t)} for {s.Source}");
            }

            if (!bySource.TryGetValue(s.Source, out var list))
            {
                bySource[s.Source] = list = [];
            }

            list.Add(placements[i].Index);
        }

        return bySource;
    }

    private static void ReadExactly(SafeFileHandle handle, long offset, Span<byte> destination)
    {
        var done = 0;
        while (done < destination.Length)
        {
            var n = RandomAccess.Read(handle, destination[done..], offset + done);
            if (n == 0)
            {
                throw Mismatch("the file ended inside the safetensors header");
            }

            done += n;
        }
    }

    private static WeightLoadException Mismatch(string message, Exception? inner = null) =>
        new(RoostErrorCode.AssetMismatch, message, inner);
}

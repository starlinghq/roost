using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>A control message as the worker read it.</summary>
/// <param name="Kind">Sleep, wake or replace weights.</param>
/// <param name="ControlId">The slot's step id.</param>
/// <param name="Level">The sleep level (sleep only).</param>
/// <param name="WeightsDigest">Wake: the digest the weights must have afterwards. Replace: the new file's locked digest. 32 bytes, or default for a sleep.</param>
/// <param name="Source">Replace only: the new weights.</param>
public sealed record ControlRequest(RingSlotKind Kind, ulong ControlId, SleepLevel Level, ImmutableArray<byte> WeightsDigest, WeightSource? Source);

/// <summary>A control result as the host read it.</summary>
/// <param name="Kind">As the message.</param>
/// <param name="ControlId">As the message.</param>
/// <param name="Status">The outcome.</param>
/// <param name="ErrorCode">The status behind a failure.</param>
/// <param name="WorkerDuration">Message taken to answer written, on the worker's clock (a wake's restore time).</param>
/// <param name="Bytes">Sleep: bytes released; wake and replace: bytes allocated.</param>
/// <param name="WorkerState">The worker's state afterwards (<see cref="WorkerState.Sleeping"/> or <see cref="WorkerState.Ready"/>).</param>
/// <param name="WeightsDigest">The resident weights' verified digest afterwards; default while they are released.</param>
/// <param name="ErrorText">Why it failed; empty on success.</param>
public sealed record ControlReply(
    RingSlotKind Kind,
    ulong ControlId,
    ControlStatus Status,
    int ErrorCode,
    TimeSpan WorkerDuration,
    long Bytes,
    WorkerState WorkerState,
    ImmutableArray<byte> WeightsDigest,
    string ErrorText);

/// <summary>
/// Writes and reads the C8 control messages of layout 3 (R-100): a <see cref="ControlMessage"/> in a plan buffer, answered by
/// a <see cref="ControlResult"/> in a result buffer. Both ends use this one codec. Control messages are rare (an operator or a
/// training loop sends them), so unlike the step path this allocates.
/// </summary>
public static class ControlMessages
{
    /// <summary>Longest weights path accepted, in UTF-8 bytes.</summary>
    public const int MaxPathBytes = 4096;

    /// <summary>Longest error text written, in UTF-8 bytes; longer text is cut.</summary>
    public const int MaxErrorTextBytes = 1024;

    private static int MessageSize => Unsafe.SizeOf<ControlMessage>();

    private static int ResultSize => Unsafe.SizeOf<ControlResult>();

    /// <summary>Writes a sleep message.</summary>
    /// <param name="buffer">The plan buffer.</param>
    /// <param name="controlId">The slot's step id.</param>
    /// <param name="level">What to release.</param>
    /// <returns>Bytes written.</returns>
    public static int WriteSleep(Span<byte> buffer, ulong controlId, SleepLevel level)
    {
        if (level is not (SleepLevel.Kv or SleepLevel.Weights))
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "A sleep releases the KV cache or the KV cache and the weights.");
        }

        var message = Header(RingSlotKind.Sleep, controlId);
        message.SleepLevel = level;
        return Write(buffer, message, []);
    }

    /// <summary>Writes a wake message.</summary>
    /// <param name="buffer">The plan buffer.</param>
    /// <param name="controlId">The slot's step id.</param>
    /// <param name="expectedDigest">The SHA-256 the resident weights must have afterwards, 32 bytes.</param>
    /// <returns>Bytes written.</returns>
    public static int WriteWake(Span<byte> buffer, ulong controlId, ReadOnlySpan<byte> expectedDigest)
    {
        var message = Header(RingSlotKind.Wake, controlId);
        SetDigest(ref message, expectedDigest);
        return Write(buffer, message, []);
    }

    /// <summary>Writes a replace-weights message.</summary>
    /// <param name="buffer">The plan buffer.</param>
    /// <param name="controlId">The slot's step id.</param>
    /// <param name="source">The new weights; validated.</param>
    /// <returns>Bytes written.</returns>
    public static int WriteReplaceWeights(Span<byte> buffer, ulong controlId, WeightSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        source.Validate();
        var path = Encoding.UTF8.GetBytes(source.Path);
        if (path.Length > MaxPathBytes)
        {
            throw new ArgumentException($"The weights path is longer than {MaxPathBytes} UTF-8 bytes.", nameof(source));
        }

        var message = Header(RingSlotKind.ReplaceWeights, controlId);
        SetDigest(ref message, source.Sha256.AsSpan());
        message.WeightsSize = (ulong)source.Size;
        message.WeightsPath = new TableRef { Offset = (ulong)Align(MessageSize), Count = (uint)path.Length, Stride = 1 };
        return Write(buffer, message, path);
    }

    /// <summary>Reads a control message from the plan buffer its slot names.</summary>
    /// <param name="buffer">The plan buffer.</param>
    /// <param name="slot">The slot.</param>
    /// <returns>The request.</returns>
    /// <exception cref="InvalidDataException">The message is malformed or does not match the slot.</exception>
    public static ControlRequest Read(ReadOnlySpan<byte> buffer, in RingSlot slot)
    {
        if (buffer.Length < MessageSize)
        {
            throw new InvalidDataException("The plan buffer is smaller than a control message.");
        }

        var m = MemoryMarshal.Read<ControlMessage>(buffer);
        if (m.Magic != SharedLayout.MagicControl || m.LayoutVersion != SharedLayout.Version || m.HeaderSize < (uint)MessageSize
            || m.TotalBytes < m.HeaderSize || m.TotalBytes > (ulong)buffer.Length)
        {
            throw new InvalidDataException("The control message header is not valid (magic, layout version or sizes).");
        }

        if (m.Kind != slot.Kind || !m.Kind.IsControl() || m.ControlId != slot.StepId)
        {
            throw new InvalidDataException("The control message does not match its ring slot.");
        }

        if (m.Flags != 0 || m.Reserved0 != 0 || m.Reserved1 != 0)
        {
            throw new InvalidDataException("A reserved control message field is not zero.");
        }

        switch (m.Kind)
        {
            case RingSlotKind.Sleep:
                if (m.SleepLevel is not (SleepLevel.Kv or SleepLevel.Weights))
                {
                    throw new InvalidDataException($"Sleep level {(uint)m.SleepLevel} is not defined.");
                }

                return new ControlRequest(m.Kind, m.ControlId, m.SleepLevel, default, null);
            case RingSlotKind.Wake:
                return new ControlRequest(m.Kind, m.ControlId, SleepLevel.None, Digest(m.WeightsDigest0, m.WeightsDigest1, m.WeightsDigest2, m.WeightsDigest3), null);
            default:
                var path = m.WeightsPath;
                if (path.Stride != 1 || path.Count == 0 || path.Count > MaxPathBytes || path.Offset < m.HeaderSize
                    || path.Offset + path.Count > m.TotalBytes || m.WeightsSize > long.MaxValue)
                {
                    throw new InvalidDataException("The replace-weights message has no valid path or size.");
                }

                var digest = Digest(m.WeightsDigest0, m.WeightsDigest1, m.WeightsDigest2, m.WeightsDigest3);
                string text;
                try
                {
                    text = new UTF8Encoding(false, true).GetString(buffer.Slice((int)path.Offset, (int)path.Count));
                }
                catch (DecoderFallbackException e)
                {
                    throw new InvalidDataException("The weights path is not UTF-8.", e);
                }

                var source = new WeightSource(text, (long)m.WeightsSize, digest);
                try
                {
                    source.Validate();
                }
                catch (ArgumentException e)
                {
                    throw new InvalidDataException(e.Message, e);
                }

                return new ControlRequest(m.Kind, m.ControlId, SleepLevel.None, digest, source);
        }
    }

    /// <summary>Writes a control result.</summary>
    /// <param name="buffer">The result buffer.</param>
    /// <param name="kind">The message's kind.</param>
    /// <param name="controlId">The message's id.</param>
    /// <param name="outcome">What happened.</param>
    /// <param name="durationNs">Message taken to answer written.</param>
    /// <param name="state">The worker's state afterwards.</param>
    /// <param name="residentDigest">The resident weights' digest, or default.</param>
    /// <returns>Bytes written.</returns>
    public static int WriteResult(Span<byte> buffer, RingSlotKind kind, ulong controlId, in ControlOutcome outcome, ulong durationNs, WorkerState state, ImmutableArray<byte> residentDigest)
    {
        var text = string.IsNullOrEmpty(outcome.Message) ? [] : Encoding.UTF8.GetBytes(outcome.Message);
        int start = Align(ResultSize);
        int room = Math.Max(0, Math.Min(MaxErrorTextBytes, buffer.Length - start));
        if (text.Length > room)
        {
            text = text.AsSpan(0, room).ToArray();
        }

        var result = new ControlResult
        {
            Magic = SharedLayout.MagicControlResult,
            LayoutVersion = SharedLayout.Version,
            HeaderSize = (uint)ResultSize,
            Kind = kind,
            ControlId = controlId,
            TotalBytes = (ulong)(text.Length == 0 ? ResultSize : start + text.Length),
            Status = outcome.Status,
            ErrorCode = outcome.ErrorCode,
            DurationNs = durationNs,
            Bytes = (ulong)Math.Max(0, outcome.Bytes),
            WorkerState = state,
            ErrorText = text.Length == 0 ? default : new TableRef { Offset = (ulong)start, Count = (uint)text.Length, Stride = 1 },
        };
        if (!residentDigest.IsDefault && residentDigest.Length == 32)
        {
            var d = residentDigest.AsSpan();
            result.WeightsDigest0 = BinaryPrimitives.ReadUInt64LittleEndian(d);
            result.WeightsDigest1 = BinaryPrimitives.ReadUInt64LittleEndian(d[8..]);
            result.WeightsDigest2 = BinaryPrimitives.ReadUInt64LittleEndian(d[16..]);
            result.WeightsDigest3 = BinaryPrimitives.ReadUInt64LittleEndian(d[24..]);
        }

        MemoryMarshal.Write(buffer, in result);
        text.CopyTo(buffer[start..]);
        return (int)result.TotalBytes;
    }

    /// <summary>Reads a control result from the result buffer its slot names.</summary>
    /// <param name="buffer">The result buffer.</param>
    /// <param name="slot">The result slot.</param>
    /// <returns>The reply.</returns>
    /// <exception cref="InvalidDataException">The result is malformed or does not match the slot.</exception>
    public static ControlReply ReadResult(ReadOnlySpan<byte> buffer, in RingSlot slot)
    {
        if (buffer.Length < ResultSize)
        {
            throw new InvalidDataException("The result buffer is smaller than a control result.");
        }

        var r = MemoryMarshal.Read<ControlResult>(buffer);
        if (r.Magic != SharedLayout.MagicControlResult || r.LayoutVersion != SharedLayout.Version || r.HeaderSize < (uint)ResultSize
            || r.TotalBytes < r.HeaderSize || r.TotalBytes > (ulong)buffer.Length || r.Kind != slot.Kind || r.ControlId != slot.StepId)
        {
            throw new InvalidDataException("The control result is not valid or does not match its ring slot.");
        }

        var text = string.Empty;
        if (r.ErrorText.Count > 0)
        {
            if (r.ErrorText.Stride != 1 || r.ErrorText.Offset + r.ErrorText.Count > r.TotalBytes)
            {
                throw new InvalidDataException("The control result's error text is out of range.");
            }

            text = Encoding.UTF8.GetString(buffer.Slice((int)r.ErrorText.Offset, (int)r.ErrorText.Count));
        }

        var digest = r.WeightsDigest0 == 0 && r.WeightsDigest1 == 0 && r.WeightsDigest2 == 0 && r.WeightsDigest3 == 0
            ? default
            : Digest(r.WeightsDigest0, r.WeightsDigest1, r.WeightsDigest2, r.WeightsDigest3);
        return new ControlReply(
            r.Kind,
            r.ControlId,
            r.Status,
            r.ErrorCode,
            TimeSpan.FromTicks((long)Math.Min(r.DurationNs / 100, (ulong)TimeSpan.MaxValue.Ticks)),
            (long)Math.Min(r.Bytes, (ulong)long.MaxValue),
            r.WorkerState,
            digest,
            text);
    }

    private static ControlMessage Header(RingSlotKind kind, ulong controlId) => new()
    {
        Magic = SharedLayout.MagicControl,
        LayoutVersion = SharedLayout.Version,
        HeaderSize = (uint)MessageSize,
        Kind = kind,
        ControlId = controlId,
    };

    private static void SetDigest(ref ControlMessage message, ReadOnlySpan<byte> digest)
    {
        if (digest.Length != 32)
        {
            throw new ArgumentException("The weights digest is SHA-256: 32 bytes.", nameof(digest));
        }

        message.WeightsDigest0 = BinaryPrimitives.ReadUInt64LittleEndian(digest);
        message.WeightsDigest1 = BinaryPrimitives.ReadUInt64LittleEndian(digest[8..]);
        message.WeightsDigest2 = BinaryPrimitives.ReadUInt64LittleEndian(digest[16..]);
        message.WeightsDigest3 = BinaryPrimitives.ReadUInt64LittleEndian(digest[24..]);
    }

    private static int Write(Span<byte> buffer, ControlMessage message, ReadOnlySpan<byte> path)
    {
        int total = path.Length == 0 ? MessageSize : Align(MessageSize) + path.Length;
        if (buffer.Length < total)
        {
            throw new ArgumentException($"The plan buffer holds {buffer.Length} bytes; the control message needs {total}.", nameof(buffer));
        }

        message.TotalBytes = (ulong)total;
        buffer[..total].Clear();
        MemoryMarshal.Write(buffer, in message);
        path.CopyTo(buffer[Align(MessageSize)..]);
        return total;
    }

    private static ImmutableArray<byte> Digest(ulong d0, ulong d1, ulong d2, ulong d3)
    {
        var bytes = new byte[32];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, d0);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(8), d1);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(16), d2);
        BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(24), d3);
        return ImmutableCollectionsMarshal.AsImmutableArray(bytes);
    }

    private static int Align(int bytes) => (bytes + SharedLayout.Alignment - 1) / SharedLayout.Alignment * SharedLayout.Alignment;
}

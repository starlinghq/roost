using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Tests.Stress;

/// <summary>
/// What the two sides of the stress test write and check. Every ring slot and every word of every buffer is a pure
/// function of the step number, so a stale, torn, duplicated, reordered or lost slot or buffer is detected at the
/// first word that differs.
/// </summary>
internal static class StressProtocol
{
    public const ulong FirstStepId = 1_000_003;
    public const int PlanWordsMax = 1022;
    public const int ResultWordsMax = 254;

    public static SegmentOptions Options { get; } = new()
    {
        PlanSlotCount = 8,
        ResultSlotCount = 8,
        PlanBufferCount = 2,
        ResultBufferCount = 2,
        PlanBufferBytes = 16 + (PlanWordsMax * 8),
        ResultBufferBytes = 16 + (ResultWordsMax * 8),
    };

    public static ulong Mix(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    public static RingSlot PlanSlot(ulong sequence, int bufferCount)
    {
        var id = FirstStepId + sequence;
        return new RingSlot
        {
            StepId = id,
            Kind = RingSlotKind.Step,
            BufferIndex = (uint)(sequence % (ulong)bufferCount),
            DeadlineNs = Mix(id ^ 1),
            PublishedNs = Mix(id ^ 2),
        };
    }

    public static RingSlot ResultSlot(ulong sequence, int bufferCount)
    {
        var id = FirstStepId + sequence;
        return new RingSlot
        {
            StepId = id,
            Kind = RingSlotKind.Step,
            BufferIndex = (uint)(sequence % (ulong)bufferCount),
            DeadlineNs = Mix(id ^ 3),
            PublishedNs = Mix(id ^ 4),
        };
    }

    public static void FillPlan(Span<byte> buffer, ulong stepId) => Fill(buffer, stepId, 0x70, PlanWordsMax);

    public static void FillResult(Span<byte> buffer, ulong stepId) => Fill(buffer, stepId, 0x7e, ResultWordsMax);

    public static string? VerifyPlan(ReadOnlySpan<byte> buffer, ulong stepId) => Verify(buffer, stepId, 0x70, PlanWordsMax);

    public static string? VerifyResult(ReadOnlySpan<byte> buffer, ulong stepId) => Verify(buffer, stepId, 0x7e, ResultWordsMax);

    public static string? CheckSlot(in RingSlot slot, in RingSlot expected, string what)
    {
        if (slot.StepId != expected.StepId)
        {
            return $"{what}: step id {slot.StepId}, expected {expected.StepId}";
        }

        if (slot.Kind != expected.Kind || slot.BufferIndex != expected.BufferIndex)
        {
            return $"{what} {expected.StepId}: kind/buffer {slot.Kind}/{slot.BufferIndex}, expected {expected.Kind}/{expected.BufferIndex}";
        }

        if (slot.DeadlineNs != expected.DeadlineNs || slot.PublishedNs != expected.PublishedNs)
        {
            return $"{what} {expected.StepId}: deadline/published fields differ";
        }

        return null;
    }

    private static void Fill(Span<byte> buffer, ulong stepId, ulong salt, int maxWords)
    {
        var words = Words(stepId, maxWords);
        var u = MemoryMarshal.Cast<byte, ulong>(buffer);
        u[0] = stepId;
        u[1] = (ulong)words;
        for (var i = 0; i < words; i++)
        {
            u[2 + i] = Mix(stepId ^ (salt << 56) ^ ((ulong)i << 8));
        }

        u[2 + words] = ~stepId; // trailer: a buffer overwritten midway shows up as a header/trailer mismatch
    }

    private static string? Verify(ReadOnlySpan<byte> buffer, ulong stepId, ulong salt, int maxWords)
    {
        var u = MemoryMarshal.Cast<byte, ulong>(buffer);
        if (u[0] != stepId)
        {
            return $"buffer header holds step {u[0]}, expected {stepId}";
        }

        var words = Words(stepId, maxWords);
        if (u[1] != (ulong)words)
        {
            return $"step {stepId}: word count {u[1]}, expected {words}";
        }

        for (var i = 0; i < words; i++)
        {
            if (u[2 + i] != Mix(stepId ^ (salt << 56) ^ ((ulong)i << 8)))
            {
                return $"step {stepId}: word {i} of {words} is wrong (stale or torn buffer)";
            }
        }

        return u[2 + words] == ~stepId ? null : $"step {stepId}: trailer wrong (buffer rewritten while in use)";
    }

    private static int Words(ulong stepId, int maxWords) => 1 + (int)(Mix(stepId ^ 0xA5) % (ulong)(maxWords - 3));

    /// <summary>A small allocation-free generator (xorshift64*), one per thread.</summary>
    public struct Rng(ulong seed)
    {
        private ulong _state = seed == 0 ? 0x2545F4914F6CDD1DUL : seed;

        public uint Next(uint bound)
        {
            _state ^= _state >> 12;
            _state ^= _state << 25;
            _state ^= _state >> 27;
            return (uint)(((_state * 0x2545F4914F6CDD1DUL) >> 33) % bound);
        }

        /// <summary>Usually nothing; sometimes a short spin; rarely a sleep, so both rings reach full and empty.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RandomDelay()
        {
            var r = Next(4096);
            if (r < 64)
            {
                Thread.SpinWait((int)Next(400));
            }
            else if (r < 66)
            {
                Thread.SpinWait((int)Next(20_000));
            }
            else if (r == 66)
            {
                Thread.Sleep(1);
            }
        }
    }
}

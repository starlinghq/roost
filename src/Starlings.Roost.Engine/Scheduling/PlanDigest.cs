using System.Buffers.Binary;
using System.Security.Cryptography;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Engine.Scheduling;

/// <summary>Fingerprint of a plan: SHA-256 over each field in a fixed order, little-endian, independent of struct padding.</summary>
internal static class PlanDigest
{
    public static PlannedStep Seal(PlannedStep step)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var w = new Writer(h);
        w.U64(step.StepId);
        w.U32((uint)step.GroupCount);
        w.U32((uint)step.Flags);
        w.U32((uint)step.MathMode);
        w.U32((uint)step.Sequences.Length);
        foreach (var s in step.Sequences)
        {
            w.U64(s.SequenceHandle);
            w.U32(s.TokenStart);
            w.U32(s.TokenCount);
            w.U32(s.PositionOffset);
            w.U32(s.GroupBlocksStart);
            w.I32(s.RecurrentSlot);
            w.I32(s.LoraSlot);
            w.I32(s.GrammarSlot);
            w.I32(s.EmbeddingSlot);
            w.U32(s.SamplingIndex);
            w.U32(s.DraftCount);
            w.U32((uint)s.Flags);
            w.U32(s.StopStart);
            w.U32(s.StopCount);
            w.U32(s.MaxNewTokensLeft);
            w.U32(s.SpecTreeStart);
            w.U32(s.SpecTreeCount);
            w.I32(s.HeadRow);
            w.I32(s.EncoderOutput);
        }

        w.Ints(step.TokenIds);
        w.Ints(step.Positions);
        w.U32((uint)step.GroupBlocks.Length);
        foreach (var g in step.GroupBlocks)
        {
            w.U32(g.BlockStart);
            w.U32(g.BlockCount);
        }

        w.Ints(step.BlockIds);
        w.U32((uint)step.Sampling.Length);
        foreach (var s in step.Sampling)
        {
            w.U32(BitConverter.SingleToUInt32Bits(s.Temperature));
            w.U32(BitConverter.SingleToUInt32Bits(s.TopP));
            w.I32(s.TopK);
            w.U32(BitConverter.SingleToUInt32Bits(s.MinP));
            w.U32(BitConverter.SingleToUInt32Bits(s.RepetitionPenalty));
            w.U32(BitConverter.SingleToUInt32Bits(s.PresencePenalty));
            w.U32(BitConverter.SingleToUInt32Bits(s.FrequencyPenalty));
            w.U32(s.Logprobs);
            w.U64(s.Seed);
            w.U64(s.PhiloxOffset);
            w.U32((uint)s.Flags);
        }

        w.U32((uint)step.CacheOps.Length);
        foreach (var o in step.CacheOps)
        {
            w.U32((uint)o.Kind);
            w.U32(o.Group);
            w.U32(o.Src);
            w.U32(o.Dst);
            w.U32(o.Count);
            w.U32((uint)o.SrcTier);
            w.U32((uint)o.DstTier);
        }

        w.Ints(step.StopTokenIds);
        return new PlannedStep(step.StepId, step.GroupCount, step.Flags, step.MathMode, step.Sequences, step.TokenIds, step.Positions,
            step.GroupBlocks, step.BlockIds, step.Sampling, step.CacheOps, step.StopTokenIds, Convert.ToHexStringLower(h.GetHashAndReset()));
    }

    private readonly ref struct Writer(IncrementalHash hash)
    {
        public void U32(uint v)
        {
            Span<byte> b = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(b, v);
            hash.AppendData(b);
        }

        public void I32(int v) => U32((uint)v);

        public void U64(ulong v)
        {
            Span<byte> b = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(b, v);
            hash.AppendData(b);
        }

        public void Ints(System.Collections.Immutable.ImmutableArray<int> values)
        {
            U32((uint)values.Length);
            foreach (var v in values)
            {
                I32(v);
            }
        }
    }
}

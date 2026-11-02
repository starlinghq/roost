using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Worker.Eager;

/// <summary>What <see cref="HostSampler.Sample"/> produced for one step.</summary>
/// <param name="Tokens">The token of each sampling sequence, in plan order.</param>
/// <param name="Logprobs">
/// Per sampling sequence, its (1 + top-n) logprob rows, the chosen token first and then the top-n alternatives, or null
/// when the row asked for none (<see cref="PlanSampling.Logprobs"/> 0).
/// </param>
/// <param name="Stops">Per sampling sequence, whether the token is one of its stop ids.</param>
public sealed record SampledStep(int[] Tokens, ResultLogprob[]?[] Logprobs, bool[] Stops);

/// <summary>
/// The CPU sampler of a worker (R-042, CPU half): it runs the C2 sampling table of each plan through
/// <see cref="CpuOps.Sample"/>, with the token counts and prompt mask the penalties need, the plan's stop ids and the
/// top-n logprobs, and keeps the per-sequence token history across steps. A worker that uses it reports
/// <see cref="PlanLimits.Sampling"/>. The CUDA eager worker does not: it stays greedy until the GPU sampler exists.
/// </summary>
/// <remarks>
/// <para><b>History.</b> Every token of a sequence passes through a plan, the prompt in its prefill chunks and each sampled
/// token as the next decode input, so the sampler records each chunk's tokens at their positions and keeps the context
/// <c>[0, position_offset + token_count)</c> per sequence handle. The prompt is the context at the sequence's first sampling
/// row; the generated output is everything after it. A sequence preempted and recomputed (the planner restarts it at position
/// 0 with the same handle) keeps its prompt length, so its penalties continue unchanged.</para>
/// <para><b>Lifetime.</b> A sequence's history is dropped when it samples a stop id or its last allowed token. A sequence the
/// engine drops without telling the worker (a cancellation, a stop string matched on the host) is dropped when more than
/// <c>4 * maxSequences + 16</c> histories are held, least recently used first; a sequence that comes back after that starts a
/// new history, and its earlier output then counts as prompt.</para>
/// <para><b>Parameters.</b> A zero repetition penalty is an unset 1 (a zeroed table). Seeds and Philox offsets come from the
/// plan unchanged (the planner uses the generated-token count as the offset), so a draw depends only on the request's seed and
/// position, never on the batch.</para>
/// </remarks>
public sealed unsafe class HostSampler
{
    private readonly int _vocab;
    private readonly int _capacity;
    private readonly SampleMath _math;
    private readonly Dictionary<ulong, History> _histories = [];
    private long _clock;

    /// <summary>Creates the sampler.</summary>
    /// <param name="vocab">Logits per row.</param>
    /// <param name="maxSequences">Sequences per step, which sizes the history cache.</param>
    /// <param name="math">The sampling math; <see cref="SampleMath.Reference"/> by default.</param>
    public HostSampler(int vocab, int maxSequences, SampleMath math = SampleMath.Reference)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequences);
        _vocab = vocab;
        _capacity = (4 * maxSequences) + 16;
        _math = math;
    }

    /// <summary>Sequence histories held (diagnostics and tests).</summary>
    public int HistoryCount => _histories.Count;

    /// <summary>Records the plan's tokens and samples every sampling sequence.</summary>
    /// <param name="plan">A plan checked with <see cref="PlanLimits.Sampling"/> set.</param>
    /// <param name="logits">The selected rows of float32 logits, [sampling sequences, vocab], in plan order.</param>
    /// <returns>The tokens, logprobs and stop flags.</returns>
    /// <exception cref="ArgumentException"><paramref name="logits"/> does not hold one row per sampling sequence.</exception>
    public SampledStep Sample(ParsedPlan plan, ReadOnlySpan<float> logits)
    {
        ArgumentNullException.ThrowIfNull(plan);
        int rows = plan.Selected;
        if (logits.Length != (long)rows * _vocab)
        {
            throw new ArgumentException($"{logits.Length} logits for {rows} rows of {_vocab}", nameof(logits));
        }

        _clock++;
        var sampled = new ParsedSequence[rows];
        var histories = new History[rows];
        int row = 0;
        foreach (var s in plan.Sequences)
        {
            var h = Record(plan, s);
            if (s.Sample)
            {
                if (h.PromptLength < 0)
                {
                    h.PromptLength = s.KvLength;
                }

                sampled[row] = s;
                histories[row++] = h;
            }
        }

        var result = new SampledStep(new int[rows], new ResultLogprob[]?[rows], new bool[rows]);
        if (rows > 0)
        {
            Run(plan, logits, sampled, histories, result);
        }

        for (int r = 0; r < rows; r++)
        {
            if (result.Stops[r] || sampled[r].MaxNewTokensLeft == 1)
            {
                _histories.Remove(sampled[r].Handle);
            }
        }

        Evict();
        return result;
    }

    private History Record(ParsedPlan plan, in ParsedSequence s)
    {
        if (!_histories.TryGetValue(s.Handle, out var h))
        {
            h = new History();
            _histories.Add(s.Handle, h);
        }

        h.LastUsed = _clock;
        var tokens = h.Tokens;
        if (s.PositionOffset > tokens.Count)
        {
            // A history dropped by Evict and a sequence resumed mid-way: the gap is unknown, so the context starts here.
            tokens.Clear();
            h.PromptLength = -1;
            h.Base = s.PositionOffset;
        }

        int start = s.PositionOffset - h.Base;
        if (start < 0)
        {
            tokens.Clear();
            h.Base = 0;
            start = s.PositionOffset;
        }

        if (tokens.Count > start)
        {
            tokens.RemoveRange(start, tokens.Count - start);
        }

        for (int t = 0; t < s.TokenCount; t++)
        {
            tokens.Add(plan.TokenIds[s.TokenStart + t]);
        }

        return h;
    }

    private void Evict()
    {
        while (_histories.Count > _capacity)
        {
            ulong oldest = 0;
            long when = long.MaxValue;
            foreach (var (handle, h) in _histories)
            {
                if (h.LastUsed < when)
                {
                    when = h.LastUsed;
                    oldest = handle;
                }
            }

            _histories.Remove(oldest);
        }
    }

    private void Run(ParsedPlan plan, ReadOnlySpan<float> logits, ParsedSequence[] seqs, History[] histories, SampledStep result)
    {
        int rows = seqs.Length;
        int vocab = _vocab;
        int words = (vocab + 31) / 32;
        int top = 0;
        bool penalties = false;
        foreach (var s in seqs)
        {
            top = Math.Max(top, (int)s.Sampling.Logprobs);
            penalties |= s.Sampling.RepetitionPenalty is not (0f or 1f) || s.Sampling.PresencePenalty != 0f || s.Sampling.FrequencyPenalty != 0f;
        }

        top = Math.Min(top, vocab);
        var temperature = new float[rows];
        var topK = new int[rows];
        var topP = new float[rows];
        var minP = new float[rows];
        var seeds = new ulong[rows];
        var offsets = new ulong[rows];
        var repetition = new float[rows];
        var presence = new float[rows];
        var frequency = new float[rows];
        var counts = penalties ? new int[rows * vocab] : [];
        var prompt = penalties ? new uint[rows * words] : [];
        var stopRanges = new int[rows * 2];
        var stopIds = plan.StopIds.Length == 0 ? new int[1] : plan.StopIds;
        var ids = new int[rows];
        var logprob = new float[rows];
        var stopFlags = new int[rows];
        var topIds = new int[Math.Max(1, rows * top)];
        var topValues = new float[Math.Max(1, rows * top)];
        for (int r = 0; r < rows; r++)
        {
            var s = seqs[r];
            var sp = s.Sampling;
            bool greedy = (sp.Flags & SamplingOptions.Greedy) != 0;
            temperature[r] = greedy ? 0f : sp.Temperature;
            topK[r] = sp.TopK;
            topP[r] = sp.TopP == 0f ? 1f : sp.TopP;
            minP[r] = sp.MinP;
            seeds[r] = sp.Seed;
            offsets[r] = sp.PhiloxOffset;
            repetition[r] = sp.RepetitionPenalty == 0f ? 1f : sp.RepetitionPenalty;
            presence[r] = sp.PresencePenalty;
            frequency[r] = sp.FrequencyPenalty;
            stopRanges[2 * r] = s.StopStart;
            stopRanges[(2 * r) + 1] = s.StopCount;
            if (penalties)
            {
                var h = histories[r];
                var tokens = CollectionsMarshal.AsSpan(h.Tokens);
                int promptEnd = Math.Clamp(h.PromptLength - h.Base, 0, tokens.Length);
                foreach (int t in tokens[..promptEnd])
                {
                    prompt[(r * words) + (t >> 5)] |= 1u << (t & 31);
                }

                foreach (int t in tokens[promptEnd..])
                {
                    counts[(r * vocab) + t]++;
                }
            }
        }

        fixed (float* pLogits = logits, pTemp = temperature, pTopP = topP, pMinP = minP, pRep = repetition, pPres = presence, pFreq = frequency, pLogprob = logprob, pTopValues = topValues)
        fixed (int* pTopK = topK, pCounts = counts, pStopRanges = stopRanges, pStopIds = stopIds, pIds = ids, pStopFlags = stopFlags, pTopIds = topIds)
        fixed (ulong* pSeeds = seeds, pOffsets = offsets)
        fixed (uint* pPrompt = prompt)
        {
            CpuOps.Sample(new SampleArgs
            {
                Logits = Span(pLogits, logits.Length, DType.F32),
                Temperature = Span(pTemp, rows, DType.F32),
                TopK = Span(pTopK, rows, DType.I32),
                TopP = Span(pTopP, rows, DType.F32),
                MinP = Span(pMinP, rows, DType.F32),
                Seeds = Span(pSeeds, rows, DType.U64),
                Offsets = Span(pOffsets, rows, DType.U64),
                Ids = Span(pIds, rows, DType.I32),
                Logprobs = top > 0 ? Span(pLogprob, rows, DType.F32) : default,
                Rows = rows,
                Vocab = vocab,
                Stride = vocab,
                RepetitionPenalty = Span(pRep, rows, DType.F32),
                PresencePenalty = Span(pPres, rows, DType.F32),
                FrequencyPenalty = Span(pFreq, rows, DType.F32),
                TokenCounts = penalties ? Span(pCounts, counts.Length, DType.I32) : default,
                PromptMask = penalties ? Span(pPrompt, prompt.Length, DType.U32) : default,
                StopIds = Span(pStopIds, stopIds.Length, DType.I32),
                StopRanges = Span(pStopRanges, stopRanges.Length, DType.I32),
                StopFlags = Span(pStopFlags, rows, DType.I32),
                TopLogprobIds = top > 0 ? Span(pTopIds, rows * top, DType.I32) : default,
                TopLogprobValues = top > 0 ? Span(pTopValues, rows * top, DType.F32) : default,
                TopLogprobs = top,
                Math = _math,
            });
        }

        for (int r = 0; r < rows; r++)
        {
            result.Tokens[r] = ids[r];
            result.Stops[r] = stopFlags[r] != 0;
            int n = Math.Min((int)seqs[r].Sampling.Logprobs, top);
            if (n > 0)
            {
                var entry = new ResultLogprob[1 + n];
                entry[0] = new ResultLogprob { Token = ids[r], Logprob = logprob[r] };
                for (int i = 0; i < n; i++)
                {
                    entry[1 + i] = new ResultLogprob { Token = topIds[(r * top) + i], Logprob = topValues[(r * top) + i] };
                }

                result.Logprobs[r] = entry;
            }
        }
    }

    private static DeviceSpan Span(void* address, long length, DType dtype) =>
        new() { Address = (nint)address, Length = (ulong)length, Dtype = dtype };

    private sealed class History
    {
        // Tokens at positions [Base, Base + Count); Base is 0 unless the history was dropped and the sequence resumed mid-way.
        public List<int> Tokens { get; } = [];

        public int Base { get; set; }

        public int PromptLength { get; set; } = -1;

        public long LastUsed { get; set; }
    }
}

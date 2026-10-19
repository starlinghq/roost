using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Caching;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Scheduling;

/// <summary>What <see cref="StepPlanner.Add"/> did with a request.</summary>
public enum AddResult
{
    /// <summary>Queued.</summary>
    Added = 0,

    /// <summary>A sequence with this id is already live.</summary>
    DuplicateId,

    /// <summary>The prompt is empty, or <c>MaxTokens</c> is not positive (<see cref="RoostErrorCode.InvalidOption"/>).</summary>
    InvalidOption,

    /// <summary>The prompt alone needs more blocks than a pool has (<see cref="RoostErrorCode.PromptTooLong"/>).</summary>
    PromptTooLong,

    /// <summary>The request uses a feature the planner does not support yet (LoRA, grammar, multimodal, pooling, decision, speculation).</summary>
    Unsupported,
}

/// <summary>What <see cref="StepPlanner.Fork"/> did.</summary>
public enum ForkResult
{
    /// <summary>The branch row exists and shares the parent's prefix.</summary>
    Forked = 0,

    /// <summary>No live parent with that handle, or the id of the branch is taken.</summary>
    UnknownSequence,

    /// <summary>The parent holds no state yet, or it waits for a sampled token (its plan has not returned).</summary>
    NotForkable,

    /// <summary>The recurrent pool has no free slots. Nothing changed.</summary>
    NoCapacity,
}

/// <summary>
/// The step planner (R-029, stream G): turns the live sequences and the block pools into step plans (contract C2) and
/// the results of those plans back into sequence state.
/// </summary>
/// <remarks>
/// <para><b>Policy.</b> Sequences are ordered by priority (high first), then deadline (earliest first, none last),
/// then arrival. The order is a total order, so a plan never depends on hash or list order. A plan carries at most
/// <see cref="PlannerOptions.MaxSequencesPerStep"/> rows and <see cref="PlannerOptions.MaxTokensPerStep"/> tokens. A
/// decode row takes one token; a prefill row takes what is left of its prompt, cut to what is left of the budget
/// (chunked prefill is always on). Forked branch rows are ordinary sequences and count against the row cap.</para>
/// <para><b>Preemption.</b> When a sequence cannot get the blocks or the recurrent slot it needs, the planner preempts
/// sequences that are later in the order, from the last: recompute preemption. A victim keeps its tokens, including
/// those it generated, loses all state, and goes back to waiting; it prefills them again from position 0 (a worker
/// treats <c>position_offset == 0</c> as a zero recurrent state). A victim is chosen only when no uncompleted plan
/// carries it, and only if preempting every such victim could satisfy the request, so a stalled sequence does not
/// strip others for nothing. Sampling uses the generated-token count as the Philox offset, so a recomputed sequence
/// draws what it would have drawn.</para>
/// <para><b>R-028's obligations.</b> A block or slot named by an op or a table of a plan that has not completed is
/// never returned to its pool: it is parked, and released when the last such plan completes (<see cref="ApplyResult"/>,
/// <see cref="OnWorkerRestarted"/>). Ops waiting for the next plan (forks) count as named. A block shared by a fork is
/// made exclusive before a row writes into it (copy-on-write op in the plan).</para>
/// <para><b>Overlap.</b> A sequence whose plan sampled a token waits for that token; a sequence in mid-prefill is
/// planned again at once. Plans may therefore be built while earlier ones run (R-032 adds the device-side patching that
/// removes the wait for decode rows).</para>
/// <para><b>Determinism.</b> No clock, no randomness, no hash-order iteration: the plans are a function of the
/// sequence of calls (<see cref="PlannerInput"/>) and the block manager's initial state. <see cref="PlannedStep.Digest"/>
/// fingerprints a plan so a recorded run can be replayed and compared.</para>
/// <para><b>Threading.</b> Engine thread only, like the pools.</para>
/// <para><b>Not supported yet</b> (reported by <see cref="AddResult.Unsupported"/> or a constructor exception): LoRA,
/// grammar, multimodal input, pooling and decision rows, speculation, more than one recurrent group, and the cross,
/// draft, adapter, expert and encoder-output group kinds. Prefix-cache hits start a sequence at a nonzero computed
/// count in R-030; admission refusals are R-031.</para>
/// </remarks>
public sealed class StepPlanner
{
    private readonly BlockManager _blocks;
    private readonly PlannerOptions _options;
    private readonly int _groupCount;
    private readonly int _recurrentGroup = -1;
    private readonly List<SchedSeq> _seqs = [];
    private readonly Dictionary<ulong, SchedSeq> _byHandle = [];
    private readonly Dictionary<ulong, PlanRecord> _inFlight = [];
    private readonly Dictionary<long, int> _named = [];
    private readonly List<long> _parked = [];
    private readonly List<PlanCacheOp> _pendingOps = [];
    private readonly List<long> _pendingPins = [];
    private readonly List<PlannerNotice> _notices = [];
    private readonly int[] _need;
    private readonly PrefixCache? _cache;
    private long _arrival;
    private ulong _lastStep;

    /// <summary>Creates a planner over a block manager.</summary>
    /// <param name="blocks">The pools. The planner is their only user: it allocates, forks and releases through them.</param>
    /// <param name="options">The limits.</param>
    /// <exception cref="NotSupportedException">A group kind or the number of recurrent groups is not supported.</exception>
    public StepPlanner(BlockManager blocks, PlannerOptions options)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _blocks = blocks;
        _options = options;
        _groupCount = blocks.Groups.Length;
        _need = new int[_groupCount];
        foreach (var g in blocks.Groups)
        {
            switch (g.Kind)
            {
                case StateGroupKind.Full:
                case StateGroupKind.Sliding:
                case StateGroupKind.MlaLatent:
                    break;
                case StateGroupKind.Recurrent:
                    if (_recurrentGroup >= 0)
                    {
                        throw new NotSupportedException("The planner supports one recurrent group: the plan has one recurrent_slot per row.");
                    }

                    _recurrentGroup = g.Index;
                    break;
                default:
                    throw new NotSupportedException($"State group kind {g.Kind} (group {g.Name}) is not supported by the planner.");
            }
        }

        if (options.PrefixCache is { } cacheOptions)
        {
            _cache = new PrefixCache(blocks, cacheOptions, key => _named.ContainsKey(key));
        }
    }

    /// <summary>Live sequences (waiting and resident).</summary>
    public int SequenceCount => _seqs.Count;

    /// <summary>Plans handed out and not yet answered.</summary>
    public int InFlightPlans => _inFlight.Count;

    /// <summary>Blocks and slots held back from their pools until an uncompleted plan that names them completes.</summary>
    public int ParkedBlocks => _parked.Count;

    /// <summary>The prefix cache (R-030), or null when <see cref="PlannerOptions.PrefixCache"/> is not set.</summary>
    public PrefixCache? PrefixCache => _cache;

    /// <summary>Queues a request.</summary>
    /// <param name="request">The request. Its id value is the sequence handle.</param>
    /// <returns>The outcome.</returns>
    public AddResult Add(Request request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Lora.Value != 0 || request.Grammar.Value != 0 || request.ResponseFormat.Value != 0 || !request.MultimodalParts.IsDefaultOrEmpty
            || request.Decision is not null || request.Pooling is not null
            || request.Speculation.Mode is not (SpeculationMode.Default or SpeculationMode.Disabled))
        {
            return AddResult.Unsupported;
        }

        if (request.PromptTokenIds.IsDefaultOrEmpty || request.Sampling.MaxTokens <= 0)
        {
            return AddResult.InvalidOption;
        }

        if (_byHandle.ContainsKey(request.Id.Value))
        {
            return AddResult.DuplicateId;
        }

        foreach (var g in _blocks.Groups)
        {
            var pool = _blocks.GetPool(g.Index);
            var need = g.Kind == StateGroupKind.Recurrent ? 1
                : g.Kind == StateGroupKind.Sliding && g.Window > 0 ? CeilDiv(Math.Min(request.PromptTokenIds.Length + 1, g.Window + g.BlockSize + 1), g.BlockSize)
                : CeilDiv(request.PromptTokenIds.Length + 1, g.BlockSize);
            if (need > pool.Capacity)
            {
                return AddResult.PromptTooLong;
            }
        }

        var seq = new SchedSeq(request, _arrival++, _groupCount);
        seq.Tokens.AddRange(request.PromptTokenIds);
        seq.PromptLength = request.PromptTokenIds.Length;
        _seqs.Add(seq);
        _byHandle.Add(seq.Handle, seq);
        return AddResult.Added;
    }

    /// <summary>
    /// Creates a branch row from a live sequence (n &gt; 1, beam search): it shares the parent's paged blocks
    /// (copy-on-write) and gets a copy of its recurrent slot. The copy runs as ops of the next plan.
    /// </summary>
    /// <param name="parent">The parent's handle.</param>
    /// <param name="branch">The branch's request: its id, sampling and stop ids. Its prompt is ignored; the branch has the parent's tokens.</param>
    /// <returns>The outcome.</returns>
    public ForkResult Fork(ulong parent, Request branch)
    {
        ArgumentNullException.ThrowIfNull(branch);
        if (!_byHandle.TryGetValue(parent, out var p) || _byHandle.ContainsKey(branch.Id.Value))
        {
            return ForkResult.UnknownSequence;
        }

        if (!p.Resident || p.Computed >= p.Tokens.Count)
        {
            return ForkResult.NotForkable;
        }

        var child = new SchedSeq(branch, _arrival++, _groupCount) { PromptLength = p.PromptLength, Computed = p.Computed, Resident = true };
        child.Tokens.AddRange(p.Tokens);
        if (_recurrentGroup >= 0)
        {
            Span<int> src = [p.Slot];
            Span<int> dst = [BlockPool.NullBlock];
            Span<PlanCacheOp> ops = new PlanCacheOp[1];
            var n = _blocks.GetPool(_recurrentGroup).Fork(src, dst, ops);
            if (n < 0)
            {
                return ForkResult.NoCapacity;
            }

            child.Slot = dst[0];
            for (var i = 0; i < n; i++)
            {
                QueueOp(ops[i]);
            }
        }

        foreach (var g in _blocks.Groups)
        {
            if (g.Kind == StateGroupKind.Recurrent)
            {
                continue;
            }

            var src = p.Tables[g.Index];
            var dst = new int[src.Count];
            _blocks.GetPool(g.Index).Fork(src.ToArray(), dst, []);
            child.Tables[g.Index].AddRange(dst);
        }

        _seqs.Add(child);
        _byHandle.Add(child.Handle, child);
        return ForkResult.Forked;
    }

    /// <summary>Drops a sequence (client cancel). Its blocks return to the pools once no uncompleted plan names them.</summary>
    /// <param name="handle">The handle.</param>
    /// <returns>False when no such sequence is live.</returns>
    public bool Cancel(ulong handle)
    {
        if (!_byHandle.TryGetValue(handle, out var s))
        {
            return false;
        }

        Remove(s);
        return true;
    }

    /// <summary>A view of one sequence.</summary>
    /// <param name="handle">The handle.</param>
    /// <param name="info">The view.</param>
    /// <returns>False when no such sequence is live.</returns>
    public bool TryGetInfo(ulong handle, out SequenceInfo info)
    {
        if (_byHandle.TryGetValue(handle, out var s))
        {
            info = new SequenceInfo(handle, s.Tokens.Count, s.Computed, s.Resident, s.InFlight, s.Preemptions);
            return true;
        }

        info = default;
        return false;
    }

    /// <summary>Returns and clears the preemption and abort notices since the last call.</summary>
    /// <returns>The notices in the order they happened.</returns>
    public ImmutableArray<PlannerNotice> DrainNotices()
    {
        var result = _notices.ToImmutableArray();
        _notices.Clear();
        return result;
    }

    /// <summary>Builds the next plan.</summary>
    /// <returns>The plan, or null when no sequence can run now (nothing live, or every one waits for a result or for blocks).</returns>
    public PlannedStep? Plan()
    {
        while (true)
        {
            var step = TryBuildPlan(out var stalledHead);
            if (step is not null)
            {
                return step;
            }

            // Nothing could run. With no plan in flight nothing will free blocks, and the best-ranked sequence has
            // already had every later one as a possible victim: it cannot fit alone. Drop it and try the rest.
            if (stalledHead is null || _inFlight.Count > 0)
            {
                return null;
            }

            _notices.Add(new PlannerNotice(PlannerNoticeKind.Aborted, stalledHead.Handle));
            Remove(stalledHead);
        }
    }

    /// <summary>Applies the result of a plan: appends sampled tokens, finishes sequences, and releases what only that plan kept alive.</summary>
    /// <param name="outcome">The result.</param>
    /// <returns>The sequences that finished.</returns>
    /// <exception cref="InvalidOperationException">No such plan is in flight, or a row sampled more than one token.</exception>
    public ImmutableArray<SequenceFinished> ApplyResult(StepOutcome outcome)
    {
        ArgumentNullException.ThrowIfNull(outcome);
        if (!_inFlight.Remove(outcome.StepId, out var plan))
        {
            throw new InvalidOperationException($"Step {outcome.StepId} is not in flight.");
        }

        var finished = ImmutableArray.CreateBuilder<SequenceFinished>();
        foreach (var r in outcome.Sequences)
        {
            if (r.Tokens.Length > 1)
            {
                throw new InvalidOperationException("More than one sampled token per row needs speculation, which is not supported yet.");
            }
        }

        foreach (var handle in plan.Handles)
        {
            if (_byHandle.TryGetValue(handle, out var s))
            {
                s.InFlight--;
            }
        }

        foreach (var r in outcome.Sequences)
        {
            if (!_byHandle.TryGetValue(r.Handle, out var s))
            {
                continue; // cancelled, finished or dropped while the plan ran
            }

            // A preempted sequence keeps its token: it was sampled from valid state and recompute needs it.
            var reason = r.Finish;
            if (!r.Tokens.IsDefaultOrEmpty)
            {
                s.Tokens.Add(r.Tokens[0]);
            }

            if (reason == FinishReason.None && s.Request.Sampling.MaxTokens is { } max && s.Tokens.Count - s.PromptLength >= max)
            {
                reason = FinishReason.Length;
            }

            if (reason != FinishReason.None)
            {
                Remove(s);
                finished.Add(new SequenceFinished(s.Handle, reason));
            }
        }

        Unpin(plan.Pins);
        return finished.ToImmutable();
    }

    /// <summary>
    /// The worker was restarted: its device state is gone and no uncompleted plan will return. Every sequence is
    /// preempted (it recomputes), pending ops are dropped and nothing stays parked.
    /// </summary>
    public void OnWorkerRestarted()
    {
        foreach (var plan in _inFlight.Values)
        {
            Unpin(plan.Pins);
        }

        _inFlight.Clear();
        Unpin(_pendingPins);
        _pendingPins.Clear();
        _pendingOps.Clear();
        foreach (var s in _seqs)
        {
            s.InFlight = 0;
            if (s.Resident)
            {
                Preempt(s);
            }
        }

        _cache?.Clear(); // the restarted worker holds none of the cached blocks or snapshots
    }

    /// <summary>
    /// Checks that every block's reference count equals the number of table entries and parked entries that name it,
    /// and that the pools' own invariants hold. O(blocks), for tests.
    /// </summary>
    /// <exception cref="InvalidOperationException">An invariant does not hold.</exception>
    public void CheckInvariants()
    {
        var expected = new Dictionary<long, int>();
        foreach (var s in _seqs)
        {
            foreach (var g in _blocks.Groups)
            {
                if (g.Kind == StateGroupKind.Recurrent)
                {
                    if (s.Resident != (s.Slot >= 0))
                    {
                        throw new InvalidOperationException($"Sequence {s.Handle}: residency and recurrent slot disagree.");
                    }

                    if (s.Slot >= 0)
                    {
                        Bump(expected, Key(g.Index, s.Slot));
                    }

                    continue;
                }

                foreach (var b in s.Tables[g.Index])
                {
                    if (b != BlockPool.NullBlock)
                    {
                        Bump(expected, Key(g.Index, b));
                    }
                }
            }
        }

        foreach (var k in _parked)
        {
            Bump(expected, k);
        }

        if (_cache is not null)
        {
            _cache.CheckInvariants();
            _cache.CountReferences(expected);
            foreach (var s in _seqs)
            {
                if (s.PendingSnapshot >= 0)
                {
                    Bump(expected, SnapKey(_recurrentGroup, s.PendingSnapshot));
                }
            }
        }

        foreach (var g in _blocks.Groups)
        {
            var pool = _blocks.GetPool(g.Index);
            pool.CheckInvariants();
            for (var b = 0; b < pool.Capacity; b++)
            {
                expected.TryGetValue(Key(g.Index, b), out var want);
                if (pool.RefCount(b) != want)
                {
                    throw new InvalidOperationException($"Group {g.Index} block {b}: count {pool.RefCount(b)}, planner state names it {want} times.");
                }
            }

            if (g.Kind == StateGroupKind.Recurrent)
            {
                var snapshots = _blocks.GetSnapshotPool(g.Index);
                snapshots.CheckInvariants();
                for (var b = 0; b < snapshots.Capacity; b++)
                {
                    expected.TryGetValue(SnapKey(g.Index, b), out var want);
                    if (snapshots.RefCount(b) != want)
                    {
                        throw new InvalidOperationException($"Group {g.Index} snapshot {b}: count {snapshots.RefCount(b)}, planner state names it {want} times.");
                    }
                }
            }
        }

        foreach (var (k, v) in _named)
        {
            if (v <= 0)
            {
                throw new InvalidOperationException($"Pin {k} has count {v}.");
            }
        }
    }

    private static void Bump(Dictionary<long, int> map, long key) => map[key] = map.GetValueOrDefault(key) + 1;

    private static long Key(int group, int id) => BlockKey.Live(group, id);

    private static long SnapKey(int group, int id) => BlockKey.Snapshot(group, id);

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;

    private static int Compare(SchedSeq a, SchedSeq b)
    {
        var c = ((int)b.Request.Priority).CompareTo((int)a.Request.Priority);
        if (c != 0)
        {
            return c;
        }

        var da = a.Request.Deadline?.UtcTicks ?? long.MaxValue;
        var db = b.Request.Deadline?.UtcTicks ?? long.MaxValue;
        c = da.CompareTo(db);
        return c != 0 ? c : a.Arrival.CompareTo(b.Arrival);
    }

    private PlannedStep? TryBuildPlan(out SchedSeq? stalledHead)
    {
        stalledHead = null;
        var order = new List<SchedSeq>(_seqs);
        order.Sort(Compare);
        var rows = new List<(SchedSeq Seq, int Start, int Count)>();
        var ops = new List<PlanCacheOp>();
        var pins = new List<long>();
        var budget = _options.MaxTokensPerStep;

        for (var i = 0; i < order.Count && rows.Count < _options.MaxSequencesPerStep && budget > 0; i++)
        {
            var s = order[i];
            if (s.Computed >= s.Tokens.Count || (!s.Resident && s.InFlight > 0))
            {
                continue; // waits for its sampled token, or for the plan that still names its old state
            }

            var attached = _cache is not null && !s.Resident && s.Computed == 0 && AttachPrefix(s);
            var n = Math.Min(Math.Min(s.Tokens.Count - s.Computed, budget), _options.MaxChunkTokens);
            n = CapAtSnapshotBoundary(s, n);
            if (!TryReserve(s, n, order, i, ops, pins))
            {
                if (attached)
                {
                    DetachPrefix(s);
                }

                stalledHead ??= s;
                continue;
            }

            if (_cache is not null)
            {
                SnapshotAtRowStart(s, ops, pins);
            }

            rows.Add((s, s.Computed, n));
            budget -= n;
            s.Computed += n;
        }

        if (rows.Count == 0)
        {
            return null;
        }

        if (_cache is not null)
        {
            // Nodes for the blocks this plan completes are added only now, so no row of this plan reuses a block that the
            // same forward is still writing; every later plan runs after this one.
            foreach (var (s, _, _) in rows)
            {
                if (!s.CacheStuck)
                {
                    s.CacheStuck = !_cache.Sync(CollectionsMarshal.AsSpan(s.Tokens), s.Computed, s.Request.Lora.Value, s.Tables, s.Nodes);
                }
            }
        }

        return Finish(rows, ops, pins);
    }

    private bool TryReserve(SchedSeq s, int n, List<SchedSeq> order, int index, List<PlanCacheOp> ops, List<long> pins)
    {
        ComputeNeed(s, n);
        if (!Satisfied())
        {
            EvictCachedFor();
        }

        if (!Satisfied())
        {
            if (!Feasible(order, index))
            {
                return false;
            }

            for (var j = order.Count - 1; j > index && !Satisfied(); j--)
            {
                var v = order[j];
                if (v.Resident && v.InFlight == 0)
                {
                    Preempt(v);
                    EvictCachedFor(); // the victim's blocks may now be held by the cache alone
                }
            }

            if (!Satisfied())
            {
                return false;
            }
        }

        foreach (var g in _blocks.Groups)
        {
            var pool = _blocks.GetPool(g.Index);
            if (g.Kind == StateGroupKind.Recurrent)
            {
                if (!s.Resident)
                {
                    Span<int> one = [0];
                    pool.TryAllocate(one);
                    s.Slot = one[0];
                    if (s.PendingSnapshot >= 0)
                    {
                        // Restore the cached state into the new slot before this plan's forward. The plan names the
                        // snapshot until it completes, so the sequence's own reference can go back to the cache now.
                        ops.Add(_blocks.Restore(g.Index, s.PendingSnapshot, s.Slot));
                        Pin(pins, Key(g.Index, s.Slot));
                        Pin(pins, SnapKey(g.Index, s.PendingSnapshot));
                        ReleaseSnapshot(g.Index, s.PendingSnapshot);
                        s.PendingSnapshot = -1;
                    }
                }

                continue;
            }

            var table = s.Tables[g.Index];
            var bs = g.BlockSize;
            var first = s.Computed / bs;
            if (s.Computed % bs != 0 && first < table.Count && table[first] != BlockPool.NullBlock && pool.RefCount(table[first]) > 1)
            {
                if (pool.TryMakeExclusive(table[first], out var writable, out var copy) != ExclusiveResult.Copied)
                {
                    throw new InvalidOperationException("Copy-on-write failed after the free-block check.");
                }

                Pin(pins, Key(g.Index, (int)copy.Src));
                Pin(pins, Key(g.Index, (int)copy.Dst));
                ops.Add(copy);
                table[first] = writable;
            }

            var add = CeilDiv(s.Computed + n, bs) - table.Count;
            if (add > 0)
            {
                var fresh = new int[add];
                if (!pool.TryAllocate(fresh))
                {
                    throw new InvalidOperationException("Allocation failed after the free-block check.");
                }

                table.AddRange(fresh);
            }

            if (g.Kind == StateGroupKind.Sliding && g.Window > 0)
            {
                // Keys before computed - window are no longer read by this chunk or any later one.
                var drop = Math.Min(BlockPool.LeadingReleasableBlocks(s.Computed, g.Window, bs), table.Count);
                for (var k = 0; k < drop; k++)
                {
                    if (table[k] != BlockPool.NullBlock)
                    {
                        ReleaseBlock(g.Index, table[k]);
                        table[k] = BlockPool.NullBlock;
                    }
                }
            }
        }

        s.Resident = true;
        return true;
    }

    private void ComputeNeed(SchedSeq s, int n)
    {
        foreach (var g in _blocks.Groups)
        {
            if (g.Kind == StateGroupKind.Recurrent)
            {
                _need[g.Index] = s.Resident ? 0 : 1;
                continue;
            }

            var table = s.Tables[g.Index];
            var first = s.Computed / g.BlockSize;
            var cow = s.Computed % g.BlockSize != 0 && first < table.Count && table[first] != BlockPool.NullBlock
                && _blocks.GetPool(g.Index).RefCount(table[first]) > 1 ? 1 : 0;
            _need[g.Index] = Math.Max(0, CeilDiv(s.Computed + n, g.BlockSize) - table.Count) + cow;
        }
    }

    private bool Satisfied()
    {
        for (var g = 0; g < _groupCount; g++)
        {
            if (_need[g] > _blocks.GetPool(g).FreeCount)
            {
                return false;
            }
        }

        return true;
    }

    private bool Feasible(List<SchedSeq> order, int index)
    {
        for (var g = 0; g < _groupCount; g++)
        {
            var potential = _blocks.GetPool(g).FreeCount;
            for (var j = index + 1; j < order.Count; j++)
            {
                var v = order[j];
                if (v.Resident && v.InFlight == 0)
                {
                    potential += _blocks.Groups[g].Kind == StateGroupKind.Recurrent ? 1 : v.Tables[g].Count(b => b != BlockPool.NullBlock);
                }
            }

            if (_cache is not null && _blocks.Groups[g].Kind != StateGroupKind.Recurrent)
            {
                potential += _cache.EvictableNodes();
            }

            if (potential < _need[g])
            {
                return false;
            }
        }

        return true;
    }

    private PlannedStep Finish(List<(SchedSeq Seq, int Start, int Count)> rows, List<PlanCacheOp> newOps, List<long> pins)
    {
        var stepId = ++_lastStep;
        var allOps = new List<PlanCacheOp>(_pendingOps.Count + newOps.Count);
        allOps.AddRange(_pendingOps);
        allOps.AddRange(newOps);
        pins.AddRange(_pendingPins);
        _pendingOps.Clear();
        _pendingPins.Clear();

        var sequences = new List<PlanSequence>(rows.Count);
        var tokenIds = new List<int>();
        var positions = new List<int>();
        var groupBlocks = new List<PlanGroupBlocks>(rows.Count * _groupCount);
        var blockIds = new List<int>();
        var sampling = new List<PlanSampling>(rows.Count);
        var stops = new List<int>();
        var flags = PlanOptions.None;
        var handles = new List<ulong>(rows.Count);

        foreach (var (s, start, count) in rows)
        {
            var sampled = start + count == s.Tokens.Count;
            var decode = count == 1 && start >= s.PromptLength;
            var seqFlags = (decode ? SequenceOptions.None : SequenceOptions.Prefill) | (sampled ? SequenceOptions.Sample : SequenceOptions.None);
            if (!decode)
            {
                flags |= PlanOptions.Prefill;
            }

            var generated = s.Tokens.Count - s.PromptLength;
            var stopStart = stops.Count;
            stops.AddRange(s.Request.StopTokenIds.IsDefault ? [] : s.Request.StopTokenIds);
            stops.AddRange(_options.ModelStopTokenIds);
            var max = s.Request.Sampling.MaxTokens;
            var gbStart = groupBlocks.Count;
            foreach (var g in _blocks.Groups)
            {
                if (g.Kind == StateGroupKind.Recurrent)
                {
                    groupBlocks.Add(new PlanGroupBlocks { BlockStart = (uint)blockIds.Count, BlockCount = 0 });
                    Pin(pins, Key(g.Index, s.Slot));
                    continue;
                }

                var table = s.Tables[g.Index];
                groupBlocks.Add(new PlanGroupBlocks { BlockStart = (uint)blockIds.Count, BlockCount = (uint)table.Count });
                blockIds.AddRange(table);
                foreach (var b in table)
                {
                    if (b != BlockPool.NullBlock)
                    {
                        Pin(pins, Key(g.Index, b));
                    }
                }
            }

            sequences.Add(new PlanSequence
            {
                SequenceHandle = s.Handle,
                TokenStart = (uint)tokenIds.Count,
                TokenCount = (uint)count,
                PositionOffset = (uint)start,
                GroupBlocksStart = (uint)gbStart,
                RecurrentSlot = s.Slot,
                LoraSlot = -1,
                GrammarSlot = -1,
                EmbeddingSlot = -1,
                SamplingIndex = (uint)sampling.Count,
                DraftCount = 0,
                Flags = seqFlags,
                StopStart = (uint)stopStart,
                StopCount = (uint)(stops.Count - stopStart),
                MaxNewTokensLeft = max is { } m ? (uint)Math.Max(1, m - generated) : 0,
                HeadRow = -1,
                EncoderOutput = -1,
            });
            for (var i = 0; i < count; i++)
            {
                tokenIds.Add(s.Tokens[start + i]);
                positions.Add(start + i);
            }

            sampling.Add(MakeSampling(s.Request.Sampling, generated));
            s.InFlight++;
            handles.Add(s.Handle);
        }

        var step = new PlannedStep(stepId, _groupCount, flags, _options.MathMode, [.. sequences], [.. tokenIds], [.. positions],
            [.. groupBlocks], [.. blockIds], [.. sampling], [.. allOps], [.. stops], string.Empty);
        step = PlanDigest.Seal(step);
        _inFlight.Add(stepId, new PlanRecord(pins, handles));
        return step;
    }

    private static PlanSampling MakeSampling(SamplingParameters p, int generated) => new()
    {
        Temperature = p.Temperature,
        TopP = p.TopP,
        TopK = p.TopK,
        MinP = p.MinP,
        RepetitionPenalty = p.RepetitionPenalty,
        PresencePenalty = p.PresencePenalty,
        FrequencyPenalty = p.FrequencyPenalty,
        Logprobs = p.Logprobs ? (uint)Math.Max(1, p.TopLogprobs) : 0,
        Seed = p.Seed ?? 0,
        PhiloxOffset = (ulong)generated,
        Flags = (p.Temperature <= 0 ? SamplingOptions.Greedy : SamplingOptions.None) | (p.Seed.HasValue ? SamplingOptions.Seeded : SamplingOptions.None),
    };

    private void Preempt(SchedSeq s)
    {
        ReleaseAll(s);
        s.Computed = 0;
        s.Preemptions++;
        _notices.Add(new PlannerNotice(PlannerNoticeKind.Preempted, s.Handle));
    }

    private void Remove(SchedSeq s)
    {
        ReleaseAll(s);
        _seqs.Remove(s);
        _byHandle.Remove(s.Handle);
    }

    private void ReleaseAll(SchedSeq s)
    {
        foreach (var g in _blocks.Groups)
        {
            if (g.Kind == StateGroupKind.Recurrent)
            {
                if (s.Slot >= 0)
                {
                    ReleaseBlock(g.Index, s.Slot);
                }

                continue;
            }

            var table = s.Tables[g.Index];
            foreach (var b in table)
            {
                if (b != BlockPool.NullBlock)
                {
                    ReleaseBlock(g.Index, b);
                }
            }

            table.Clear();
        }

        if (s.PendingSnapshot >= 0)
        {
            ReleaseSnapshot(_recurrentGroup, s.PendingSnapshot);
            s.PendingSnapshot = -1;
        }

        s.Nodes.Clear();
        s.CacheStuck = false;
        s.Slot = -1;
        s.Resident = false;
    }

    private void ReleaseBlock(int group, int id)
    {
        var key = Key(group, id);
        if (_named.ContainsKey(key))
        {
            _parked.Add(key);
        }
        else
        {
            _blocks.GetPool(group).Release([id]);
        }
    }

    private void ReleaseSnapshot(int group, int id)
    {
        var key = SnapKey(group, id);
        if (_named.ContainsKey(key))
        {
            _parked.Add(key);
        }
        else
        {
            _blocks.GetSnapshotPool(group).Release([id]);
        }
    }

    private void QueueOp(in PlanCacheOp op)
    {
        _pendingOps.Add(op);
        for (var k = 0; k < (int)op.Count; k++)
        {
            Pin(_pendingPins, op.Kind == CacheOpKind.SnapshotRestore ? SnapKey((int)op.Group, (int)op.Src + k) : Key((int)op.Group, (int)op.Src + k));
            Pin(_pendingPins, op.Kind == CacheOpKind.SnapshotWrite ? SnapKey((int)op.Group, (int)op.Dst + k) : Key((int)op.Group, (int)op.Dst + k));
        }
    }

    // Prefix cache (R-030).

    /// <summary>Reuses the longest cached prefix of a sequence that is not resident. Returns whether any was reused.</summary>
    private bool AttachPrefix(SchedSeq s)
    {
        if (!_cache!.TryAttach(CollectionsMarshal.AsSpan(s.Tokens), s.Request.Lora.Value, s.Tables, s.Nodes, out var computed, out var snapshot))
        {
            return false;
        }

        s.Computed = computed;
        s.PendingSnapshot = snapshot;
        return true;
    }

    /// <summary>Gives back what <see cref="AttachPrefix"/> took when the sequence could not be made resident.</summary>
    private void DetachPrefix(SchedSeq s)
    {
        foreach (var g in _blocks.Groups)
        {
            if (g.Kind == StateGroupKind.Recurrent)
            {
                continue;
            }

            foreach (var b in s.Tables[g.Index])
            {
                if (b != BlockPool.NullBlock)
                {
                    ReleaseBlock(g.Index, b);
                }
            }

            s.Tables[g.Index].Clear();
        }

        if (s.PendingSnapshot >= 0)
        {
            ReleaseSnapshot(_recurrentGroup, s.PendingSnapshot);
            s.PendingSnapshot = -1;
        }

        s.Nodes.Clear();
        s.Computed = 0;
    }

    /// <summary>
    /// With a recurrent group, a chunk ends no later than the next snapshot boundary: a snapshot copies the state a
    /// finished plan left, so it can only be taken where a chunk ended.
    /// </summary>
    private int CapAtSnapshotBoundary(SchedSeq s, int n)
    {
        if (_cache is null || _recurrentGroup < 0)
        {
            return n;
        }

        var stride = _cache.SnapshotStrideTokens;
        return Math.Min(n, (((s.Computed / stride) + 1) * stride) - s.Computed);
    }

    /// <summary>
    /// If the sequence's state sits exactly on a snapshot boundary and the node there has no snapshot, writes one. The
    /// op runs before this plan's forward and after every earlier plan, so the slot holds the state after
    /// <c>Computed</c> tokens however many plans are in flight.
    /// </summary>
    private void SnapshotAtRowStart(SchedSeq s, List<PlanCacheOp> ops, List<long> pins)
    {
        if (_recurrentGroup < 0 || s.Computed == 0 || s.Computed % _cache!.SnapshotStrideTokens != 0)
        {
            return;
        }

        var index = (s.Computed / _cache.BlockSize) - 1;
        if (index >= s.Nodes.Count || s.Nodes[index].Dead || s.Nodes[index].Snapshot >= 0)
        {
            return;
        }

        if (_cache.TrySnapshot(s.Nodes[index], s.Slot, out var write))
        {
            ops.Add(write);
            Pin(pins, Key(_recurrentGroup, s.Slot));
            Pin(pins, SnapKey(_recurrentGroup, (int)write.Dst));
        }
    }

    /// <summary>Evicts least-recently-used cache nodes until the current need is met or none can go.</summary>
    private void EvictCachedFor()
    {
        if (_cache is null)
        {
            return;
        }

        while (!Satisfied() && _cache.EvictOne())
        {
        }
    }

    private void Pin(List<long> keys, long key)
    {
        _named[key] = _named.GetValueOrDefault(key) + 1;
        keys.Add(key);
    }

    private void Unpin(List<long> keys)
    {
        foreach (var k in keys)
        {
            if (--_named[k] == 0)
            {
                _named.Remove(k);
            }
        }

        var kept = 0;
        for (var i = 0; i < _parked.Count; i++)
        {
            var k = _parked[i];
            if (_named.ContainsKey(k))
            {
                _parked[kept++] = k;
            }
            else
            {
                var pool = BlockKey.IsSnapshot(k) ? _blocks.GetSnapshotPool(BlockKey.Group(k)) : _blocks.GetPool(BlockKey.Group(k));
                pool.Release([BlockKey.Id(k)]);
            }
        }

        _parked.RemoveRange(kept, _parked.Count - kept);
    }

    private sealed class PlanRecord(List<long> pins, List<ulong> handles)
    {
        public List<long> Pins { get; } = pins;

        public List<ulong> Handles { get; } = handles;
    }

    private sealed class SchedSeq(Request request, long arrival, int groups)
    {
        public Request Request { get; } = request;

        public ulong Handle { get; } = request.Id.Value;

        public long Arrival { get; } = arrival;

        public List<int> Tokens { get; } = [];

        public int PromptLength { get; set; }

        public int Computed { get; set; }

        public bool Resident { get; set; }

        public int Slot { get; set; } = -1;

        /// <summary>A snapshot id this sequence holds until its restore op is in a plan (prefix cache), or -1.</summary>
        public int PendingSnapshot { get; set; } = -1;

        /// <summary>The cache nodes of this sequence's full blocks, in order (prefix cache).</summary>
        public List<PrefixCache.Node> Nodes { get; } = [];

        /// <summary>A hash collision stopped this sequence's chain; it adds no more nodes.</summary>
        public bool CacheStuck { get; set; }

        public int InFlight { get; set; }

        public int Preemptions { get; set; }

        public List<int>[] Tables { get; } = Enumerable.Range(0, groups).Select(_ => new List<int>()).ToArray();
    }
}

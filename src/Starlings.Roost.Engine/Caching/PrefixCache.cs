using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Caching;

/// <summary>
/// The prefix cache (R-030, stream G): a tree of attention-block nodes over the block manager's pools, with a recurrent
/// snapshot per node for a hybrid model, evicted least-recently-used.
/// </summary>
/// <remarks>
/// <para><b>Nodes.</b> A node is one full attention block of one prefix. It names that block in every attention group
/// (all attention groups share one block size) and, when the model has a recurrent group, may name the snapshot of the
/// recurrent state after the block's last token. Its key in each group is the hash chain of <see cref="PrefixHash"/>; a
/// lookup also compares the parent node, the LoRA id and the block's token ids, so a hash collision is a miss, never a
/// wrong hit. A node is a hit only when it has a snapshot (or the model has no recurrent group) and every node before
/// it exists; a cached prefix therefore ends at an attention block boundary where the recurrent state is known.</para>
/// <para><b>References.</b> The cache holds one pool reference on each block and each snapshot it names. A sequence that
/// reuses a node takes its own reference. "Zero-reference" in the sense of R-030 is a node whose every id is held by the
/// cache alone (count 1) and is not pinned by an uncompleted plan: only those are evicted, and only leaves (a node with
/// children is kept, so no suffix is stranded without its prefix), least recently used first. Recency is a counter of
/// touches, never the clock, so a recorded run replays to the same evictions.</para>
/// <para><b>Threading.</b> Engine thread only, like the pools.</para>
/// </remarks>
public sealed class PrefixCache
{
    private readonly BlockManager _blocks;
    private readonly PrefixCacheOptions _options;
    private readonly Func<long, bool> _isPinned;
    private readonly int[] _attention;
    private readonly int _recurrent = -1;
    private readonly Dictionary<PrefixKey, Node> _index = [];
    private Node? _mru;
    private Node? _lru;
    private int _count;
    private long _lookups;
    private long _hits;
    private long _hitTokens;
    private long _inserted;
    private long _evicted;
    private long _snapshotsTaken;
    private long _snapshotsDropped;

    /// <summary>Creates a cache over a block manager.</summary>
    /// <param name="blocks">The block manager.</param>
    /// <param name="options">Settings.</param>
    /// <param name="isPinned">Whether a pool slot (a <see cref="BlockKey"/>) is named by an uncompleted plan; pinned slots are never evicted.</param>
    /// <exception cref="NotSupportedException">A group kind or block size the cache cannot serve.</exception>
    /// <exception cref="InvalidOperationException">A recurrent group without snapshot slots.</exception>
    internal PrefixCache(BlockManager blocks, PrefixCacheOptions options, Func<long, bool> isPinned)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(isPinned);
        options.Validate();
        _blocks = blocks;
        _options = options;
        _isPinned = isPinned;
        var attention = new List<int>();
        foreach (var g in blocks.Groups)
        {
            switch (g.Kind)
            {
                case StateGroupKind.Full:
                case StateGroupKind.MlaLatent:
                    if (attention.Count > 0 && blocks.Groups[attention[0]].BlockSize != g.BlockSize)
                    {
                        throw new NotSupportedException("The prefix cache needs one block size for all attention groups.");
                    }

                    attention.Add(g.Index);
                    break;
                case StateGroupKind.Recurrent:
                    if (blocks.GetSnapshotPool(g.Index).Capacity == 0)
                    {
                        throw new InvalidOperationException($"Recurrent group {g.Name} has no snapshot slots, so no prefix could be restored.");
                    }

                    _recurrent = g.Index;
                    break;
                default:
                    throw new NotSupportedException($"The prefix cache does not support state group kind {g.Kind} (group {g.Name}). Sliding-window groups are unsupported: a prefix hit would need the blocks inside the window, which the planner releases.");
            }
        }

        if (attention.Count == 0)
        {
            throw new NotSupportedException("The prefix cache needs an attention group.");
        }

        _attention = [.. attention];
        BlockSize = blocks.Groups[_attention[0]].BlockSize;
    }

    /// <summary>Tokens per attention block, the unit of reuse.</summary>
    public int BlockSize { get; }

    /// <summary>The recurrent group, or -1.</summary>
    internal int RecurrentGroup => _recurrent;

    /// <summary>Nodes in the cache.</summary>
    public int NodeCount => _count;

    /// <summary>The counters.</summary>
    public PrefixCacheStats Stats => new(_lookups, _hits, _hitTokens, _inserted, _evicted, _snapshotsTaken, _snapshotsDropped);

    /// <summary>Tokens at which a snapshot is taken for a recurrent group: a multiple of this, in tokens.</summary>
    internal int SnapshotStrideTokens => _options.SnapshotStrideBlocks * BlockSize;

    /// <summary>
    /// Finds the longest cached prefix of <paramref name="tokens"/> that leaves at least one token to compute, and
    /// gives the caller a reference to every block (and the snapshot) it reuses.
    /// </summary>
    /// <param name="tokens">The sequence's token ids.</param>
    /// <param name="lora">LoRA id; 0 for none.</param>
    /// <param name="tables">The sequence's block tables per group; the reused blocks are appended (they must be empty).</param>
    /// <param name="nodes">The sequence's node chain; the reused nodes are appended.</param>
    /// <param name="computed">Tokens covered by the reused blocks (0 for a miss).</param>
    /// <param name="snapshot">The held snapshot id to restore from, or -1.</param>
    /// <returns>True on a hit.</returns>
    internal bool TryAttach(ReadOnlySpan<int> tokens, ulong lora, List<int>[] tables, List<Node> nodes, out int computed, out int snapshot)
    {
        _lookups++;
        computed = 0;
        snapshot = -1;
        var maxBlocks = (tokens.Length - 1) / BlockSize;
        Node? parent = null;
        var chain = new List<Node>();
        var keys = new PrefixKey[_attention.Length];
        for (var k = 0; k < maxBlocks; k++)
        {
            var block = tokens.Slice(k * BlockSize, BlockSize);
            ChainKeys(parent, lora, block, keys);
            if (!_index.TryGetValue(keys[0], out var node) || !node.Matches(parent, lora, block, keys))
            {
                break;
            }

            chain.Add(node);
            parent = node;
        }

        var depth = chain.Count;
        while (depth > 0 && _recurrent >= 0 && chain[depth - 1].Snapshot < 0)
        {
            depth--;
        }

        if (depth == 0)
        {
            return false;
        }

        for (var k = depth - 1; k >= 0; k--)
        {
            Touch(chain[k]); // tail first, so the head is the most recent and leaves age out before their prefix
        }

        for (var k = 0; k < depth; k++)
        {
            var node = chain[k];
            for (var a = 0; a < _attention.Length; a++)
            {
                _blocks.GetPool(_attention[a]).Retain([node.Blocks[a]]);
                tables[_attention[a]].Add(node.Blocks[a]);
            }

            nodes.Add(node);
        }

        if (_recurrent >= 0)
        {
            snapshot = chain[depth - 1].Snapshot;
            _blocks.GetSnapshotPool(_recurrent).Retain([snapshot]);
        }

        computed = depth * BlockSize;
        _hits++;
        _hitTokens += computed;
        return true;
    }

    /// <summary>
    /// Adds a node for each full block of a sequence that has no node yet. Call it with the number of tokens the
    /// sequence has computed (counting the plan being built): those blocks hold, or will hold by the time any later plan
    /// runs, their final contents.
    /// </summary>
    /// <param name="tokens">The sequence's token ids.</param>
    /// <param name="computed">Tokens computed so far.</param>
    /// <param name="lora">LoRA id.</param>
    /// <param name="tables">The sequence's block tables per group.</param>
    /// <param name="nodes">The sequence's node chain, extended here.</param>
    /// <returns>False when a hash collision with different contents stopped the chain; the sequence should stop trying.</returns>
    internal bool Sync(ReadOnlySpan<int> tokens, int computed, ulong lora, List<int>[] tables, List<Node> nodes)
    {
        while (nodes.Count > 0 && nodes[^1].Dead)
        {
            nodes.RemoveAt(nodes.Count - 1);
        }

        var keys = new PrefixKey[_attention.Length];
        var full = computed / BlockSize;
        while (nodes.Count < full)
        {
            var k = nodes.Count;
            var parent = k > 0 ? nodes[k - 1] : null;
            var block = tokens.Slice(k * BlockSize, BlockSize);
            ChainKeys(parent, lora, block, keys);
            if (_index.TryGetValue(keys[0], out var existing))
            {
                if (!existing.Matches(parent, lora, block, keys))
                {
                    return false;
                }

                Touch(existing);
                nodes.Add(existing);
                continue;
            }

            var ids = new int[_attention.Length];
            for (var a = 0; a < _attention.Length; a++)
            {
                ids[a] = tables[_attention[a]][k];
            }

            for (var a = 0; a < _attention.Length; a++)
            {
                _blocks.GetPool(_attention[a]).Retain([ids[a]]);
            }

            var node = new Node(parent, lora, block.ToArray(), (PrefixKey[])keys.Clone(), ids);
            if (parent is not null)
            {
                parent.Children++;
            }

            _index.Add(keys[0], node);
            PushFront(node);
            _count++;
            _inserted++;
            nodes.Add(node);
        }

        return true;
    }

    /// <summary>
    /// Gives a node the snapshot of a live recurrent slot, as an op for the plan being built. The slot must hold the
    /// state after the node's last token.
    /// </summary>
    /// <param name="node">The node.</param>
    /// <param name="slot">The live slot.</param>
    /// <param name="write">The op.</param>
    /// <returns>False when no snapshot slot could be freed; nothing changed.</returns>
    internal bool TrySnapshot(Node node, int slot, out PlanCacheOp write)
    {
        if (!_blocks.TrySnapshot(_recurrent, slot, out var id, out write))
        {
            if (!DropOneSnapshot(node) || !_blocks.TrySnapshot(_recurrent, slot, out id, out write))
            {
                write = default;
                return false;
            }
        }

        node.Snapshot = id;
        Touch(node);
        _snapshotsTaken++;
        return true;
    }

    /// <summary>The number of nodes that eviction could free now (a tree may need several rounds).</summary>
    /// <returns>The count.</returns>
    internal int EvictableNodes()
    {
        var n = 0;
        for (var node = _lru; node is not null; node = node.MoreRecent)
        {
            if (IsZeroReference(node))
            {
                n++;
            }
        }

        return n;
    }

    /// <summary>Evicts the least recently used leaf whose blocks and snapshot only the cache holds, and is not pinned.</summary>
    /// <returns>False when there is none.</returns>
    internal bool EvictOne()
    {
        for (var node = _lru; node is not null; node = node.MoreRecent)
        {
            if (node.Children == 0 && IsZeroReference(node))
            {
                Remove(node);
                _evicted++;
                return true;
            }
        }

        return false;
    }

    /// <summary>Releases everything the cache holds (the worker restarted, so no cached state exists any more).</summary>
    internal void Clear()
    {
        while (_lru is not null)
        {
            Remove(_lru, force: true);
        }

        _index.Clear();
    }

    /// <summary>Adds the references the cache holds to <paramref name="expected"/>, keyed by <see cref="BlockKey"/>.</summary>
    /// <param name="expected">The map to add to.</param>
    internal void CountReferences(Dictionary<long, int> expected)
    {
        for (var node = _mru; node is not null; node = node.Older)
        {
            for (var a = 0; a < _attention.Length; a++)
            {
                var key = BlockKey.Live(_attention[a], node.Blocks[a]);
                expected[key] = expected.GetValueOrDefault(key) + 1;
            }

            if (node.Snapshot >= 0)
            {
                var key = BlockKey.Snapshot(_recurrent, node.Snapshot);
                expected[key] = expected.GetValueOrDefault(key) + 1;
            }
        }
    }

    /// <summary>Checks the tree against the index and the LRU list. For tests and diagnostics.</summary>
    /// <exception cref="InvalidOperationException">An invariant does not hold.</exception>
    public void CheckInvariants()
    {
        var seen = 0;
        var children = new Dictionary<Node, int>();
        Node? previous = null;
        for (var node = _mru; node is not null; node = node.Older)
        {
            seen++;
            if (node.Dead || node.MoreRecent != previous || !_index.TryGetValue(node.Keys[0], out var indexed) || indexed != node)
            {
                throw new InvalidOperationException("The LRU list and the index disagree.");
            }

            if (node.Parent is { } p)
            {
                children[p] = children.GetValueOrDefault(p) + 1;
            }

            previous = node;
        }

        if (previous != _lru || seen != _count || seen != _index.Count)
        {
            throw new InvalidOperationException($"LRU list holds {seen} nodes, count {_count}, index {_index.Count}.");
        }

        for (var node = _mru; node is not null; node = node.Older)
        {
            if (node.Children != children.GetValueOrDefault(node))
            {
                throw new InvalidOperationException("A node's child count is wrong.");
            }
        }
    }

    private void ChainKeys(Node? parent, ulong lora, ReadOnlySpan<int> block, PrefixKey[] keys)
    {
        for (var a = 0; a < _attention.Length; a++)
        {
            var from = parent is null
                ? PrefixHash.Seed(_options.Salt, _attention[a], BlockSize, lora)
                : parent.Keys[a];
            keys[a] = PrefixHash.Next(from, block);
        }
    }

    private bool IsZeroReference(Node node)
    {
        for (var a = 0; a < _attention.Length; a++)
        {
            if (_blocks.GetPool(_attention[a]).RefCount(node.Blocks[a]) != 1 || _isPinned(BlockKey.Live(_attention[a], node.Blocks[a])))
            {
                return false;
            }
        }

        return node.Snapshot < 0
            || (_blocks.GetSnapshotPool(_recurrent).RefCount(node.Snapshot) == 1 && !_isPinned(BlockKey.Snapshot(_recurrent, node.Snapshot)));
    }

    private bool DropOneSnapshot(Node except)
    {
        var pool = _blocks.GetSnapshotPool(_recurrent);
        for (var node = _lru; node is not null; node = node.MoreRecent)
        {
            if (node != except && node.Snapshot >= 0 && pool.RefCount(node.Snapshot) == 1 && !_isPinned(BlockKey.Snapshot(_recurrent, node.Snapshot)))
            {
                pool.Release([node.Snapshot]);
                node.Snapshot = -1;
                _snapshotsDropped++;
                return true;
            }
        }

        return false;
    }

    private void Remove(Node node, bool force = false)
    {
        if (!force && node.Children != 0)
        {
            throw new InvalidOperationException("Only a leaf is evicted.");
        }

        Unlink(node);
        for (var a = 0; a < _attention.Length; a++)
        {
            _blocks.GetPool(_attention[a]).Release([node.Blocks[a]]);
        }

        if (node.Snapshot >= 0)
        {
            _blocks.GetSnapshotPool(_recurrent).Release([node.Snapshot]);
            node.Snapshot = -1;
        }

        _index.Remove(node.Keys[0]);
        if (node.Parent is not null)
        {
            node.Parent.Children--;
        }

        node.Dead = true;
        _count--;
    }

    private void Touch(Node node)
    {
        if (_mru == node)
        {
            return;
        }

        Unlink(node);
        PushFront(node);
    }

    private void PushFront(Node node)
    {
        node.MoreRecent = null;
        node.Older = _mru;
        if (_mru is not null)
        {
            _mru.MoreRecent = node;
        }

        _mru = node;
        _lru ??= node;
    }

    private void Unlink(Node node)
    {
        if (node.MoreRecent is not null)
        {
            node.MoreRecent.Older = node.Older;
        }
        else
        {
            _mru = node.Older;
        }

        if (node.Older is not null)
        {
            node.Older.MoreRecent = node.MoreRecent;
        }
        else
        {
            _lru = node.MoreRecent;
        }

        node.MoreRecent = null;
        node.Older = null;
    }

    /// <summary>One cached attention block (see the class remarks).</summary>
    internal sealed class Node(Node? parent, ulong lora, int[] tokens, PrefixKey[] keys, int[] blocks)
    {
        public Node? Parent { get; } = parent;

        public ulong Lora { get; } = lora;

        public int[] Tokens { get; } = tokens;

        public PrefixKey[] Keys { get; } = keys;

        public int[] Blocks { get; } = blocks;

        public int Snapshot { get; set; } = -1;

        public int Children { get; set; }

        public bool Dead { get; set; }

        public Node? MoreRecent { get; set; }

        public Node? Older { get; set; }

        public bool Matches(Node? parent, ulong lora, ReadOnlySpan<int> block, PrefixKey[] keys)
        {
            if (Dead || Parent != parent || Lora != lora || !Tokens.AsSpan().SequenceEqual(block))
            {
                return false;
            }

            for (var a = 0; a < keys.Length; a++)
            {
                if (Keys[a] != keys[a])
                {
                    return false;
                }
            }

            return true;
        }
    }
}

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
/// <para><b>Host tier (R-094).</b> With a host-tier manager (<see cref="HostTier"/>), a node lives on one tier: its blocks
/// are device ids or host ids. Device eviction then demotes instead of dropping: the least recently used device node that
/// only the cache holds, is not pinned and has no device child gets host blocks and a <see cref="CacheOpKind.TierMove"/>
/// op (device to host) for the next plan, and its device blocks go back to the pool at once. That is safe because the
/// block is held by the cache alone and named by no uncompleted plan, and the op is queued ahead of every later op and
/// forward, so the copy reads the block before anything can write it. When the host tier is full, the least recently
/// used host leaf that is not pinned is dropped; when nothing on the host can go, the device leaf is dropped as without the
/// tier. A lookup that reaches host nodes restores them in chain order: device blocks (evicting or demoting other nodes if
/// needed), a <see cref="CacheOpKind.TierMove"/> op host to device, and the host blocks released once the plan that
/// carries the op completes (the planner pins both sides). The recurrent snapshot of a node stays in the device snapshot
/// pool whatever the tier of its attention blocks, so a hybrid model restores the same snapshot as before.</para>
/// <para><b>Threading.</b> Engine thread only, like the pools.</para>
/// </remarks>
public sealed class PrefixCache
{
    private readonly BlockManager _blocks;
    private readonly PrefixCacheOptions _options;
    private readonly Func<long, bool> _isPinned;
    private readonly BlockManager? _host;
    private readonly Action<PlanCacheOp>? _emitTierMove;
    private readonly Action<long>? _releaseHost;
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
    private long _demoted;
    private long _restored;
    private long _hostHits;
    private long _hostHitTokens;
    private long _hostEvicted;
    private int _hostCount;

    /// <summary>Creates a cache over a block manager.</summary>
    /// <param name="blocks">The block manager.</param>
    /// <param name="options">Settings.</param>
    /// <param name="isPinned">Whether a pool slot (a <see cref="BlockKey"/>) is named by an uncompleted plan; pinned slots are never evicted.</param>
    /// <exception cref="NotSupportedException">A group kind or block size the cache cannot serve.</exception>
    /// <exception cref="InvalidOperationException">A recurrent group without snapshot slots.</exception>
    internal PrefixCache(BlockManager blocks, PrefixCacheOptions options, Func<long, bool> isPinned)
        : this(blocks, options, isPinned, null, null, null)
    {
    }

    /// <summary>Creates a cache over a block manager with a host tier (R-094).</summary>
    /// <param name="blocks">The device block manager.</param>
    /// <param name="options">Settings.</param>
    /// <param name="isPinned">Whether a pool slot (a <see cref="BlockKey"/>, host keys included) is named by an uncompleted plan or a queued op.</param>
    /// <param name="host">The host-tier manager (<see cref="HostTier.Create"/>), or null for no host tier.</param>
    /// <param name="emitTierMove">Queues a <see cref="CacheOpKind.TierMove"/> op for the next plan and pins its host side (and, for a restore, its device side).</param>
    /// <param name="releaseHost">Drops the cache's reference to a host block, parking it while a plan or queued op names it.</param>
    internal PrefixCache(
        BlockManager blocks, PrefixCacheOptions options, Func<long, bool> isPinned, BlockManager? host, Action<PlanCacheOp>? emitTierMove, Action<long>? releaseHost)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(isPinned);
        options.Validate();
        if (host is not null)
        {
            ArgumentNullException.ThrowIfNull(emitTierMove);
            ArgumentNullException.ThrowIfNull(releaseHost);
            if (host.Groups.Length != blocks.Groups.Length)
            {
                throw new ArgumentException("The host tier must have the device groups.", nameof(host));
            }
        }

        _blocks = blocks;
        _options = options;
        _isPinned = isPinned;
        _host = host;
        _emitTierMove = emitTierMove;
        _releaseHost = releaseHost;
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
        if (host is not null)
        {
            foreach (var a in _attention)
            {
                var h = host.GetPool(a).Group;
                var d = blocks.Groups[a];
                if (h.Tier != BlockTier.Host || h.Kind != d.Kind || h.BlockSize != d.BlockSize || h.BytesPerBlock != d.BytesPerBlock)
                {
                    throw new ArgumentException($"Host group {a} does not match device group {d.Name}.", nameof(host));
                }
            }
        }
    }

    /// <summary>Tokens per attention block, the unit of reuse.</summary>
    public int BlockSize { get; }

    /// <summary>The recurrent group, or -1.</summary>
    internal int RecurrentGroup => _recurrent;

    /// <summary>Nodes in the cache.</summary>
    public int NodeCount => _count;

    /// <summary>The counters.</summary>
    public PrefixCacheStats Stats => new(_lookups, _hits, _hitTokens, _inserted, _evicted, _snapshotsTaken, _snapshotsDropped);

    /// <summary>The host-tier counters (R-094); all zero without a host tier.</summary>
    public HostTierStats HostStats => new(_demoted, _restored, _hostHits, _hostHitTokens, _hostEvicted, _hostCount);

    /// <summary>Whether the cache has a host tier.</summary>
    public bool HasHostTier => _host is not null;

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
    /// <param name="restored">Reused nodes that were on the host tier: their restore ops are queued for the next plan.</param>
    /// <returns>True on a hit.</returns>
    internal bool TryAttach(ReadOnlySpan<int> tokens, ulong lora, List<int>[] tables, List<Node> nodes, out int computed, out int snapshot, out int restored)
    {
        _lookups++;
        computed = 0;
        snapshot = -1;
        restored = 0;
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

        var depth = TrimToSnapshot(chain, chain.Count);
        if (depth == 0)
        {
            return false;
        }

        if (_host is not null)
        {
            depth = RestoreChain(chain, depth);
            if (depth == 0)
            {
                return false; // the nodes restored before the failure stay cached on the device
            }

            for (var k = 0; k < depth; k++)
            {
                if (chain[k].RestoredBy == _lookups)
                {
                    restored++;
                }
            }

            if (restored > 0)
            {
                _hostHits++;
                _hostHitTokens += (long)restored * BlockSize;
            }
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
                parent.DeviceChildren++;
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
            if (_host is null ? IsZeroReference(node) : node.Tier == BlockTier.Device && !node.Locked && DeviceBlocksFree(node))
            {
                n++;
            }
        }

        return n;
    }

    /// <summary>
    /// Frees one device block in every attention group. Without a host tier, evicts the least recently used leaf whose
    /// blocks and snapshot only the cache holds and are not pinned. With one, demotes the least recently used device node
    /// whose blocks only the cache holds, unpinned and with no device child, to the host tier (dropping a host leaf first
    /// when the host tier is full), or drops it when it is a leaf and the host tier has no room.
    /// </summary>
    /// <returns>False when there is none.</returns>
    internal bool EvictOne()
    {
        if (_host is null)
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

        for (var node = _lru; node is not null; node = node.MoreRecent)
        {
            if (node.Tier != BlockTier.Device || node.Locked || node.DeviceChildren != 0 || !DeviceBlocksFree(node))
            {
                continue;
            }

            if (TryDemote(node))
            {
                return true;
            }

            if (node.Children == 0 && SnapshotFree(node))
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

    /// <summary>Adds the references the cache holds to <paramref name="expected"/>, keyed by <see cref="BlockKey"/> (host keys for host-tier nodes).</summary>
    /// <param name="expected">The map to add to.</param>
    internal void CountReferences(Dictionary<long, int> expected)
    {
        for (var node = _mru; node is not null; node = node.Older)
        {
            for (var a = 0; a < _attention.Length; a++)
            {
                var key = BlockId(node, a);
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
        var host = 0;
        var children = new Dictionary<Node, int>();
        var deviceChildren = new Dictionary<Node, int>();
        Node? previous = null;
        for (var node = _mru; node is not null; node = node.Older)
        {
            seen++;
            if (node.Dead || node.MoreRecent != previous || !_index.TryGetValue(node.Keys[0], out var indexed) || indexed != node)
            {
                throw new InvalidOperationException("The LRU list and the index disagree.");
            }

            if (node.Locked || (node.Tier == BlockTier.Host && _host is null))
            {
                throw new InvalidOperationException("A node is locked outside a lookup, or on a tier the cache does not have.");
            }

            if (node.Parent is { } p)
            {
                children[p] = children.GetValueOrDefault(p) + 1;
                if (node.Tier == BlockTier.Device)
                {
                    deviceChildren[p] = deviceChildren.GetValueOrDefault(p) + 1;
                }
            }

            if (node.Tier == BlockTier.Host)
            {
                host++;
            }

            previous = node;
        }

        if (previous != _lru || seen != _count || seen != _index.Count || host != _hostCount)
        {
            throw new InvalidOperationException($"LRU list holds {seen} nodes ({host} on the host), count {_count} ({_hostCount}), index {_index.Count}.");
        }

        for (var node = _mru; node is not null; node = node.Older)
        {
            if (node.Children != children.GetValueOrDefault(node) || node.DeviceChildren != deviceChildren.GetValueOrDefault(node))
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

    private int TrimToSnapshot(List<Node> chain, int depth)
    {
        while (depth > 0 && _recurrent >= 0 && chain[depth - 1].Snapshot < 0)
        {
            depth--;
        }

        return depth;
    }

    private long BlockId(Node node, int a) =>
        node.Tier == BlockTier.Host ? BlockKey.Host(_attention[a], node.Blocks[a]) : BlockKey.Live(_attention[a], node.Blocks[a]);

    private bool IsZeroReference(Node node) => node.Tier == BlockTier.Device && DeviceBlocksFree(node) && SnapshotFree(node);

    /// <summary>The node's device blocks are held by the cache alone and named by no uncompleted plan or queued op.</summary>
    private bool DeviceBlocksFree(Node node)
    {
        for (var a = 0; a < _attention.Length; a++)
        {
            if (_blocks.GetPool(_attention[a]).RefCount(node.Blocks[a]) != 1 || _isPinned(BlockKey.Live(_attention[a], node.Blocks[a])))
            {
                return false;
            }
        }

        return true;
    }

    private bool SnapshotFree(Node node) =>
        node.Snapshot < 0
        || (_blocks.GetSnapshotPool(_recurrent).RefCount(node.Snapshot) == 1 && !_isPinned(BlockKey.Snapshot(_recurrent, node.Snapshot)));

    /// <summary>Restores the host nodes of <c>chain[0..depth)</c> in order; returns how far the chain is now on the device, trimmed to a snapshot.</summary>
    private int RestoreChain(List<Node> chain, int depth)
    {
        for (var k = 0; k < depth; k++)
        {
            chain[k].Locked = true; // neither demoted, dropped nor used as room for its own chain
        }

        var reached = depth;
        try
        {
            for (var k = 0; k < depth; k++)
            {
                if (chain[k].Tier == BlockTier.Host && !TryRestore(chain[k]))
                {
                    reached = k;
                    break;
                }
            }
        }
        finally
        {
            for (var k = 0; k < depth; k++)
            {
                chain[k].Locked = false;
            }
        }

        return TrimToSnapshot(chain, reached);
    }

    private bool HasRoom(BlockManager manager)
    {
        foreach (var a in _attention)
        {
            if (manager.GetPool(a).FreeCount == 0)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Moves a host node to new device blocks: one <see cref="CacheOpKind.TierMove"/> per group, host to device.</summary>
    private bool TryRestore(Node node)
    {
        while (!HasRoom(_blocks))
        {
            if (!EvictOne())
            {
                return false;
            }
        }

        Span<int> one = stackalloc int[1];
        for (var a = 0; a < _attention.Length; a++)
        {
            var g = _attention[a];
            _blocks.GetPool(g).TryAllocate(one);
            var hostId = node.Blocks[a];
            _emitTierMove!(new PlanCacheOp
            {
                Kind = CacheOpKind.TierMove,
                Group = (uint)g,
                Src = (uint)hostId,
                Dst = (uint)one[0],
                Count = 1,
                SrcTier = BlockTier.Host,
                DstTier = BlockTier.Device,
            });
            _releaseHost!(BlockKey.Host(g, hostId)); // pinned by the op just queued, so parked until that plan completes
            node.Blocks[a] = one[0];
        }

        node.Tier = BlockTier.Device;
        node.RestoredBy = _lookups;
        if (node.Parent is not null)
        {
            node.Parent.DeviceChildren++;
        }

        _hostCount--;
        _restored++;
        return true;
    }

    /// <summary>Moves a device node to new host blocks: one <see cref="CacheOpKind.TierMove"/> per group, device to host (see the class remarks).</summary>
    private bool TryDemote(Node node)
    {
        while (!HasRoom(_host!))
        {
            if (!EvictHostOne())
            {
                return false;
            }
        }

        Span<int> one = stackalloc int[1];
        for (var a = 0; a < _attention.Length; a++)
        {
            var g = _attention[a];
            _host!.GetPool(g).TryAllocate(one);
            var deviceId = node.Blocks[a];
            _emitTierMove!(new PlanCacheOp
            {
                Kind = CacheOpKind.TierMove,
                Group = (uint)g,
                Src = (uint)deviceId,
                Dst = (uint)one[0],
                Count = 1,
                SrcTier = BlockTier.Device,
                DstTier = BlockTier.Host,
            });
            _blocks.GetPool(g).Release([deviceId]);
            node.Blocks[a] = one[0];
        }

        node.Tier = BlockTier.Host;
        if (node.Parent is not null)
        {
            node.Parent.DeviceChildren--;
        }

        _hostCount++;
        _demoted++;
        return true;
    }

    /// <summary>Drops the least recently used host leaf that no plan or queued op names. Host pressure only.</summary>
    private bool EvictHostOne()
    {
        for (var node = _lru; node is not null; node = node.MoreRecent)
        {
            if (node.Tier != BlockTier.Host || node.Locked || node.Children != 0 || !SnapshotFree(node))
            {
                continue;
            }

            var pinned = false;
            for (var a = 0; a < _attention.Length && !pinned; a++)
            {
                pinned = _isPinned(BlockKey.Host(_attention[a], node.Blocks[a]));
            }

            if (!pinned)
            {
                Remove(node);
                _hostEvicted++;
                _evicted++;
                return true;
            }
        }

        return false;
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
        var tier = node.Tier == BlockTier.Host ? _host! : _blocks;
        for (var a = 0; a < _attention.Length; a++)
        {
            tier.GetPool(_attention[a]).Release([node.Blocks[a]]);
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
            if (node.Tier == BlockTier.Device)
            {
                node.Parent.DeviceChildren--;
            }
        }

        if (node.Tier == BlockTier.Host)
        {
            _hostCount--;
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

        /// <summary>Children whose blocks are on the device (R-094); a device node is demoted only when this is 0.</summary>
        public int DeviceChildren { get; set; }

        /// <summary>The tier of <see cref="Blocks"/>: device ids, or host ids after a demotion (R-094).</summary>
        public BlockTier Tier { get; set; } = BlockTier.Device;

        /// <summary>Held by the lookup in progress: not demoted, dropped or restored over.</summary>
        public bool Locked { get; set; }

        /// <summary>The lookup number that last restored this node from the host tier, or 0.</summary>
        public long RestoredBy { get; set; }

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

using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Blocks;

/// <summary>
/// Randomised, seeded, model-based test of the block manager. Each seed replays the same operation sequence, so a
/// failure message carries the seed and the step and the run can be repeated. A reference model (a reference count per
/// id, plus the holders' tables) is updated by every operation and compared with the pools after every step, together
/// with <see cref="BlockPool.CheckInvariants"/>. The pools are engine-thread-only by design, so the test is single
/// threaded: it covers interleavings of operations, not memory-ordering races. Nothing here runs in a kernel, so no
/// compute-sanitizer claim is made for it.
/// </summary>
public sealed class BlockStressTests
{
    private const int Steps = 600;

    private static readonly int[] Seeds = [.. Enumerable.Range(1, 150)];

    public static TheoryData<int> SeedData() => [.. Seeds];

    [Theory]
    [MemberData(nameof(SeedData))]
    public async Task RandomOperationsKeepEveryInvariant(int seed)
    {
        using var run = new Run(seed);
        await run.ExecuteAsync();
    }

    private sealed class Holder(int group)
    {
        public int Group { get; } = group;

        public List<int> Blocks { get; } = [];

        public int Tokens { get; set; }
    }

    private sealed class Run : IDisposable
    {
        private readonly int _seed;
        private readonly Random _rng;
        private readonly BlockManager _manager;
        private readonly InProcessBlockTransfer _transfer;
        private readonly Dictionary<int, int>[] _live = [[], [], []];
        private readonly Dictionary<int, int> _snapshots = [];
        private readonly List<Holder> _holders = [];
        private readonly List<int> _snapshotHolders = [];
        private readonly Dictionary<ulong, (BlockExport Export, int[] Blocks)> _exports = [];
        private readonly List<int[]> _queued = [];
        private readonly List<(long Ticket, int[] Blocks)> _inFlight = [];
        private int _step;

        public Run(int seed)
        {
            _seed = seed;
            _rng = new Random(seed);
            _manager = new BlockManager(
                [TestGroups.Full(0, 4), TestGroups.Sliding(1, 2, 5), TestGroups.Recurrent(2, snapshotSlots: 4)],
                [new GroupCapacity(0, 24, 0), new GroupCapacity(1, 16, 0), new GroupCapacity(2, 6, 4)]);
            _transfer = new InProcessBlockTransfer(_manager, _manager);
        }

        public void Dispose() => _transfer.Dispose();

        public async Task ExecuteAsync()
        {
            for (_step = 0; _step < Steps; _step++)
            {
                switch (_rng.Next(14))
                {
                    case 0 or 1: Allocate(); break;
                    case 2 or 3: ForkHolder(); break;
                    case 4 or 5: ReleaseHolder(); break;
                    case 6 or 7: MakeExclusive(); break;
                    case 8: Slide(); break;
                    case 9: Snapshot(); break;
                    case 10: RestoreOrReleaseSnapshot(); break;
                    case 11: await ExportAsync(); break;
                    case 12: await ImportOrCancelAsync(); break;
                    default: DrainOrComplete(); break;
                }

                Verify();
            }

            // Wind down: everything given back leaves every pool full.
            foreach (var h in _holders.ToArray())
            {
                ReleaseHolder(h);
            }

            foreach (var s in _snapshotHolders.ToArray())
            {
                _manager.GetSnapshotPool(2).Release([s]);
                Drop(_snapshots, s);
            }

            _transfer.Dispose();
            foreach (var blocks in _exports.Values.Select(e => e.Blocks).Concat(_queued).Concat(_inFlight.Select(f => f.Blocks)))
            {
                foreach (var b in blocks)
                {
                    Drop(_live[0], b);
                }
            }

            _exports.Clear();
            _queued.Clear();
            _inFlight.Clear();
            Verify();
            for (var g = 0; g < 3; g++)
            {
                Assert.True(_manager.GetPool(g).FreeCount == _manager.GetPool(g).Capacity, Where($"group {g} not empty at the end"));
            }

            Assert.True(_manager.GetSnapshotPool(2).FreeCount == 4, Where("snapshots not empty at the end"));
        }

        private string Where(string what) => $"{what} (seed {_seed}, step {_step})";

        private static void Drop(Dictionary<int, int> model, int id)
        {
            if (--model[id] == 0)
            {
                model.Remove(id);
            }
        }

        private static void Take(Dictionary<int, int> model, int id) => model[id] = model.GetValueOrDefault(id) + 1;

        private int Free(int group) => _manager.GetPool(group).Capacity - _live[group].Count;

        private Holder? PickHolder(Func<Holder, bool>? where = null)
        {
            var candidates = _holders.Where(h => where?.Invoke(h) ?? true).ToList();
            return candidates.Count == 0 ? null : candidates[_rng.Next(candidates.Count)];
        }

        private void Allocate()
        {
            var group = _rng.Next(3);
            var n = group == 2 ? 1 : _rng.Next(1, 4);
            var ids = new int[n];
            var ok = _manager.GetPool(group).TryAllocate(ids);
            Assert.True(ok == (n <= Free(group)), Where("allocation outcome differs from the model"));
            if (!ok)
            {
                return;
            }

            var holder = new Holder(group);
            foreach (var id in ids)
            {
                Assert.False(_live[group].ContainsKey(id), Where($"block {id} of group {group} handed out while held"));
                Take(_live[group], id);
                holder.Blocks.Add(id);
            }

            holder.Tokens = group == 1 ? n * 2 : n * 4;
            _holders.Add(holder);
        }

        private void ForkHolder()
        {
            var parent = PickHolder();
            if (parent is null)
            {
                return;
            }

            var pool = _manager.GetPool(parent.Group);
            var src = parent.Blocks.ToArray();
            var forked = new int[src.Length];
            var ops = new PlanCacheOp[src.Length];
            var live = src.Count(b => b != BlockPool.NullBlock);
            var n = pool.Fork(src, forked, ops);
            var child = new Holder(parent.Group) { Tokens = parent.Tokens };
            if (parent.Group == 2)
            {
                if (live > Free(2))
                {
                    Assert.True(n == -1, Where("fork should have been refused"));
                    return;
                }

                Assert.True(n >= 0, Where("fork refused with slots free"));
                var copied = 0;
                for (var i = 0; i < n; i++)
                {
                    Assert.Equal(CacheOpKind.Fork, ops[i].Kind);
                    copied += (int)ops[i].Count;
                    for (var k = 0; k < ops[i].Count; k++)
                    {
                        Assert.True(_live[2].ContainsKey((int)ops[i].Src + k), Where("fork op source is not held"));
                        Assert.False(_live[2].ContainsKey((int)ops[i].Dst + k), Where("fork op destination was held"));
                    }
                }

                Assert.True(copied == live, Where("fork ops do not cover the slots"));
                foreach (var id in forked.Where(b => b != BlockPool.NullBlock))
                {
                    Take(_live[2], id);
                }
            }
            else
            {
                Assert.True(n == 0, Where("a paged fork produced ops"));
                Assert.True(src.SequenceEqual(forked), Where("a paged fork changed block ids"));
                foreach (var id in forked.Where(b => b != BlockPool.NullBlock))
                {
                    Take(_live[parent.Group], id);
                }
            }

            child.Blocks.AddRange(forked);
            _holders.Add(child);
        }

        private void ReleaseHolder()
        {
            var h = PickHolder();
            if (h is not null)
            {
                ReleaseHolder(h);
            }
        }

        private void ReleaseHolder(Holder h)
        {
            _manager.GetPool(h.Group).Release(h.Blocks.ToArray());
            foreach (var id in h.Blocks.Where(b => b != BlockPool.NullBlock))
            {
                Drop(_live[h.Group], id);
            }

            _holders.Remove(h);
        }

        private void MakeExclusive()
        {
            var h = PickHolder(x => x.Group != 2 && x.Blocks.Any(b => b != BlockPool.NullBlock));
            if (h is null)
            {
                return;
            }

            var candidates = Enumerable.Range(0, h.Blocks.Count).Where(i => h.Blocks[i] != BlockPool.NullBlock).ToList();
            var idx = candidates[_rng.Next(candidates.Count)];
            var old = h.Blocks[idx];
            var shared = _live[h.Group][old] > 1;
            var result = _manager.GetPool(h.Group).TryMakeExclusive(old, out var writable, out var op);
            if (!shared)
            {
                Assert.True(result == ExclusiveResult.AlreadyExclusive && writable == old, Where("an exclusive block was copied"));
            }
            else if (Free(h.Group) == 0)
            {
                Assert.True(result == ExclusiveResult.OutOfBlocks && writable == old, Where("copy-on-write should have failed"));
            }
            else
            {
                Assert.True(result == ExclusiveResult.Copied, Where("copy-on-write did not copy"));
                Assert.True(op.Kind == CacheOpKind.Copy && op.Src == (uint)old && op.Dst == (uint)writable && op.Count == 1, Where("wrong copy op"));
                Assert.False(_live[h.Group].ContainsKey(writable), Where("the copy target was held"));
                Drop(_live[h.Group], old);
                Take(_live[h.Group], writable);
                h.Blocks[idx] = writable;
            }
        }

        private void Slide()
        {
            var h = PickHolder(x => x.Group == 1);
            if (h is null)
            {
                return;
            }

            // Extend by a few tokens, allocating blocks as the table grows, then release what left the window.
            var newTokens = h.Tokens + _rng.Next(1, 5);
            var needed = (newTokens + 1) / 2;
            var pool = _manager.GetPool(1);
            while (h.Blocks.Count < needed)
            {
                var one = new int[1];
                if (!pool.TryAllocate(one))
                {
                    Assert.True(Free(1) == 0, Where("sliding allocation failed with blocks free"));
                    return;
                }

                Assert.False(_live[1].ContainsKey(one[0]), Where("sliding block handed out while held"));
                Take(_live[1], one[0]);
                h.Blocks.Add(one[0]);
            }

            h.Tokens = newTokens;
            var table = h.Blocks.ToArray();
            // Independent oracle: a block goes when every position in it is below tokens - window (window 5, block 2).
            var leading = Enumerable.Range(0, table.Length).Count(i => ((i + 1) * 2) <= h.Tokens - 5);
            var expected = table.Take(Math.Min(leading, table.Length)).Count(b => b != BlockPool.NullBlock);
            var released = pool.ReleaseOutOfWindow(table, h.Tokens);
            Assert.True(released == expected, Where("wrong number of blocks released from the window"));
            for (var i = 0; i < table.Length; i++)
            {
                if (i < leading && h.Blocks[i] != BlockPool.NullBlock)
                {
                    Assert.True(table[i] == BlockPool.NullBlock, Where("a block that left the window is still in the table"));
                    Drop(_live[1], h.Blocks[i]);
                }
                else
                {
                    Assert.True(table[i] == h.Blocks[i], Where("a block in the window was changed"));
                }

                h.Blocks[i] = table[i];
            }
        }

        private void Snapshot()
        {
            var h = PickHolder(x => x.Group == 2 && x.Blocks.Any(b => b != BlockPool.NullBlock));
            if (h is null)
            {
                return;
            }

            var slot = h.Blocks.First(b => b != BlockPool.NullBlock);
            var ok = _manager.TrySnapshot(2, slot, out var snap, out var op);
            Assert.True(ok == (_snapshots.Count < 4), Where("snapshot outcome differs from the model"));
            if (!ok)
            {
                return;
            }

            Assert.False(_snapshots.ContainsKey(snap), Where("snapshot id handed out while held"));
            Assert.True(op.Kind == CacheOpKind.SnapshotWrite && op.Src == (uint)slot && op.Dst == (uint)snap, Where("wrong snapshot op"));
            Take(_snapshots, snap);
            _snapshotHolders.Add(snap);
        }

        private void RestoreOrReleaseSnapshot()
        {
            if (_snapshotHolders.Count == 0)
            {
                return;
            }

            var snap = _snapshotHolders[_rng.Next(_snapshotHolders.Count)];
            var target = PickHolder(x => x.Group == 2 && x.Blocks.Any(b => b != BlockPool.NullBlock));
            switch (_rng.Next(3))
            {
                case 0 when target is not null:
                    var slot = target.Blocks.First(b => b != BlockPool.NullBlock);
                    var op = _manager.Restore(2, snap, slot);
                    Assert.True(op.Kind == CacheOpKind.SnapshotRestore && op.Src == (uint)snap && op.Dst == (uint)slot, Where("wrong restore op"));
                    break;
                case 1:
                    _manager.GetSnapshotPool(2).Retain([snap]); // a second owner, as the prefix cache will have
                    Take(_snapshots, snap);
                    _snapshotHolders.Add(snap);
                    break;
                default:
                    _manager.GetSnapshotPool(2).Release([snap]);
                    Drop(_snapshots, snap);
                    _snapshotHolders.Remove(snap);
                    break;
            }
        }

        private async Task ExportAsync()
        {
            var h = PickHolder(x => x.Group == 0 && x.Blocks.Any(b => b != BlockPool.NullBlock));
            if (h is null)
            {
                return;
            }

            var blocks = h.Blocks.Where(b => b != BlockPool.NullBlock).ToArray();
            var export = await _transfer.ExportBlocksAsync(0, blocks, CancellationToken.None);
            foreach (var b in blocks)
            {
                Take(_live[0], b); // the pin
            }

            _exports.Add(export.TransferId, (export, blocks));
        }

        private async Task ImportOrCancelAsync()
        {
            if (_exports.Count == 0)
            {
                return;
            }

            var id = _exports.Keys.ElementAt(_rng.Next(_exports.Count));
            var (export, blocks) = _exports[id];
            if (_rng.Next(3) == 0)
            {
                Assert.True(_transfer.CancelExport(id), Where("cancel of a pending export failed"));
                foreach (var b in blocks)
                {
                    Drop(_live[0], b);
                }

                _exports.Remove(id);
                return;
            }

            var dst = new int[blocks.Length];
            if (blocks.Length > Free(0))
            {
                await Assert.ThrowsAsync<InvalidOperationException>(async () => await _transfer.ImportBlocksAsync(export, dst, CancellationToken.None));
                Assert.True(_transfer.PendingExports == _exports.Count, Where("a failed import dropped the export"));
                return;
            }

            await _transfer.ImportBlocksAsync(export, dst, CancellationToken.None);
            var holder = new Holder(0) { Tokens = blocks.Length * 4 };
            foreach (var b in dst)
            {
                Assert.False(_live[0].ContainsKey(b), Where("import destination was held"));
                Take(_live[0], b);
                holder.Blocks.Add(b);
            }

            _holders.Add(holder);
            _exports.Remove(id);
            _queued.Add(blocks); // the pin stays until the plan completes
        }

        private void DrainOrComplete()
        {
            if (_rng.Next(2) == 0 && _inFlight.Count > 0)
            {
                var pick = _inFlight[_rng.Next(_inFlight.Count)].Ticket;
                _transfer.Complete(pick);
                foreach (var f in _inFlight.Where(f => f.Ticket <= pick).ToArray())
                {
                    foreach (var b in f.Blocks)
                    {
                        Drop(_live[0], b);
                    }

                    _inFlight.Remove(f);
                }

                return;
            }

            var ops = new PlanCacheOp[64];
            var n = _transfer.Drain(ops, out var ticket);
            if (n == 0)
            {
                Assert.True(_queued.Count == 0 && ticket == 0, Where("drain returned nothing with imports queued"));
                return;
            }

            var covered = 0;
            for (var i = 0; i < n; i++)
            {
                covered += (int)ops[i].Count;
            }

            var drained = _queued.ToArray();
            Assert.True(covered == drained.Sum(b => b.Length), Where("drained ops do not cover the queued imports"));
            foreach (var blocks in drained)
            {
                _inFlight.Add((ticket, blocks));
            }

            _queued.Clear();
        }

        private void Verify()
        {
            for (var g = 0; g < 3; g++)
            {
                var pool = _manager.GetPool(g);
                pool.CheckInvariants();
                for (var b = 0; b < pool.Capacity; b++)
                {
                    Assert.True(pool.RefCount(b) == _live[g].GetValueOrDefault(b), Where($"group {g} block {b}: count {pool.RefCount(b)}, model {_live[g].GetValueOrDefault(b)}"));
                }
            }

            var snaps = _manager.GetSnapshotPool(2);
            snaps.CheckInvariants();
            for (var b = 0; b < snaps.Capacity; b++)
            {
                Assert.True(snaps.RefCount(b) == _snapshots.GetValueOrDefault(b), Where($"snapshot {b} count differs from the model"));
            }

            Assert.True(_transfer.PendingExports == _exports.Count, Where("pending exports differ"));
            Assert.True(_transfer.QueuedImports + _transfer.InFlightImports == _queued.Count + _inFlight.Count, Where("import batches differ"));
        }
    }
}

using System.Runtime.CompilerServices;
using Starlings.Roost.Engine.Rings;

namespace Starlings.Roost.Engine.Tests.Rings;

public sealed class MpscQueueTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(12)]
    public void CapacityMustBeAPowerOfTwo(int capacity) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new MpscQueue<object>(capacity));

    [Fact]
    public void FifoFullAndWrap()
    {
        var q = new MpscQueue<string>(4);
        Assert.False(q.TryDequeue(out _));
        for (var lap = 0; lap < 10; lap++)
        {
            for (var i = 0; i < 4; i++)
            {
                Assert.True(q.TryEnqueue($"{lap}:{i}"));
            }

            Assert.False(q.TryEnqueue("overflow"));
            Assert.Equal(4, q.Count);
            for (var i = 0; i < 4; i++)
            {
                Assert.True(q.TryDequeue(out var s));
                Assert.Equal($"{lap}:{i}", s);
            }

            Assert.False(q.TryDequeue(out _));
        }
    }

    [Fact]
    public void ClearsDequeuedReferences()
    {
        var q = new MpscQueue<object>(2);
        var weak = EnqueueNew(q);
        Assert.True(DequeueAndDrop(q));
        for (var i = 0; i < 3 && weak.IsAlive; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        Assert.False(weak.IsAlive);
    }

    [Fact]
    public async Task ManyProducersOneConsumerKeepPerProducerOrderAndLoseNothing()
    {
        const int producers = 6;
        const int perProducer = 200_000;
        var q = new MpscQueue<Item>(256);
        var tasks = new Task[producers];
        for (var p = 0; p < producers; p++)
        {
            var id = p;
            tasks[p] = Task.Factory.StartNew(
                () =>
                {
                    var rng = new Random(id);
                    for (var i = 0; i < perProducer; i++)
                    {
                        var item = new Item(id, i);
                        var spin = new SpinWait();
                        while (!q.TryEnqueue(item))
                        {
                            spin.SpinOnce(-1);
                        }

                        if (rng.Next(1024) == 0)
                        {
                            Thread.SpinWait(rng.Next(2000));
                        }
                    }
                },
                TaskCreationOptions.LongRunning);
        }

        var next = new int[producers];
        var received = 0;
        var deadline = Environment.TickCount64 + 120_000;
        var wait = new SpinWait();
        while (received < producers * perProducer)
        {
            if (q.TryDequeue(out var item))
            {
                Assert.Equal(next[item.Producer], item.Sequence);
                next[item.Producer]++;
                received++;
                wait.Reset();
            }
            else
            {
                Assert.True(Environment.TickCount64 < deadline, "queue stalled");
                wait.SpinOnce(-1);
            }
        }

        await Task.WhenAll(tasks);
        Assert.False(q.TryDequeue(out _));
        Assert.All(next, n => Assert.Equal(perProducer, n));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference EnqueueNew(MpscQueue<object> q)
    {
        var box = new object();
        Assert.True(q.TryEnqueue(box));
        return new WeakReference(box);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool DequeueAndDrop(MpscQueue<object> q) => q.TryDequeue(out _);

    private sealed record Item(int Producer, int Sequence);
}

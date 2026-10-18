namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// Measures bytes allocated by the current thread across calls of a step (S4's method). <see cref="AssertSteady"/>
/// first warms the step past the runtime's tier-up (the call-count threshold plus the background compile delay), then
/// requires every measured round to allocate nothing.
/// </summary>
public static class AllocationProbe
{
    public static long[] Rounds(Action step, int warmup = 400, int rounds = 6, int calls = 2000)
    {
        for (int i = 0; i < warmup; i++)
        {
            step();
        }

        var result = new long[rounds];
        for (int r = 0; r < rounds; r++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < calls; i++)
            {
                step();
            }

            result[r] = GC.GetAllocatedBytesForCurrentThread() - before;
        }

        return result;
    }

    /// <summary>
    /// Warms the step until the runtime has promoted it, then requires a full set of rounds that all allocate zero bytes.
    /// Tier-up runs on a background timer, so on a loaded machine it can land late; the measurement is therefore retried
    /// (up to five times, with a settling pause) and the step passes when one attempt has every round at zero. A step
    /// that allocates per call allocates in every round of every attempt and still fails.
    /// </summary>
    public static void AssertSteady(Action step)
    {
        for (int i = 0; i < 4000; i++)
        {
            step();
        }

        var seen = new List<string>();
        for (int attempt = 0; attempt < 5; attempt++)
        {
            Thread.Sleep(400);
            long[] rounds = Rounds(step, warmup: 4000);
            if (rounds.All(r => r == 0))
            {
                return;
            }

            seen.Add(string.Join(", ", rounds));
        }

        Assert.Fail("bytes allocated per round, per attempt: " + string.Join(" | ", seen));
    }
}

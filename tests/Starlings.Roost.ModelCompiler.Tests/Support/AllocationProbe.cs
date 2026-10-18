namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// Measures bytes allocated by the current thread across calls of a step (S4's method). The first rounds can include
/// a one-off allocation when the runtime promotes a hot method to its optimised tier; a step that allocates per call
/// allocates in every round, so the steady-state claim is that the smallest round allocates nothing.
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
}

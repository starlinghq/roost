using Starlings.Roost.Kernels;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Qwen35;
using Starlings.Roost.Worker.Weights;
using Starlings.Roost.WorkerHost;

// roost-worker (R-062): the packaged GPU worker process. Starlings.Generation.Roost starts it with the arguments of
// WorkerArguments followed by WorkerEntry.SegmentFlag and the segment path; the worker opens the segment, loads the kernel library
// for its device's SM and the locked weights, and serves steps until the host shuts it down (contract C8, decision 21). The
// consumer's process never loads CUDA or the kernel library.
if (!WorkerEntry.IsWorkerInvocation(args))
{
    Console.Error.WriteLine("roost-worker is started by Starlings.Generation.Roost (RoostChatClient); it is not run by hand.");
    return 2;
}

return WorkerEntry.Run(args, Executor.Create);

/// <summary>Makes the step executor a launch asks for.</summary>
internal static class Executor
{
    /// <summary>The kernel manifest shipped beside the executable in the runtime package.</summary>
    internal const string ManifestFileName = "roost-kernels.manifest.toml";

    /// <summary>Builds the eager executor for the launch on the command line.</summary>
    /// <param name="args">The worker's command line.</param>
    /// <returns>The executor.</returns>
    internal static IStepExecutor Create(string[] args)
    {
        var launch = WorkerArguments.Parse(args);
        IEagerModel model = launch.Model == Qwen35EagerModel.ModelId
            ? new Qwen35EagerModel()
            : throw new ArgumentException($"roost-worker: no binding for model '{launch.Model}' (this worker serves {Qwen35EagerModel.ModelId})");
        string here = AppContext.BaseDirectory;
        var manifest = KernelManifest.Load(launch.ManifestPath ?? Path.Combine(here, ManifestFileName));
        var options = new EagerWorkerOptions
        {
            Device = launch.Device,
            MaxTokens = launch.MaxTokens,
            MaxSequences = launch.MaxSequences,
            MaxKvBlocks = launch.MaxKvBlocks,
            AttentionBlocks = launch.AttentionBlocks,
            RecurrentSlots = launch.RecurrentSlots,
            KernelDirectory = launch.KernelDirectory ?? here,
            AllowUntested = launch.AllowUntested,
        };
        var locked = new LockedFile(Path.GetFileName(launch.WeightsPath), launch.WeightsSize, launch.WeightsSha256);
        return new EagerStepExecutor(model, options, manifest, launch.WeightsPath, locked);
    }
}

using System.Collections.Immutable;
using System.Globalization;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// What a packaged worker executable (<c>roost-worker</c>, R-062) needs to serve a model: which model, the weights file and its
/// locked size and digest, the kernel manifest and library directory, and the step and pool capacities. The host writes it as
/// arguments before <see cref="WorkerEntry.SegmentFlag"/>, and the worker parses them back. Both sides take the capacities from
/// the same values, so the plans the host writes always fit what the worker allocated.
/// </summary>
public sealed record WorkerArguments
{
    private const string ModelFlag = "--model";
    private const string WeightsFlag = "--weights";
    private const string WeightsSizeFlag = "--weights-size";
    private const string WeightsSha256Flag = "--weights-sha256";
    private const string ManifestFlag = "--kernel-manifest";
    private const string KernelsFlag = "--kernels";
    private const string AllowUntestedFlag = "--allow-untested";
    private const string DeviceFlag = "--device";
    private const string MaxTokensFlag = "--max-tokens";
    private const string MaxSequencesFlag = "--max-sequences";
    private const string MaxKvBlocksFlag = "--max-kv-blocks";
    private const string AttentionBlocksFlag = "--attention-blocks";
    private const string RecurrentSlotsFlag = "--recurrent-slots";

    /// <summary>The model id the worker binds (the <c>RoostModel</c> name, for example <c>qwen3.5-0.8b</c>).</summary>
    public required string Model { get; init; }

    /// <summary>The safetensors weights file.</summary>
    public required string WeightsPath { get; init; }

    /// <summary>The weights file's locked size in bytes.</summary>
    public required long WeightsSize { get; init; }

    /// <summary>The weights file's locked SHA-256, lowercase hex; the worker verifies it as it reads (R-024).</summary>
    public required string WeightsSha256 { get; init; }

    /// <summary>The kernel manifest (<c>kernels/manifest.toml</c>); null means the one shipped beside the worker executable.</summary>
    public string? ManifestPath { get; init; }

    /// <summary>The directory holding <c>libroost_kernels_sm&lt;N&gt;.so</c>; null means the worker executable's directory.</summary>
    public string? KernelDirectory { get; init; }

    /// <summary>Whether an SM whose manifest status is <c>untested</c> may load.</summary>
    public bool AllowUntested { get; init; }

    /// <summary>CUDA ordinal after <c>CUDA_VISIBLE_DEVICES</c>.</summary>
    public int Device { get; init; }

    /// <summary>Largest token count of a step.</summary>
    public required int MaxTokens { get; init; }

    /// <summary>Largest sequence count of a step.</summary>
    public required int MaxSequences { get; init; }

    /// <summary>Largest total of attention block ids in a step.</summary>
    public required int MaxKvBlocks { get; init; }

    /// <summary>Attention pool capacity in blocks.</summary>
    public required int AttentionBlocks { get; init; }

    /// <summary>Recurrent pool capacity in slots.</summary>
    public required int RecurrentSlots { get; init; }

    /// <summary>The arguments that carry this launch, in a fixed order.</summary>
    /// <returns>The arguments.</returns>
    public ImmutableArray<string> ToArguments()
    {
        var a = ImmutableArray.CreateBuilder<string>();
        void Add(string flag, string value)
        {
            a.Add(flag);
            a.Add(value);
        }

        Add(ModelFlag, Model);
        Add(WeightsFlag, WeightsPath);
        Add(WeightsSizeFlag, WeightsSize.ToString(CultureInfo.InvariantCulture));
        Add(WeightsSha256Flag, WeightsSha256);
        if (ManifestPath is not null)
        {
            Add(ManifestFlag, ManifestPath);
        }

        if (KernelDirectory is not null)
        {
            Add(KernelsFlag, KernelDirectory);
        }

        if (AllowUntested)
        {
            a.Add(AllowUntestedFlag);
        }

        Add(DeviceFlag, Device.ToString(CultureInfo.InvariantCulture));
        Add(MaxTokensFlag, MaxTokens.ToString(CultureInfo.InvariantCulture));
        Add(MaxSequencesFlag, MaxSequences.ToString(CultureInfo.InvariantCulture));
        Add(MaxKvBlocksFlag, MaxKvBlocks.ToString(CultureInfo.InvariantCulture));
        Add(AttentionBlocksFlag, AttentionBlocks.ToString(CultureInfo.InvariantCulture));
        Add(RecurrentSlotsFlag, RecurrentSlots.ToString(CultureInfo.InvariantCulture));
        return a.ToImmutable();
    }

    /// <summary>Reads a launch from a worker command line; arguments it does not know (such as the segment flag) are skipped.</summary>
    /// <param name="args">The command line.</param>
    /// <returns>The launch.</returns>
    /// <exception cref="ArgumentException">A required argument is missing, repeated or malformed.</exception>
    public static WorkerArguments Parse(IReadOnlyList<string> args)
    {
        ArgumentNullException.ThrowIfNull(args);
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        bool allowUntested = false;
        string[] valued =
        [
            ModelFlag, WeightsFlag, WeightsSizeFlag, WeightsSha256Flag, ManifestFlag, KernelsFlag, DeviceFlag, MaxTokensFlag,
            MaxSequencesFlag, MaxKvBlocksFlag, AttentionBlocksFlag, RecurrentSlotsFlag,
        ];
        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            if (arg == AllowUntestedFlag)
            {
                allowUntested = true;
            }
            else if (Array.IndexOf(valued, arg) >= 0)
            {
                if (i + 1 >= args.Count)
                {
                    throw new ArgumentException($"roost worker: {arg} needs a value", nameof(args));
                }

                if (!values.TryAdd(arg, args[++i]))
                {
                    throw new ArgumentException($"roost worker: {arg} is given twice", nameof(args));
                }
            }
        }

        string Required(string flag) =>
            values.TryGetValue(flag, out var v) && v.Length > 0 ? v : throw new ArgumentException($"roost worker: missing {flag}", nameof(args));
        long Long(string flag) =>
            long.TryParse(Required(flag), NumberStyles.None, CultureInfo.InvariantCulture, out var v) ? v : throw new ArgumentException($"roost worker: {flag} is not a non-negative integer", nameof(args));
        int Int(string flag) =>
            int.TryParse(Required(flag), NumberStyles.None, CultureInfo.InvariantCulture, out var v) ? v : throw new ArgumentException($"roost worker: {flag} is not a non-negative integer", nameof(args));

        string sha = Required(WeightsSha256Flag);
        if (sha.Length != 64 || !sha.All(char.IsAsciiHexDigitLower))
        {
            throw new ArgumentException($"roost worker: {WeightsSha256Flag} is not 64 lowercase hex digits", nameof(args));
        }

        return new WorkerArguments
        {
            Model = Required(ModelFlag),
            WeightsPath = Required(WeightsFlag),
            WeightsSize = Long(WeightsSizeFlag),
            WeightsSha256 = sha,
            ManifestPath = values.GetValueOrDefault(ManifestFlag),
            KernelDirectory = values.GetValueOrDefault(KernelsFlag),
            AllowUntested = allowUntested,
            Device = values.ContainsKey(DeviceFlag) ? Int(DeviceFlag) : 0,
            MaxTokens = Int(MaxTokensFlag),
            MaxSequences = Int(MaxSequencesFlag),
            MaxKvBlocks = Int(MaxKvBlocksFlag),
            AttentionBlocks = Int(AttentionBlocksFlag),
            RecurrentSlots = Int(RecurrentSlotsFlag),
        };
    }
}

using System.Text.Json;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Requests;
using Starlings.Roost.Worker.Weights;

// R-024 GPU evidence: the locked Qwen3.5-0.8B checkpoint streamed to the device in slot order and verified as read.
//   usage: weights-check <checkpoint-file> <lock.toml> [--expect-mismatch]
// With --expect-mismatch the file is a substituted copy: the run passes only if the load fails with AssetMismatch,
// the arena is discarded and no address is handed out. Everything here returns statuses or throws managed exceptions;
// nothing raises a signal. Run it through tools/gpu-lease so CUDA_VISIBLE_DEVICES selects the leased device (ordinal 0).
if (args.Length < 2)
{
    Console.Error.WriteLine("usage: weights-check <checkpoint-file> <lock.toml> [--expect-mismatch]");
    return 2;
}

return Check.Run(args[0], args[1], args.Contains("--expect-mismatch"));

internal static unsafe class Check
{
    private const string LockedName = "model.safetensors-00001-of-00001.safetensors";
    private const string Prefix = "model.language_model.";

    private static int _failures;

    public static int Run(string file, string lockPath, bool expectMismatch)
    {
        CudaLibraryResolver.Install(13);
        int runtime, count;
        CudaCheck.Runtime(CudaRuntime.cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        CudaCheck.Runtime(CudaRuntime.cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        Console.WriteLine($"cudart {runtime}, {count} visible device(s), device 0 is SM {CudaCheck.ComputeCapability(0)}");

        var lockFile = WeightLock.Load(lockPath);
        var locked = lockFile.Get(LockedName);
        Console.WriteLine($"lock: {lockFile.Repo} @ {lockFile.Revision}; {locked.Path} {locked.Size} bytes sha256 {locked.Sha256}");

        var header = ReadHeader(file);
        var slots = BuildSlots(header);
        Console.WriteLine($"{slots.Count} slots (embed_tokens, 24 layers, final norm, tied lm_head), {slots.Sum(static s => s.Bytes) / 1e6:F1} MB of weights");

        using var arena = new CudaWeightArena(0);
        if (expectMismatch)
        {
            return ExpectMismatch(file, locked, slots, arena);
        }

        var order = new List<string>();
        var layout = WeightStreamer.Load(file, locked, slots, arena, slotCopied: s => order.Add(s.Name));
        var seconds = layout.Elapsed.TotalSeconds;
        Console.WriteLine($"VERIFIED sha256 {layout.Sha256}");
        Console.WriteLine($"streamed {layout.BytesRead / 1e6:F1} MB read+hashed, {layout.TotalBytes / 1e6:F1} MB arena, {seconds:F3} s, {layout.BytesRead / 1e6 / seconds:F0} MB/s");

        // Every slot's device bytes equal the file bytes at the tensor's offset.
        using var fs = File.OpenRead(file);
        long checkedBytes = 0;
        foreach (var p in layout.Slots)
        {
            var t = header[slots[p.Index].Source];
            var expected = new byte[p.Bytes];
            fs.Seek(t.DataStart + t.Begin, SeekOrigin.Begin);
            fs.ReadExactly(expected);
            var actual = new byte[p.Bytes];
            arena.CopyToHost(p, actual);
            Expect(expected.AsSpan().SequenceEqual(actual), $"slot {p.Index} {p.Name} equals file bytes");
            Expect(p.Offset % WeightLayout.Alignment == 0, $"slot {p.Name} aligned");
            checkedBytes += p.Bytes;
        }

        Console.WriteLine($"slot bytes compared with the file: {checkedBytes} of {slots.Sum(static s => s.Bytes)}");
        var ready = 0;
        while (ready < order.Count && order[ready] == slots[ready].Name)
        {
            ready++;
        }

        Console.WriteLine($"{order.Count} slots copied in file order; first {ready} also in slot order (first copied: {order[0]})");
        Console.WriteLine(_failures == 0 ? "PASS" : $"FAIL ({_failures})");
        return _failures == 0 ? 0 : 1;
    }

    private static int ExpectMismatch(string file, LockedFile locked, List<WeightSlotSpec> slots, CudaWeightArena arena)
    {
        try
        {
            _ = WeightStreamer.Load(file, locked, slots, arena);
            Console.WriteLine("FAIL: a substituted file was accepted");
            return 1;
        }
        catch (WeightLoadException e)
        {
            Console.WriteLine($"refused: {e.Error.CodeName}: {e.Message}");
            Expect(e.Error.Code == RoostErrorCode.AssetMismatch, "code is AssetMismatch");
            try
            {
                _ = arena.Span(new SlotPlacement(0, "x", 0, 2, DType.BF16, 1));
                Expect(false, "no device address after a refusal");
            }
            catch (InvalidOperationException)
            {
                Expect(true, "no device address after a refusal");
            }

            Console.WriteLine(_failures == 0 ? "PASS" : $"FAIL ({_failures})");
            return _failures == 0 ? 0 : 1;
        }
    }

    private static void Expect(bool ok, string what)
    {
        if (!ok)
        {
            _failures++;
            Console.WriteLine($"FAIL: {what}");
        }
    }

    private readonly record struct Entry(DType Dtype, long Elements, long Begin, long DataStart);

    private static Dictionary<string, Entry> ReadHeader(string file)
    {
        using var fs = File.OpenRead(file);
        var prefix = new byte[8];
        fs.ReadExactly(prefix);
        var n = (int)BitConverter.ToUInt64(prefix);
        var json = new byte[n];
        fs.ReadExactly(json);
        using var doc = JsonDocument.Parse(json);
        var map = new Dictionary<string, Entry>();
        foreach (var p in doc.RootElement.EnumerateObject().Where(static p => p.Name != "__metadata__"))
        {
            var dtype = p.Value.GetProperty("dtype").GetString() switch { "BF16" => DType.BF16, "F32" => DType.F32, var d => throw new InvalidOperationException(d) };
            long elements = 1;
            foreach (var d in p.Value.GetProperty("shape").EnumerateArray())
            {
                elements *= d.GetInt64();
            }

            map[p.Name] = new Entry(dtype, elements, p.Value.GetProperty("data_offsets")[0].GetInt64(), 8 + n);
        }

        return map;
    }

    /// <summary>The model's slot order, as the generated loader would give it: embedding, each layer's tensors, final norm, lm_head.</summary>
    private static List<WeightSlotSpec> BuildSlots(Dictionary<string, Entry> header)
    {
        var slots = new List<WeightSlotSpec>();
        void Add(string source)
        {
            var e = header[source];
            slots.Add(new WeightSlotSpec(source, source, e.Dtype, e.Elements));
        }

        Add(Prefix + "embed_tokens.weight");
        for (var layer = 0; layer < 24; layer++)
        {
            foreach (var name in header.Keys.Where(k => k.StartsWith($"{Prefix}layers.{layer}.", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
            {
                Add(name);
            }
        }

        Add(Prefix + "norm.weight");
        var embed = header[Prefix + "embed_tokens.weight"]; // tie_word_embeddings is true: lm_head is the embedding matrix
        slots.Add(new WeightSlotSpec("lm_head.weight", Prefix + "embed_tokens.weight", embed.Dtype, embed.Elements));
        return slots;
    }
}

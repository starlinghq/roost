using System.Text;
using System.Text.Json;
using Starlings.Roost.Abi;
using Starlings.Roost.Native.Tests.Support;

namespace Starlings.Roost.Native.Tests;

/// <summary>
/// Soundness of the CPU sampler path with grammar masks (R-051): a masked token is never sampled, whatever its logit, and every
/// output of the tiny models under a JSON schema parses and satisfies the schema, greedy and sampled.
/// </summary>
public sealed unsafe class SamplerSoundnessTests
{
    private const int Budget = 400;

    public static TheoryData<TinyKind, string> ModelsAndSchemas()
    {
        var data = new TheoryData<TinyKind, string>();
        foreach (var kind in new[] { TinyKind.Dense, TinyKind.Moe })
        {
            foreach (var schema in new[] { "s1-flat", "s2-nested", "s3-refs" })
            {
                data.Add(kind, schema);
            }
        }

        return data;
    }

    [Fact]
    public void AMaskedTokenIsNeverSampledEvenWithTheLargestLogit()
    {
        var tokenizer = Grammar.Require();
        var schema = File.ReadAllText(SchemaPath("s2-nested"));
        var rng = new Random(20261102);
        foreach (var temperature in new[] { 0f, 1f, 4f })
        {
            using var slots = new GrammarSlots(1, tokenizer.VocabSize);
            int slot = slots.Assign(tokenizer.CreateMatcher(GrammarSpec.JsonSchema(schema)));
            var masked = 0;
            var stopped = false;
            for (var step = 0; step < Budget && !stopped; step++)
            {
                slots.ComputeMasks([slot]);
                var mask = slots.Mask(slot).ToArray();

                // Adversarial logits: every token the grammar forbids outscores every token it allows.
                var logits = new float[tokenizer.VocabSize];
                for (var v = 0; v < logits.Length; v++)
                {
                    bool allowed = Grammar.Allowed(mask, v);
                    logits[v] = allowed ? (float)rng.NextDouble() : 50f + (float)rng.NextDouble();
                    masked += allowed ? 0 : 1;
                }

                // Row 1 has no grammar: it must still pick a forbidden token (the mask is per row).
                var ids = Sample([logits, logits], [slot, -1], slots, temperature, 7, (ulong)step);
                Assert.True(Grammar.Allowed(mask, ids[0]), $"step {step}: token {ids[0]} was masked");
                Assert.False(Grammar.Allowed(mask, ids[1]));
                stopped = ids[0] == Grammar.StopId;
            }

            Assert.True(stopped, $"temperature {temperature}: no stop within {Budget} tokens");
            Assert.True(masked > 0);
        }
    }

    [Theory]
    [MemberData(nameof(ModelsAndSchemas))]
    public void EveryOutputOfTheTinyModelsParsesAndSatisfiesTheSchema(TinyKind kind, string name)
    {
        var tokenizer = Grammar.Require();
        var schemaText = File.ReadAllText(SchemaPath(name));
        using var schema = JsonDocument.Parse(schemaText);
        var prompt = Encoding.ASCII.GetBytes("Reply with JSON: ").Select(b => (int)b).ToArray();
        var outputs = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (temperature, seed) in new[] { (0f, 0UL), (1f, 1UL), (1f, 2UL), (1.5f, 3UL) })
        {
            using var model = new TinyModel(kind);
            using var slots = new GrammarSlots(1, tokenizer.VocabSize);
            int slot = slots.Assign(tokenizer.CreateMatcher(GrammarSpec.JsonSchema(schemaText)));
            var bytes = new List<byte>();
            var logits = model.Step(prompt);
            var stopped = false;
            for (var step = 0; step < Budget; step++)
            {
                slots.ComputeMasks([slot]);
                var mask = slots.Mask(slot).ToArray();
                int id = Sample([logits], [slot], slots, temperature, seed, (ulong)step)[0];
                Assert.True(Grammar.Allowed(mask, id), $"step {step}: token {id} was masked");
                if (id == Grammar.StopId)
                {
                    Assert.True(slots.Get(slot).IsAccepting);
                    stopped = true;
                    break;
                }

                Assert.InRange(id, 0, TinyModel.ModelVocab - 1); // the merged ids are never chosen (logit -inf)
                bytes.Add((byte)id); // the tiny tokenizer's ids below 256 are the byte values
                logits = model.Step([id]);
            }

            var text = Encoding.UTF8.GetString([.. bytes]);
            Assert.True(stopped, $"{kind} {name} T={temperature}: no stop within {Budget} tokens: {text}");
            using var output = JsonDocument.Parse(text);
            Assert.Null(SchemaCheck.Violation(output.RootElement, schema.RootElement));
            outputs.Add(text);
            TestContext.Current.TestOutputHelper?.WriteLine($"{kind} {name} T={temperature} seed={seed}: {text}");
        }

        Assert.True(outputs.Count >= 2, "sampling at different seeds produced one output only: the test would not probe the masks");
    }

    [Fact]
    public void TheSamplerRefusesAPresetMaskAndAVocabularyMismatch()
    {
        var tokenizer = Grammar.Require();
        using var slots = new GrammarSlots(1, tokenizer.VocabSize);
        _ = slots.Assign(tokenizer.CreateMatcher(GrammarSpec.JsonObject));
        var logits = new float[tokenizer.VocabSize - 1];
        _ = Assert.Throws<ArgumentException>(() => Sample([logits], [0], slots, 0f, 0, 0));
    }

    private static string SchemaPath(string name) => Path.Combine(AppContext.BaseDirectory, "Schemas", name + ".json");

    /// <summary>One <see cref="GrammarSampler.Sample"/> call over the rows, all with the same temperature, seed and offset.</summary>
    private static int[] Sample(float[][] rows, int[] rowSlots, GrammarSlots slots, float temperature, ulong seed, ulong offset)
    {
        int n = rows.Length;
        int vocab = rows[0].Length;
        var logits = rows.SelectMany(r => r).ToArray();
        var temperatures = Enumerable.Repeat(temperature, n).ToArray();
        var topK = new int[n];
        var topP = Enumerable.Repeat(1f, n).ToArray();
        var minP = new float[n];
        var seeds = Enumerable.Repeat(seed, n).ToArray();
        var offsets = Enumerable.Repeat(offset, n).ToArray();
        var ids = new int[n];
        fixed (float* l = logits)
        fixed (float* t = temperatures)
        fixed (int* k = topK)
        fixed (float* p = topP)
        fixed (float* m = minP)
        fixed (ulong* s = seeds)
        fixed (ulong* o = offsets)
        fixed (int* i = ids)
        {
            var args = new Ops.SampleArgs
            {
                Logits = Span(l, logits.Length, DType.F32),
                Temperature = Span(t, n, DType.F32),
                TopK = Span(k, n, DType.I32),
                TopP = Span(p, n, DType.F32),
                MinP = Span(m, n, DType.F32),
                Seeds = Span(s, n, DType.U64),
                Offsets = Span(o, n, DType.U64),
                Ids = Span(i, n, DType.I32),
                Rows = n,
                Vocab = vocab,
                Stride = vocab,
            };
            GrammarSampler.Sample(in args, rowSlots, slots);
        }

        return ids;
    }

    private static DeviceSpan Span(void* address, int length, DType dtype) => new() { Address = (nint)address, Length = (ulong)length, Dtype = dtype };
}

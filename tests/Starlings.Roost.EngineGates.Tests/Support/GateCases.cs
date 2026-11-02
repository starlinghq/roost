using System.Collections.Immutable;
using Starlings.Tokenizers;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>One gate case: a raw text prompt, its ids under the tiny tokenizer, and how many tokens to generate.</summary>
/// <param name="Id">Case id (letters, digits, '.', '_' and '-', as the evaluator's reference requires).</param>
/// <param name="Prompt">The prompt text; rendered as given (raw completion, no chat template).</param>
/// <param name="PromptIds">The prompt's ids.</param>
/// <param name="MaxTokens">Tokens to generate (both tiny models have no stop token in their 256-id vocabulary, so every case ends at the length limit).</param>
internal sealed record GateCase(string Id, string Prompt, ImmutableArray<int> PromptIds, int MaxTokens);

/// <summary>
/// The gate cases and the tokenizer they are encoded with: the 4 KB byte-level BPE tokenizer of the generation layer's tests
/// (ids 0-255 are bytes, 256-262 merges, 263 <c>&lt;|im_end|&gt;</c>). The tiny models have 256 ids, so every prompt is chosen
/// to encode inside 0-255 (no merge pairs), which a check enforces.
/// </summary>
internal static class GateCases
{
    /// <summary>The 32-character prefix the two shared-prefix cases have in common: two attention blocks of 16.</summary>
    public const string SharedPrefix = "Roost engine gates, case prefix:";

    private static readonly Lazy<Tokenizer> s_tokenizer = new(() =>
        Tokenizer.FromFile(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json")));

    private static readonly Lazy<ImmutableArray<GateCase>> s_all = new(() =>
    [
        Make("short", "Hello, roost!", 6),
        Make("digits", "0123456789 9876543210 ABC", 8),
        Make("shared-a", SharedPrefix + " alpha bravo", 8),
        Make("shared-b", SharedPrefix + " zulu yankee x-ray", 8),
        Make("long", "Prefill a prompt longer as one block, so chunks cross block boundaries: abcdefghij", 8),
    ]);

    /// <summary>The tokenizer.</summary>
    public static Tokenizer Tokenizer => s_tokenizer.Value;

    /// <summary>Every case, in a fixed order.</summary>
    public static ImmutableArray<GateCase> All => s_all.Value;

    /// <summary>The case the cached-vs-uncached section reports: its prefix comes from the cache after <see cref="Primer"/> ran.</summary>
    public static GateCase Cached => All.Single(c => c.Id == "shared-b");

    /// <summary>The request that fills the cache before <see cref="Cached"/>.</summary>
    public static GateCase Primer => All.Single(c => c.Id == "shared-a");

    /// <summary>A case by id.</summary>
    /// <param name="id">The id.</param>
    /// <returns>The case.</returns>
    public static GateCase Get(string id) => All.Single(c => c.Id == id);

    private static GateCase Make(string id, string prompt, int maxTokens)
    {
        ImmutableArray<int> ids = [.. Tokenizer.Encode(prompt, false).Ids];
        if (ids.Any(t => t >= TinyCpuModel.Vocab))
        {
            throw new InvalidOperationException($"case {id}: the prompt encodes to an id outside the tiny models' vocabulary");
        }

        return new GateCase(id, prompt, ids, maxTokens);
    }
}

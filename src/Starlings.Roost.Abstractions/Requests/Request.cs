using System.Collections.Immutable;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Requests;

/// <summary>Engine-wide request id.</summary>
/// <param name="Value">The id.</param>
public readonly record struct RequestId(ulong Value);

/// <summary>Handle of a compiled grammar or response format (llguidance matcher factory, R-051).</summary>
/// <param name="Value">The handle; 0 means none.</param>
public readonly record struct GrammarHandle(ulong Value);

/// <summary>Handle of a loaded LoRA adapter (M3).</summary>
/// <param name="Value">The handle; 0 means the base model.</param>
public readonly record struct LoraHandle(ulong Value);

/// <summary>Sampling parameters (OpenAI and vLLM names in the docs).</summary>
public sealed record SamplingParameters
{
    /// <summary>Default: temperature 1, nothing else.</summary>
    public static SamplingParameters Default { get; } = new();

    /// <summary><c>temperature</c>; 0 means greedy.</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary><c>top_p</c>; 1 disables.</summary>
    public float TopP { get; init; } = 1f;

    /// <summary><c>top_k</c>; 0 disables.</summary>
    public int TopK { get; init; }

    /// <summary><c>min_p</c>; 0 disables.</summary>
    public float MinP { get; init; }

    /// <summary><c>repetition_penalty</c>; 1 disables.</summary>
    public float RepetitionPenalty { get; init; } = 1f;

    /// <summary><c>presence_penalty</c>.</summary>
    public float PresencePenalty { get; init; }

    /// <summary><c>frequency_penalty</c>.</summary>
    public float FrequencyPenalty { get; init; }

    /// <summary><c>seed</c>; null for an engine-chosen seed.</summary>
    public ulong? Seed { get; init; }

    /// <summary><c>max_tokens</c>; null for the profile limit.</summary>
    public int? MaxTokens { get; init; }

    /// <summary><c>min_tokens</c>: stop ids are ignored until this many tokens.</summary>
    public int MinTokens { get; init; }

    /// <summary><c>n</c>: sequences returned.</summary>
    public int N { get; init; } = 1;

    /// <summary><c>best_of</c>: sequences generated; null means <see cref="N"/>.</summary>
    public int? BestOf { get; init; }

    /// <summary>Beam width; 1 disables beam search.</summary>
    public int BeamWidth { get; init; } = 1;

    /// <summary><c>logprobs</c> requested.</summary>
    public bool Logprobs { get; init; }

    /// <summary><c>top_logprobs</c>: alternatives per token.</summary>
    public int TopLogprobs { get; init; }
}

/// <summary>Speculation modes a request may ask for; the profile decides what is available.</summary>
public enum SpeculationMode
{
    /// <summary>The profile's default.</summary>
    Default = 0,

    /// <summary>No speculation for this request.</summary>
    Disabled,

    /// <summary>N-gram (prompt lookup) drafts.</summary>
    Ngram,

    /// <summary>A draft model.</summary>
    DraftModel,

    /// <summary>A draft head (EAGLE-style).</summary>
    DraftHead,
}

/// <summary>Per-request speculation policy.</summary>
/// <param name="Mode">The mode.</param>
/// <param name="MaxDraftTokens">Upper bound on drafts per step; 0 for the profile default.</param>
public sealed record SpeculationPolicy(SpeculationMode Mode, int MaxDraftTokens)
{
    /// <summary>The profile's default.</summary>
    public static SpeculationPolicy Default { get; } = new(SpeculationMode.Default, 0);
}

/// <summary>Kinds of multimodal part.</summary>
public enum MultimodalKind
{
    /// <summary>An image.</summary>
    Image = 0,

    /// <summary>Audio.</summary>
    Audio,

    /// <summary>Video.</summary>
    Video,

    /// <summary>Precomputed embeddings.</summary>
    Embeddings,
}

/// <summary>A non-text input placed at a token position (M5).</summary>
/// <param name="Kind">The kind.</param>
/// <param name="MediaType">MIME type of <paramref name="Content"/>.</param>
/// <param name="Content">The bytes.</param>
/// <param name="TokenOffset">Prompt position of its placeholder tokens.</param>
public sealed record MultimodalPart(MultimodalKind Kind, string MediaType, ReadOnlyMemory<byte> Content, int TokenOffset);

/// <summary>A Kev decision question: the options the pointer head chooses between (R-086).</summary>
/// <param name="QuestionId">Caller's id, echoed in events.</param>
/// <param name="OptionTokenOffsets">Prompt token offsets of each option's anchor token.</param>
public sealed record DecisionQuestion(string QuestionId, ImmutableArray<int> OptionTokenOffsets);

/// <summary>A pooling request: the row ends in a head instead of the sampler (R-052).</summary>
/// <param name="Head">Pooling or classification head.</param>
/// <param name="Normalize">L2-normalise the output.</param>
public sealed record PoolingInput(HeadKind Head, bool Normalize);

/// <summary>Request priority classes; higher is scheduled first under pressure.</summary>
public enum RequestPriority
{
    /// <summary>Batch work.</summary>
    Low = -1,

    /// <summary>Default.</summary>
    Normal = 0,

    /// <summary>Interactive.</summary>
    High = 1,
}

/// <summary>
/// A request as the engine sees it (contract C6), after the frontend has tokenised and validated it against the
/// profile's capability table.
/// </summary>
public sealed record Request
{
    /// <summary>The id.</summary>
    public required RequestId Id { get; init; }

    /// <summary>Prompt token ids.</summary>
    public required ImmutableArray<int> PromptTokenIds { get; init; }

    /// <summary>Sampling parameters.</summary>
    public SamplingParameters Sampling { get; init; } = SamplingParameters.Default;

    /// <summary>
    /// Stop token ids: the model's generation stop ids plus the request's. The grammar matcher must be given the same
    /// ids (S10: the tokenizer's EOS is not Qwen3's generation stop id).
    /// </summary>
    public ImmutableArray<int> StopTokenIds { get; init; } = [];

    /// <summary>Stop strings, matched by the host generation layer (B1), never by the worker.</summary>
    public ImmutableArray<string> StopSequences { get; init; } = [];

    /// <summary>Grammar for structured output; default for none.</summary>
    public GrammarHandle Grammar { get; init; }

    /// <summary>Response format handle (JSON schema or JSON object), compiled to a grammar; default for none.</summary>
    public GrammarHandle ResponseFormat { get; init; }

    /// <summary>Tool-call parser id, or null.</summary>
    public string? ToolParserId { get; init; }

    /// <summary>Reasoning parser id, or null.</summary>
    public string? ReasoningParserId { get; init; }

    /// <summary>LoRA adapter; default for the base model.</summary>
    public LoraHandle Lora { get; init; }

    /// <summary>Priority.</summary>
    public RequestPriority Priority { get; init; } = RequestPriority.Normal;

    /// <summary>Deadline after which the request is cancelled; null for none.</summary>
    public DateTimeOffset? Deadline { get; init; }

    /// <summary>Speculation policy.</summary>
    public SpeculationPolicy Speculation { get; init; } = SpeculationPolicy.Default;

    /// <summary>Multimodal parts (M5).</summary>
    public ImmutableArray<MultimodalPart> MultimodalParts { get; init; } = [];

    /// <summary>Kev decision question, or null.</summary>
    public DecisionQuestion? Decision { get; init; }

    /// <summary>Pooling input, or null for generation.</summary>
    public PoolingInput? Pooling { get; init; }
}

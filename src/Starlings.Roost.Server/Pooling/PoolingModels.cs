namespace Starlings.Roost.Server.Pooling;

/// <summary>
/// A text embedding model behind <c>POST /v1/embeddings</c> (R-052). The server's implementations wrap Starling's ONNX packages
/// (<see cref="E5EmbeddingModel"/>, <see cref="EmbeddingGemma2EmbeddingModel"/>); tests pass fakes. Implementations must be safe for concurrent calls.
/// </summary>
public interface IRoostEmbeddingModel
{
    /// <summary>The model id requests name in <c>model</c> and responses report.</summary>
    string ModelId { get; }

    /// <summary>The length of every vector the model returns.</summary>
    int Dimensions { get; }

    /// <summary>Embeds texts in order, exactly as given: no prefix or prompt is added.</summary>
    /// <param name="inputs">The texts; at least one.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>One vector per input, in input order, and the token count of the whole batch.</returns>
    /// <exception cref="Starlings.Generation.Roost.RoostChatException">An input the model refuses (for example, too long).</exception>
    RoostEmbeddingResult Embed(IReadOnlyList<string> inputs, CancellationToken cancellationToken);
}

/// <summary>The vectors of one embedding call.</summary>
/// <param name="Embeddings">One vector of <see cref="IRoostEmbeddingModel.Dimensions"/> values per input, in input order.</param>
/// <param name="PromptTokens">Tokens of every input together, special tokens included.</param>
public sealed record RoostEmbeddingResult(IReadOnlyList<float[]> Embeddings, int PromptTokens);

/// <summary>
/// An entity extraction model behind <c>POST /v1/extract</c> (R-052): named labels in, labelled spans out. The server's
/// implementation wraps Starling's GLiNER2 package (<see cref="Gliner2ExtractionModel"/>); tests pass fakes. Implementations must
/// be safe for concurrent calls.
/// </summary>
public interface IRoostExtractionModel
{
    /// <summary>The model id requests name in <c>model</c> and responses report.</summary>
    string ModelId { get; }

    /// <summary>Extracts entities of the given labels from each text.</summary>
    /// <param name="inputs">The texts; at least one.</param>
    /// <param name="labels">The entity labels, in the order results are grouped; at least one, no duplicates.</param>
    /// <param name="threshold">The confidence threshold in [0, 1], or null for the model's default.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>One result per input, in input order.</returns>
    /// <exception cref="Starlings.Generation.Roost.RoostChatException">An input the model refuses (for example, too long).</exception>
    IReadOnlyList<RoostExtraction> Extract(IReadOnlyList<string> inputs, IReadOnlyList<string> labels, float? threshold, CancellationToken cancellationToken);
}

/// <summary>The entities found in one text.</summary>
/// <param name="Entities">The entities, in the model's order (labels in request order, then descending confidence).</param>
/// <param name="TokenCount">The tokens the model read for this text.</param>
public sealed record RoostExtraction(IReadOnlyList<RoostEntity> Entities, int TokenCount);

/// <summary>One labelled span.</summary>
/// <param name="Label">The label, one of the request's.</param>
/// <param name="Text">The span's text, exactly the input between <paramref name="Start"/> and <paramref name="End"/>.</param>
/// <param name="Start">The start offset in Unicode scalar values (Python string indices), inclusive.</param>
/// <param name="End">The end offset in Unicode scalar values, exclusive.</param>
/// <param name="Confidence">The model's confidence in [0, 1].</param>
public sealed record RoostEntity(string Label, string Text, int Start, int End, float Confidence);

/// <summary>
/// The pooling models a <see cref="RoostServer"/> serves (R-052). The caller owns them and disposes them after the server stops.
/// A route whose model is not configured answers <c>profile_unsupported</c>.
/// </summary>
public sealed record RoostPoolingModels
{
    /// <summary>No pooling model: <c>/v1/embeddings</c> and <c>/v1/extract</c> answer <c>profile_unsupported</c>.</summary>
    public static RoostPoolingModels None { get; } = new();

    /// <summary>The embedding models, selected by the request's <c>model</c>; ids must be distinct.</summary>
    public IReadOnlyList<IRoostEmbeddingModel> Embeddings { get; init; } = [];

    /// <summary>The extraction models, selected by the request's <c>model</c>; ids must be distinct.</summary>
    public IReadOnlyList<IRoostExtractionModel> Extraction { get; init; } = [];
}

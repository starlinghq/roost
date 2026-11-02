using Starlings.Acquisition;
using Starlings.Generation.Roost;
using Starlings.Models.E5;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.Gliner2;
using Starlings.Roost.Requests;
using Starlings.Runtime.Onnx;

namespace Starlings.Roost.Server.Pooling;

/// <summary>
/// Starling's multilingual-e5-small embedder (<c>Starlings.Models.E5</c> 0.3.0) on ONNX Runtime's CPU execution provider, in
/// the server process (decision 30): 384-dimensional unit vectors, mean pooling, inputs truncated on the right to 512 tokens by
/// the package. The caller adds e5's <c>"query: "</c> or <c>"passage: "</c> prefix; nothing is inserted.
/// </summary>
public sealed class E5EmbeddingModel : IRoostEmbeddingModel, IDisposable
{
    /// <summary>The vector length of the profile.</summary>
    public const int VectorLength = 384;

    private readonly E5Encoder _encoder;
    private readonly E5TextEmbedder _embedder;

    private E5EmbeddingModel(E5Encoder encoder, E5TextEmbedder embedder)
    {
        _encoder = encoder;
        _embedder = embedder;
    }

    /// <inheritdoc />
    public string ModelId => E5ModelAssets.Repository;

    /// <inheritdoc />
    public int Dimensions => VectorLength;

    /// <summary>
    /// Opens the model from a local snapshot directory, verified file by file against the package's asset lock (SHA-256). Never
    /// downloads.
    /// </summary>
    /// <param name="directory">The directory holding the locked files.</param>
    /// <returns>The model; the caller disposes it.</returns>
    public static E5EmbeddingModel Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        var files = E5ModelAssets.OpenDirectory(directory, CacheVerification.Sha256);
        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256);
        var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
        {
            Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cpu, ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
        });
        return new E5EmbeddingModel(encoder, new E5TextEmbedder(tokenizer, encoder));
    }

    /// <inheritdoc />
    public RoostEmbeddingResult Embed(IReadOnlyList<string> inputs, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        var counts = new int[inputs.Count];
        var vectors = _embedder.EmbedBatch(inputs, counts, cancellationToken);
        return new RoostEmbeddingResult(vectors, counts.Sum());
    }

    /// <inheritdoc />
    public void Dispose() => _encoder.Dispose();
}

/// <summary>
/// Starling's EmbeddingGemma 2 text embedder (<c>Starlings.Models.EmbeddingGemma2</c> 0.3.0) on ONNX Runtime's CPU execution
/// provider, in the server process (decision 30), at one Matryoshka dimension chosen when it is opened. Inputs are embedded with
/// no prompt (<see cref="EmbeddingGemma2Prompt.None"/>); the caller adds a task prompt if it wants one. An input over the model's
/// token limit is refused as <c>prompt_too_long</c>, never truncated (the package's behaviour).
/// </summary>
public sealed class EmbeddingGemma2EmbeddingModel : IRoostEmbeddingModel, IDisposable
{
    /// <summary>The dimensions the package supports.</summary>
    public static IReadOnlyList<int> SupportedDimensions { get; } = [768, 512, 256, 128];

    private readonly EmbeddingGemma2Encoder _encoder;
    private readonly EmbeddingGemma2TextEmbedder _embedder;

    private EmbeddingGemma2EmbeddingModel(EmbeddingGemma2Encoder encoder, EmbeddingGemma2TextEmbedder embedder, int dimensions)
    {
        _encoder = encoder;
        _embedder = embedder;
        Dimensions = dimensions;
    }

    /// <inheritdoc />
    public string ModelId => EmbeddingGemma2ModelAssets.Repository;

    /// <inheritdoc />
    public int Dimensions { get; }

    /// <summary>
    /// Opens the model from a local snapshot directory, verified file by file against the package's asset lock (SHA-256). Never
    /// downloads.
    /// </summary>
    /// <param name="directory">The directory holding the locked files.</param>
    /// <param name="dimensions">The vector length, one of <see cref="SupportedDimensions"/>.</param>
    /// <returns>The model; the caller disposes it.</returns>
    public static EmbeddingGemma2EmbeddingModel Open(string directory, int dimensions = 768)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!SupportedDimensions.Contains(dimensions))
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "EmbeddingGemma 2 supports 768, 512, 256 or 128 dimensions.");
        }

        var files = EmbeddingGemma2ModelAssets.OpenDirectory(directory, CacheVerification.Sha256);
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath, EmbeddingGemma2ModelAssets.TokenizerSha256);
        var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
        {
            Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cpu, ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 },
            Dimension = dimensions,
        });
        return new EmbeddingGemma2EmbeddingModel(encoder, new EmbeddingGemma2TextEmbedder(tokenizer, encoder), dimensions);
    }

    /// <inheritdoc />
    public RoostEmbeddingResult Embed(IReadOnlyList<string> inputs, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        var counts = new int[inputs.Count];
        try
        {
            var vectors = _embedder.EmbedBatch(inputs, EmbeddingGemma2Prompt.None, counts, cancellationToken);
            return new RoostEmbeddingResult(vectors, counts.Sum());
        }
        catch (EmbeddingGemma2InputTooLongException e)
        {
            throw PoolingErrors.TooLong(e.InputIndex, e.TokenCount, e.MaximumTokens);
        }
    }

    /// <inheritdoc />
    public void Dispose() => _encoder.Dispose();
}

/// <summary>
/// Starling's GLiNER2 entity extractor (<c>Starlings.Models.Gliner2</c> 0.3.0) on ONNX Runtime's CPU execution provider, in the
/// server process (decision 30): English text, one label set per call, at most 512 tokens per text with the schema prefix (a
/// longer text is refused as <c>prompt_too_long</c>, never truncated). Starling's deployment requirement for this profile is
/// <c>MALLOC_MMAP_THRESHOLD_=65536</c> in the process environment.
/// </summary>
public sealed class Gliner2ExtractionModel : IRoostExtractionModel, IDisposable
{
    private readonly Gliner2EntityExtractor _extractor;

    private Gliner2ExtractionModel(Gliner2EntityExtractor extractor) => _extractor = extractor;

    /// <inheritdoc />
    public string ModelId => Gliner2ModelAssets.Repository;

    /// <summary>
    /// Opens the model from a local snapshot directory, verified file by file against the package's asset lock (SHA-256). Never
    /// downloads.
    /// </summary>
    /// <param name="directory">The directory holding the locked files.</param>
    /// <returns>The model; the caller disposes it.</returns>
    public static Gliner2ExtractionModel Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        var files = Gliner2ModelAssets.OpenDirectory(directory, CacheVerification.Sha256);
        return new Gliner2ExtractionModel(Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions
        {
            Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cpu },
        }));
    }

    /// <inheritdoc />
    public IReadOnlyList<RoostExtraction> Extract(IReadOnlyList<string> inputs, IReadOnlyList<string> labels, float? threshold, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        IReadOnlyList<Gliner2ExtractionResult> results;
        try
        {
            results = _extractor.ExtractBatch(inputs, labels, threshold, cancellationToken);
        }
        catch (Gliner2InputTooLongException e)
        {
            throw PoolingErrors.TooLong(e.InputIndex, e.TokenCount, e.MaxTokens);
        }

        return [.. results.Select(r => new RoostExtraction(
            [.. r.Entities.Select(e => new RoostEntity(e.Label, e.Text, e.ScalarStart, e.ScalarEnd, e.Confidence))],
            r.TokenCount))];
    }

    /// <inheritdoc />
    public void Dispose() => _extractor.Dispose();
}

/// <summary>The C6 errors the pooling models raise.</summary>
internal static class PoolingErrors
{
    /// <summary>An input over the model's token limit: <c>prompt_too_long</c> naming <c>input[i]</c>.</summary>
    public static RoostChatException TooLong(int index, int tokens, int max) => new(new RoostError(
        RoostErrorCode.PromptTooLong,
        FormattableString.Invariant($"Input {index} has {tokens} tokens; the model reads at most {max} and nothing is truncated."),
        FormattableString.Invariant($"input[{index}]")));
}

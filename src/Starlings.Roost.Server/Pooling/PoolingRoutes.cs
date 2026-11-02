using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.OpenAi;
using Starlings.Roost.Server.Routing;
using static Starlings.Roost.Server.Routing.RequestFields;

namespace Starlings.Roost.Server.Pooling;

/// <summary>
/// The pooling routes (R-052), with the OpenAI error envelope (<see cref="ErrorBodies"/>):
/// <list type="bullet">
/// <item><c>POST /v1/embeddings</c>, OpenAI's request and response (<c>model</c>, <c>input</c> as a string or an array of strings,
/// <c>encoding_format</c> <c>float</c> or <c>base64</c>, <c>dimensions</c>, <c>user</c>) with vLLM's <c>id</c> and
/// <c>created</c>, over an <see cref="IRoostEmbeddingModel"/>.</item>
/// <item><c>POST /v1/extract</c>, Roost's own shape (neither OpenAI nor vLLM has an entity extraction route):
/// <c>{"model","input","labels","threshold"}</c> in, one list of labelled spans per input out, over an
/// <see cref="IRoostExtractionModel"/>.</item>
/// <item><c>POST /v1/rerank</c> and <c>POST /v1/classify</c>: always <c>profile_unsupported</c>, naming the capability, because no
/// Starling package provides a reranking or a classification model (nuget.org, checked 2026-11-02).</item>
/// </list>
/// A route whose model is not configured also answers <c>profile_unsupported</c>. Token-id input and vLLM's pooling extensions
/// (<c>truncate_prompt_tokens</c>, <c>normalize</c> and the like) are refused by name. Request text never reaches a log.
/// </summary>
internal sealed class PoolingRoutes
{
    /// <summary>The most inputs one request may carry, OpenAI's limit for <c>/v1/embeddings</c>.</summary>
    public const int MaxInputs = 2048;

    private readonly RoostPoolingModels _models;
    private readonly RouteRunner _runner;

    public PoolingRoutes(RoostPoolingModels models, ILogger logger)
    {
        _models = models;
        _runner = new RouteRunner(logger, ErrorBodies.Shape);
    }

    /// <summary>Checks that no two models of one kind share an id.</summary>
    /// <exception cref="ArgumentException">Two models of one kind share an id.</exception>
    public static void Validate(RoostPoolingModels models)
    {
        ArgumentNullException.ThrowIfNull(models);
        CheckDistinct(models.Embeddings.Select(m => m.ModelId), "embedding", nameof(models));
        CheckDistinct(models.Extraction.Select(m => m.ModelId), "extraction", nameof(models));
    }

    /// <summary><c>POST /v1/embeddings</c>.</summary>
    public Task EmbeddingsAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        if (_models.Embeddings.Count == 0)
        {
            throw NotConfigured("embeddings", "no embedding model is configured on this server (roost serve --pooling-e5 or --pooling-embeddinggemma2)");
        }

        string? model = null;
        List<string>? inputs = null;
        var base64 = false;
        int? dimensions = null;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            foreach (var field in Fields(body.RootElement))
            {
                var v = field.Value;
                switch (field.Name)
                {
                    case "model":
                        model = StringValue(v, "model");
                        break;
                    case "input":
                        inputs = Inputs(v);
                        break;
                    case "encoding_format":
                        base64 = StringValue(v, "encoding_format") switch
                        {
                            "float" => false,
                            "base64" => true,
                            var other => throw Refuse("encoding_format", $"encoding_format '{other}' is unsupported; use float or base64."),
                        };
                        break;
                    case "dimensions":
                        dimensions = PositiveInt(v, "dimensions");
                        break;
                    case "user":
                        _ = StringValue(v, "user");
                        break;
                    default:
                        throw Unknown(field.Name);
                }
            }
        }

        var embedder = Select(_models.Embeddings, model, m => m.ModelId, "embedding");
        if (inputs is null)
        {
            throw Refuse("input", "input is required.");
        }

        if (dimensions is { } d && d != embedder.Dimensions)
        {
            throw Refuse("dimensions", Inv($"Model '{embedder.ModelId}' is served at {embedder.Dimensions} dimensions; dimensions {d} is unsupported."));
        }

        var result = embedder.Embed(inputs, ct);
        if (result.Embeddings.Count != inputs.Count || result.Embeddings.Any(e => e.Length != embedder.Dimensions))
        {
            throw new InvalidOperationException($"The embedding model returned {result.Embeddings.Count} vectors for {inputs.Count} inputs, or a vector of the wrong length.");
        }

        var json = RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteString("id", "embd-" + Guid.NewGuid().ToString("N"));
            w.WriteString("object", "list");
            w.WriteNumber("created", DateTimeOffset.UtcNow.ToUnixTimeSeconds());
            w.WriteString("model", embedder.ModelId);
            w.WriteStartArray("data");
            for (var i = 0; i < result.Embeddings.Count; i++)
            {
                w.WriteStartObject();
                w.WriteNumber("index", i);
                w.WriteString("object", "embedding");
                if (base64)
                {
                    w.WriteString("embedding", Base64(result.Embeddings[i]));
                }
                else
                {
                    w.WriteStartArray("embedding");
                    foreach (var x in result.Embeddings[i])
                    {
                        w.WriteNumberValue(x);
                    }

                    w.WriteEndArray();
                }

                w.WriteEndObject();
            }

            w.WriteEndArray();
            WriteUsage(w, result.PromptTokens);
            w.WriteEndObject();
        });
        await RouteRunner.WriteJsonAsync(context, json).ConfigureAwait(false);
    });

    /// <summary><c>POST /v1/extract</c>.</summary>
    public Task ExtractAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        if (_models.Extraction.Count == 0)
        {
            throw NotConfigured("extract", "no extraction model is configured on this server (roost serve --pooling-gliner2)");
        }

        string? model = null;
        List<string>? inputs = null;
        List<string>? labels = null;
        float? threshold = null;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            foreach (var field in Fields(body.RootElement))
            {
                var v = field.Value;
                switch (field.Name)
                {
                    case "model":
                        model = StringValue(v, "model");
                        break;
                    case "input":
                        inputs = Inputs(v);
                        break;
                    case "labels":
                        labels = Labels(v);
                        break;
                    case "threshold":
                        var t = Number(v, "threshold");
                        threshold = t is >= 0f and <= 1f ? t : throw Refuse("threshold", Inv($"threshold must be in [0, 1]; got {t}."));
                        break;
                    case "user":
                        _ = StringValue(v, "user");
                        break;
                    default:
                        throw Unknown(field.Name);
                }
            }
        }

        var extractor = Select(_models.Extraction, model, m => m.ModelId, "extraction");
        if (inputs is null)
        {
            throw Refuse("input", "input is required.");
        }

        if (labels is null)
        {
            throw Refuse("labels", "labels is required.");
        }

        var results = extractor.Extract(inputs, labels, threshold, ct);
        if (results.Count != inputs.Count)
        {
            throw new InvalidOperationException($"The extraction model returned {results.Count} results for {inputs.Count} inputs.");
        }

        var json = RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteString("id", "extract-" + Guid.NewGuid().ToString("N"));
            w.WriteString("object", "list");
            w.WriteNumber("created", DateTimeOffset.UtcNow.ToUnixTimeSeconds());
            w.WriteString("model", extractor.ModelId);
            w.WriteStartArray("data");
            var tokens = 0;
            for (var i = 0; i < results.Count; i++)
            {
                tokens += results[i].TokenCount;
                w.WriteStartObject();
                w.WriteNumber("index", i);
                w.WriteString("object", "extraction");
                w.WriteStartArray("entities");
                foreach (var e in results[i].Entities)
                {
                    w.WriteStartObject();
                    w.WriteString("label", e.Label);
                    w.WriteString("text", e.Text);
                    w.WriteNumber("start", e.Start);
                    w.WriteNumber("end", e.End);
                    w.WriteNumber("confidence", e.Confidence);
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteEndArray();
            WriteUsage(w, tokens);
            w.WriteEndObject();
        });
        await RouteRunner.WriteJsonAsync(context, json).ConfigureAwait(false);
    });

    /// <summary><c>POST /v1/rerank</c>: <c>profile_unsupported</c>.</summary>
    public Task RerankAsync(HttpContext context) => _runner.RunAsync(context, _ =>
        throw new RoostChatException(new RoostError(
            RoostErrorCode.ProfileUnsupported,
            "Capability 'rerank' is unsupported: no Starling package provides a reranking model, so this server has none to serve (R-052).")));

    /// <summary><c>POST /v1/classify</c>: <c>profile_unsupported</c>.</summary>
    public Task ClassifyAsync(HttpContext context) => _runner.RunAsync(context, _ =>
        throw new RoostChatException(new RoostError(
            RoostErrorCode.ProfileUnsupported,
            "Capability 'classify' is unsupported: no Starling package provides a classification model, so this server has none to serve (R-052).")));

    private static RoostChatException NotConfigured(string capability, string reason) =>
        new(new RoostError(RoostErrorCode.ProfileUnsupported, $"Capability '{capability}' is unavailable: {reason}."));

    private static T Select<T>(IReadOnlyList<T> models, string? requested, Func<T, string> id, string kind)
    {
        if (requested is null)
        {
            return models.Count == 1
                ? models[0]
                : throw Refuse("model", $"model is required: this server serves {models.Count} {kind} models ({string.Join(", ", models.Select(id))}).");
        }

        foreach (var m in models)
        {
            if (string.Equals(id(m), requested, StringComparison.Ordinal))
            {
                return m;
            }
        }

        throw Refuse("model", $"Model '{requested}' is not served here; {kind} models: {string.Join(", ", models.Select(id))}.");
    }

    // OpenAI's input: a string or a non-empty array of strings. Token-id arrays are refused: the models take text only here.
    private static List<string> Inputs(JsonElement v)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return [v.GetString()!];
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("input", "input must be a string or an array of strings.");
        }

        int count = v.GetArrayLength();
        if (count == 0 || count > MaxInputs)
        {
            throw Refuse("input", Inv($"input must hold 1 to {MaxInputs} strings; got {count}."));
        }

        var inputs = new List<string>(count);
        var i = 0;
        foreach (var item in v.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String)
            {
                throw Refuse(Inv($"input[{i}]"), item.ValueKind is JsonValueKind.Number or JsonValueKind.Array
                    ? "Token-id input is unsupported: send text."
                    : "Each input must be a string.");
            }

            inputs.Add(item.GetString()!);
            i++;
        }

        return inputs;
    }

    private static List<string> Labels(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Array || v.GetArrayLength() == 0)
        {
            throw Refuse("labels", "labels must be a non-empty array of strings.");
        }

        var labels = new List<string>(v.GetArrayLength());
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var i = 0;
        foreach (var item in v.EnumerateArray())
        {
            string param = Inv($"labels[{i++}]");
            var label = item.ValueKind == JsonValueKind.String ? item.GetString()! : throw Refuse(param, "Each label must be a string.");
            if (string.IsNullOrWhiteSpace(label))
            {
                throw Refuse(param, "A label must not be empty.");
            }

            if (!seen.Add(label))
            {
                throw Refuse(param, $"Label '{label}' appears more than once.");
            }

            labels.Add(label);
        }

        return labels;
    }

    // OpenAI's base64 encoding: the vector's float32 values, little-endian, as one base64 string.
    private static string Base64(float[] vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        for (var i = 0; i < vector.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), vector[i]);
        }

        return Convert.ToBase64String(bytes);
    }

    private static void WriteUsage(Utf8JsonWriter w, int promptTokens)
    {
        w.WriteStartObject("usage");
        w.WriteNumber("prompt_tokens", promptTokens);
        w.WriteNumber("total_tokens", promptTokens);
        w.WriteEndObject();
    }

    private static void CheckDistinct(IEnumerable<string> ids, string kind, string paramName)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var id in ids)
        {
            if (!seen.Add(id))
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"Two {kind} models share the id '{id}'."), paramName);
            }
        }
    }
}

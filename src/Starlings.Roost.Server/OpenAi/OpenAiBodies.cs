using System.Text.Json.Serialization;

namespace Starlings.Roost.Server.OpenAi;

// Response bodies of the OpenAI routes, written through the source-generated OpenAiJsonContext (Native AOT, S1). Requests are
// read field by field from a JsonDocument (OpenAiRequestReader), so that every unknown or unsupported field can be named.

/// <summary>Token counts of a response.</summary>
internal sealed record UsageBody(int PromptTokens, int CompletionTokens, int TotalTokens);

/// <summary>
/// An assistant message in a chat completion. <c>content</c> is null when the message is only tool calls, as in OpenAI's API;
/// <c>reasoning_content</c> (vLLM's field) and <c>tool_calls</c> are omitted when there are none.
/// </summary>
internal sealed record ChatMessageBody(
    string Role,
    string? Content,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? ReasoningContent = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] ToolCallBody[]? ToolCalls = null);

/// <summary>The function of a tool call: its name and its arguments as a JSON object in a string.</summary>
internal sealed record FunctionCallBody(string Name, string Arguments);

/// <summary>A tool call of a chat completion message.</summary>
internal sealed record ToolCallBody(string Id, string Type, FunctionCallBody Function);

/// <summary>A tool call in a streamed delta: the whole call in one delta, at its index among the response's calls.</summary>
internal sealed record ToolCallDeltaBody(int Index, string Id, string Type, FunctionCallBody Function);

/// <summary>One choice of a chat completion.</summary>
internal sealed record ChatChoiceBody(int Index, ChatMessageBody Message, string? Logprobs, string? FinishReason);

/// <summary>A <c>chat.completion</c> object.</summary>
internal sealed record ChatCompletionBody(string Id, string Object, long Created, string Model, ChatChoiceBody[] Choices, UsageBody Usage);

/// <summary>The delta of a streamed chat chunk; absent fields are omitted.</summary>
internal sealed record ChatDeltaBody(
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Role,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Content,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? ReasoningContent = null,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] ToolCallDeltaBody[]? ToolCalls = null);

/// <summary>One choice of a streamed chat chunk.</summary>
internal sealed record ChatChunkChoiceBody(int Index, ChatDeltaBody Delta, string? Logprobs, string? FinishReason);

/// <summary>A <c>chat.completion.chunk</c> object; <c>usage</c> appears only on the usage chunk (<c>stream_options.include_usage</c>).</summary>
internal sealed record ChatCompletionChunkBody(
    string Id,
    string Object,
    long Created,
    string Model,
    ChatChunkChoiceBody[] Choices,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] UsageBody? Usage);

/// <summary>One choice of a text completion.</summary>
internal sealed record TextChoiceBody(int Index, string Text, string? Logprobs, string? FinishReason);

/// <summary>A <c>text_completion</c> object, whole or streamed; <c>usage</c> is omitted on stream chunks without it.</summary>
internal sealed record TextCompletionBody(
    string Id,
    string Object,
    long Created,
    string Model,
    TextChoiceBody[] Choices,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] UsageBody? Usage);

/// <summary>A <c>model</c> object.</summary>
internal sealed record ModelBody(string Id, string Object, long Created, string OwnedBy);

/// <summary>The <c>list</c> of <c>GET /v1/models</c>.</summary>
internal sealed record ModelListBody(string Object, ModelBody[] Data);

/// <summary>The source-generated serializer for the response bodies (no reflection; Native AOT).</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(ChatCompletionBody))]
[JsonSerializable(typeof(ChatCompletionChunkBody))]
[JsonSerializable(typeof(TextCompletionBody))]
[JsonSerializable(typeof(ModelListBody))]
internal sealed partial class OpenAiJsonContext : JsonSerializerContext;

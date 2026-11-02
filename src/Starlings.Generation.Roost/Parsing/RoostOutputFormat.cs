namespace Starlings.Generation.Roost;

/// <summary>
/// How a model writes tool calls in its completion, so that <see cref="RoostChatClient"/> can return them as
/// <c>FunctionCallContent</c> (R-053). Each format is the one the model's chat template instructs the model to use.
/// </summary>
public enum RoostToolCallFormat
{
    /// <summary>The model has no tool-call format: tool definitions are refused with <c>InvalidOption</c> naming <c>tools</c>.</summary>
    None = 0,

    /// <summary>
    /// Hermes style, as the Qwen2.5 and Qwen3 templates instruct: <c>&lt;tool_call&gt;{"name": ..., "arguments": {...}}&lt;/tool_call&gt;</c>,
    /// one JSON object per block.
    /// </summary>
    Hermes,

    /// <summary>
    /// The Qwen3.5 template's format: <c>&lt;tool_call&gt;&lt;function=NAME&gt;&lt;parameter=ARG&gt;VALUE&lt;/parameter&gt;&lt;/function&gt;&lt;/tool_call&gt;</c>,
    /// parsed by Starling's <c>Qwen35ToolCallParser</c>, which casts each value with the tool's JSON schema.
    /// </summary>
    Qwen35,
}

/// <summary>How a model writes its reasoning in its completion, so that <see cref="RoostChatClient"/> can return it as <c>TextReasoningContent</c> (R-053).</summary>
public enum RoostReasoningFormat
{
    /// <summary>The model has no reasoning format: <see cref="RoostChatClient.EnableThinkingProperty"/> is refused.</summary>
    None = 0,

    /// <summary>
    /// A think block, as the Qwen3 and Qwen3.5 templates use: with thinking enabled the completion is the reasoning, then
    /// <c>&lt;/think&gt;</c>, then the answer (the generation prompt opens the block; a repeated <c>&lt;think&gt;</c> at the start is dropped).
    /// </summary>
    ThinkTags,
}

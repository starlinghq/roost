using System.Collections.Immutable;

namespace Starlings.Roost.Generation;

/// <summary>Role of a <see cref="RoostChatMessage"/>.</summary>
public enum RoostChatRole
{
    /// <summary>System instruction.</summary>
    System,

    /// <summary>User turn.</summary>
    User,

    /// <summary>Assistant turn, optionally with reasoning and tool calls.</summary>
    Assistant,

    /// <summary>Tool result.</summary>
    Tool,
}

/// <summary>A tool call of an assistant turn. The engine-facing form of Starling's <c>Qwen35ToolCall</c>.</summary>
/// <param name="Name">Function name.</param>
/// <param name="ArgumentsJson">A JSON object with the arguments, in call order; <c>{}</c> for none.</param>
public sealed record RoostToolCall(string Name, string ArgumentsJson = "{}");

/// <summary>A tool (function) definition for the template's tools block.</summary>
/// <param name="Name">Function name.</param>
/// <param name="Description">Description shown to the model.</param>
/// <param name="ParametersJson">JSON Schema of the parameters as a JSON object, or null for a tool without parameters.</param>
public sealed record RoostToolDefinition(string Name, string Description, string? ParametersJson = null);

/// <summary>One message of a conversation handed to an <see cref="IRoostChatTemplate"/>.</summary>
/// <param name="Role">Message role.</param>
/// <param name="Content">Message text.</param>
/// <param name="ReasoningContent">Assistant reasoning (the text of a think block), or null.</param>
/// <param name="ToolCalls">Assistant tool calls, or null for none.</param>
public sealed record RoostChatMessage(
    RoostChatRole Role,
    string Content,
    string? ReasoningContent = null,
    ImmutableArray<RoostToolCall> ToolCalls = default)
{
    /// <summary>A system message.</summary>
    /// <param name="content">Text.</param>
    /// <returns>The message.</returns>
    public static RoostChatMessage System(string content) => new(RoostChatRole.System, content);

    /// <summary>A user message.</summary>
    /// <param name="content">Text.</param>
    /// <returns>The message.</returns>
    public static RoostChatMessage User(string content) => new(RoostChatRole.User, content);

    /// <summary>An assistant message.</summary>
    /// <param name="content">Text.</param>
    /// <param name="reasoningContent">Reasoning text, or null.</param>
    /// <param name="toolCalls">Tool calls, or default for none.</param>
    /// <returns>The message.</returns>
    public static RoostChatMessage Assistant(string content, string? reasoningContent = null, ImmutableArray<RoostToolCall> toolCalls = default)
        => new(RoostChatRole.Assistant, content, reasoningContent, toolCalls);

    /// <summary>A tool-result message.</summary>
    /// <param name="content">Result text.</param>
    /// <returns>The message.</returns>
    public static RoostChatMessage Tool(string content) => new(RoostChatRole.Tool, content);
}

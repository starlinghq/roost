using System.Collections.Immutable;
using Starlings.Models.QwenChat;

namespace Starlings.Roost.Generation;

/// <summary>
/// The Qwen3 template (first used by the Qwen3-1.7B profile, R-015) as the engine calls it: Roost's message types are mapped field
/// by field to Starling's and rendered by <see cref="QwenChatTemplate"/> with <see cref="QwenChatCheckpoint.Qwen3"/>. No template
/// logic lives here.
/// </summary>
/// <remarks>
/// Starling publishes this template for a text-only profile with <c>enable_thinking=false</c> (<c>Starlings.Models.QwenChat</c>
/// 0.5.0, Experimental). Tool definitions, tool calls, tool messages and thinking mode have no representation in it, so this class
/// throws <see cref="NotSupportedException"/> for them rather than rendering something the published template does not define.
/// Status: experimental (as Starling's profile); no Roost GPU evidence for a Qwen3 model yet.
/// </remarks>
public sealed class Qwen3RoostChatTemplate : IRoostChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Qwen3RoostChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Render(IReadOnlyList<RoostChatMessage> messages, ImmutableArray<RoostToolDefinition> tools = default, bool enableThinking = false)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (!tools.IsDefaultOrEmpty)
        {
            throw new NotSupportedException("The published Qwen3 template (Starlings.Models.QwenChat) does not support tools.");
        }

        if (enableThinking)
        {
            throw new NotSupportedException("The published Qwen3 template (Starlings.Models.QwenChat) renders with enable_thinking=false only.");
        }

        var mapped = new QwenChatMessage[messages.Count];
        for (var i = 0; i < mapped.Length; i++)
        {
            mapped[i] = Map(messages[i]);
        }

        return QwenChatTemplate.Render(QwenChatCheckpoint.Qwen3, mapped);
    }

    private static QwenChatMessage Map(RoostChatMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        var role = message.Role switch
        {
            RoostChatRole.System => QwenChatRole.System,
            RoostChatRole.User => QwenChatRole.User,
            RoostChatRole.Assistant => QwenChatRole.Assistant,
            RoostChatRole.Tool => throw new NotSupportedException("The published Qwen3 template (Starlings.Models.QwenChat) has no tool role."),
            _ => throw new ArgumentOutOfRangeException(nameof(message), message.Role, "Unknown chat role."),
        };
        if (!message.ToolCalls.IsDefaultOrEmpty)
        {
            throw new NotSupportedException("The published Qwen3 template (Starlings.Models.QwenChat) does not support tool calls.");
        }

        return new QwenChatMessage(role, message.Content, message.ReasoningContent);
    }
}

using System.Collections.Immutable;

namespace Starlings.Roost.Generation;

/// <summary>Renders a conversation to the prompt text of one model, ending with the assistant generation prompt.</summary>
public interface IRoostChatTemplate
{
    /// <summary>Renders a conversation.</summary>
    /// <param name="messages">The conversation, oldest first.</param>
    /// <param name="tools">Tool definitions, or default for none.</param>
    /// <param name="enableThinking">Whether the generation prompt leaves the think block open.</param>
    /// <returns>The prompt text.</returns>
    /// <remarks>An implementation throws its model's template exception when the template rejects the conversation; the Qwen3.5
    /// implementation throws <c>Qwen35TemplateException</c>.</remarks>
    string Render(IReadOnlyList<RoostChatMessage> messages, ImmutableArray<RoostToolDefinition> tools = default, bool enableThinking = false);
}

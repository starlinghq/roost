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
    /// <exception cref="Starlings.Models.Qwen35.Qwen35TemplateException">The template rejects the conversation.</exception>
    string Render(IReadOnlyList<RoostChatMessage> messages, ImmutableArray<RoostToolDefinition> tools = default, bool enableThinking = false);
}

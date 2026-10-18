using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using Starlings.Models.Qwen35;

namespace Starlings.Roost.Generation;

/// <summary>
/// The Qwen3.5-0.8B template (M0 model, decision 15) as the engine calls it: Roost's message types are mapped field by field to
/// Starling's and rendered by <see cref="Qwen35ChatTemplate"/>. No template logic lives here.
/// </summary>
public sealed class Qwen35RoostChatTemplate : IRoostChatTemplate
{
    /// <summary>The shared instance.</summary>
    public static Qwen35RoostChatTemplate Instance { get; } = new();

    /// <inheritdoc />
    public string Render(IReadOnlyList<RoostChatMessage> messages, ImmutableArray<RoostToolDefinition> tools = default, bool enableThinking = false)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var mapped = new Qwen35Message[messages.Count];
        for (var i = 0; i < mapped.Length; i++)
        {
            mapped[i] = Map(messages[i]);
        }

        if (tools.IsDefaultOrEmpty)
        {
            return Qwen35ChatTemplate.Render(mapped, enableThinking);
        }

        var mappedTools = new Qwen35Tool[tools.Length];
        for (var i = 0; i < mappedTools.Length; i++)
        {
            mappedTools[i] = Map(tools[i]);
        }

        return Qwen35ChatTemplate.Render(mapped, mappedTools, enableThinking);
    }

    private static Qwen35Message Map(RoostChatMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        var role = message.Role switch
        {
            RoostChatRole.System => Qwen35Role.System,
            RoostChatRole.User => Qwen35Role.User,
            RoostChatRole.Assistant => Qwen35Role.Assistant,
            RoostChatRole.Tool => Qwen35Role.Tool,
            _ => throw new ArgumentOutOfRangeException(nameof(message), message.Role, "Unknown chat role."),
        };
        IReadOnlyList<Qwen35ToolCall>? calls = null;
        if (!message.ToolCalls.IsDefaultOrEmpty)
        {
            calls = [.. message.ToolCalls.Select(Map)];
        }

        return new Qwen35Message(role, message.Content, message.ReasoningContent) { ToolCalls = calls };
    }

    private static Qwen35ToolCall Map(RoostToolCall call)
    {
        using var document = JsonDocument.Parse(call.ArgumentsJson);
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"Arguments of tool call '{call.Name}' must be a JSON object."), nameof(call));
        }

        // Clone: the elements must outlive the parsed document.
        var arguments = document.RootElement.EnumerateObject()
            .Select(p => new KeyValuePair<string, JsonElement>(p.Name, p.Value.Clone()))
            .ToList();
        return Qwen35ToolCall.Create(call.Name, arguments);
    }

    private static Qwen35Tool Map(RoostToolDefinition tool)
    {
        if (tool.ParametersJson is null)
        {
            return Qwen35Tool.Function(tool.Name, tool.Description);
        }

        using var document = JsonDocument.Parse(tool.ParametersJson);
        return Qwen35Tool.Function(tool.Name, tool.Description, document.RootElement.Clone());
    }
}

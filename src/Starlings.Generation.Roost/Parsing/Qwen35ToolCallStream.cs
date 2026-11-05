using System.Text.Json;
using Starlings.Models.Qwen35;
using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Parsing;

/// <summary>
/// The Qwen3.5 tool-call format through Starling's published <see cref="Qwen35ToolCallParser"/> (<c>Starlings.Models.Qwen35</c>
/// 0.5.0): Roost maps the request's tools to Starling's and the parsed calls back. No parsing logic lives here. Starling's parser
/// runs with thinking off because <see cref="ThinkBlockParser"/> has already taken the reasoning out.
/// </summary>
internal sealed class Qwen35ToolCallStream : IToolCallStream
{
    private readonly Qwen35ToolCallParser _parser;

    public Qwen35ToolCallStream(IReadOnlyList<RoostToolDefinition> tools)
    {
        var mapped = new Qwen35Tool[tools.Count];
        for (var i = 0; i < mapped.Length; i++)
        {
            mapped[i] = Map(tools[i]);
        }

        _parser = new Qwen35ToolCallParser(mapped, enableThinking: false);
    }

    /// <inheritdoc />
    public void Append(string delta, List<OutputPart> parts) => Add(_parser.Append(delta), parts);

    /// <inheritdoc />
    public void Complete(List<OutputPart> parts) => Add(_parser.Complete(), parts);

    private static void Add(IEnumerable<Qwen35OutputPart> parsed, List<OutputPart> parts)
    {
        foreach (var part in parsed)
        {
            if (part.ToolCall is { } call)
            {
                var arguments = call.Arguments.ValueKind == JsonValueKind.Undefined ? "{}" : call.Arguments.GetRawText();
                parts.Add(OutputPart.Call(new RoostToolCall(call.Name, arguments)));
            }
            else if (part.Text is { Length: > 0 } text)
            {
                parts.Add(OutputPart.Answer(text));
            }
        }
    }

    private static Qwen35Tool Map(RoostToolDefinition tool)
    {
        // As Qwen35RoostChatTemplate maps it, so the schemas that cast values are the ones the prompt showed.
        if (tool.ParametersJson is null)
        {
            return Qwen35Tool.Function(tool.Name, tool.Description);
        }

        using var document = JsonDocument.Parse(tool.ParametersJson);
        return Qwen35Tool.Function(tool.Name, tool.Description, document.RootElement.Clone());
    }
}

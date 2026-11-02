using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Parsing;

/// <summary>
/// The parsers of one chat completion, chained (R-053): the think block first (when thinking is enabled), then the tool-call
/// format on the answer text (when the request rendered tools). Text deltas in, text, reasoning and tool-call parts out.
/// </summary>
internal sealed class RoostOutputParser
{
    private readonly ThinkBlockParser? _think;
    private readonly IToolCallStream? _tools;
    private readonly List<OutputPart> _scratch = [];

    private RoostOutputParser(ThinkBlockParser? think, IToolCallStream? tools)
    {
        _think = think;
        _tools = tools;
    }

    /// <summary>The parser for a request, or null when its output is plain text (no thinking, no tools).</summary>
    public static RoostOutputParser? Create(bool enableThinking, RoostReasoningFormat reasoning, IReadOnlyList<RoostToolDefinition> tools, RoostToolCallFormat toolFormat)
    {
        var think = enableThinking && reasoning == RoostReasoningFormat.ThinkTags ? new ThinkBlockParser() : null;
        IToolCallStream? stream = tools.Count == 0 ? null : toolFormat switch
        {
            RoostToolCallFormat.Hermes => new HermesToolCallParser(),
            RoostToolCallFormat.Qwen35 => new Qwen35ToolCallStream(tools),
            _ => null,
        };
        return think is null && stream is null ? null : new RoostOutputParser(think, stream);
    }

    /// <summary>Feeds the next text delta; returns the parts that are now final.</summary>
    public List<OutputPart> Append(string delta)
    {
        var parts = new List<OutputPart>();
        if (_think is null)
        {
            _tools!.Append(delta, parts);
            return parts;
        }

        _scratch.Clear();
        _think.Append(delta, _scratch);
        Route(_scratch, parts);
        return parts;
    }

    /// <summary>Ends the completion; returns the remaining parts.</summary>
    public List<OutputPart> Complete()
    {
        var parts = new List<OutputPart>();
        if (_think is not null)
        {
            _scratch.Clear();
            _think.Complete(_scratch);
            Route(_scratch, parts);
        }

        _tools?.Complete(parts);
        return parts;
    }

    private void Route(List<OutputPart> parsed, List<OutputPart> parts)
    {
        foreach (var part in parsed)
        {
            if (part.Kind == OutputPartKind.Text && _tools is not null)
            {
                _tools.Append(part.Text, parts);
            }
            else
            {
                parts.Add(part);
            }
        }
    }
}

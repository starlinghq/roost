using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Parsing;

/// <summary>What an <see cref="OutputPart"/> holds.</summary>
internal enum OutputPartKind
{
    /// <summary>Answer text.</summary>
    Text,

    /// <summary>Reasoning text (the inside of a think block).</summary>
    Reasoning,

    /// <summary>A parsed tool call.</summary>
    ToolCall,
}

/// <summary>One piece of a parsed completion: answer text, reasoning text or a tool call.</summary>
/// <param name="Kind">What the part holds.</param>
/// <param name="Text">The text of a text or reasoning part; empty for a tool call.</param>
/// <param name="ToolCall">The call of a tool-call part.</param>
internal readonly record struct OutputPart(OutputPartKind Kind, string Text, RoostToolCall? ToolCall)
{
    public static OutputPart Answer(string text) => new(OutputPartKind.Text, text, null);

    public static OutputPart Reasoning(string text) => new(OutputPartKind.Reasoning, text, null);

    public static OutputPart Call(RoostToolCall call) => new(OutputPartKind.ToolCall, string.Empty, call);
}

/// <summary>A streaming tool-call parser: text deltas in, text and tool-call parts out, in output order.</summary>
internal interface IToolCallStream
{
    /// <summary>Feeds the next text delta and adds the parts that are now final.</summary>
    void Append(string delta, List<OutputPart> parts);

    /// <summary>Ends the completion and adds the remaining parts; an unfinished block is text.</summary>
    void Complete(List<OutputPart> parts);
}

/// <summary>Helpers for holding back text that may be the start of a tag.</summary>
internal static class Tags
{
    /// <summary>The length of the longest suffix of <paramref name="text"/> that is a proper prefix of <paramref name="tag"/>.</summary>
    public static int PartialSuffix(string text, string tag)
    {
        for (var n = Math.Min(text.Length, tag.Length - 1); n > 0; n--)
        {
            if (string.CompareOrdinal(text, text.Length - n, tag, 0, n) == 0)
            {
                return n;
            }
        }

        return 0;
    }
}

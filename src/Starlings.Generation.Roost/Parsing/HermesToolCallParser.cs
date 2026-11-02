using System.Text;
using System.Text.Json;
using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Parsing;

/// <summary>
/// Parses Hermes-style tool calls, as the Qwen2.5 and Qwen3 templates instruct the model to write them, from streamed text
/// deltas (R-053):
/// <code>
/// &lt;tool_call&gt;
/// {"name": "get_weather", "arguments": {"city": "Paris"}}
/// &lt;/tool_call&gt;
/// </code>
/// </summary>
/// <remarks>
/// <para>A block runs from <c>&lt;tool_call&gt;</c> to the next <c>&lt;/tool_call&gt;</c> and holds one JSON object with a non-empty
/// string <c>name</c> and an optional object <c>arguments</c> (absent means <c>{}</c>); other members are ignored. A block that
/// does not parse, or one still open when the completion ends, is returned as text, verbatim: model output never throws.</para>
/// <para>Whitespace-only text before the first call, between calls and after the last call is dropped (the templates write
/// their own separators), as Starling's <c>Qwen35ToolCallParser</c> does; other text is kept verbatim.</para>
/// <para>Text is returned as soon as it cannot be the start of <c>&lt;tool_call&gt;</c>; a call when its block closes. The parts of
/// any split of a completion into deltas, with adjacent text merged, equal those of the whole completion.</para>
/// </remarks>
internal sealed class HermesToolCallParser : IToolCallStream
{
    private const string Open = "<tool_call>";
    private const string Close = "</tool_call>";

    private readonly StringBuilder _buffer = new();
    private readonly StringBuilder _whitespace = new();
    private bool _inside;
    private bool _segmentStarted; // the current text segment (since the start or the last call) has non-whitespace text
    private bool _sawCall;

    /// <inheritdoc />
    public void Append(string delta, List<OutputPart> parts)
    {
        _buffer.Append(delta);
        Drain(parts, final: false);
    }

    /// <inheritdoc />
    public void Complete(List<OutputPart> parts) => Drain(parts, final: true);

    /// <summary>Parses a whole completion.</summary>
    public static List<OutputPart> Parse(string completion)
    {
        var parser = new HermesToolCallParser();
        var parts = new List<OutputPart>();
        parser.Append(completion, parts);
        parser.Complete(parts);
        return parts;
    }

    /// <summary>Reads one block's JSON; false when it is not a call.</summary>
    internal static bool TryReadCall(string block, out RoostToolCall? call)
    {
        call = null;
        try
        {
            using var document = JsonDocument.Parse(block);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object
                || !root.TryGetProperty("name", out var name)
                || name.ValueKind != JsonValueKind.String
                || name.GetString() is not { Length: > 0 } functionName)
            {
                return false;
            }

            var arguments = "{}";
            if (root.TryGetProperty("arguments", out var args))
            {
                if (args.ValueKind != JsonValueKind.Object)
                {
                    return false;
                }

                arguments = args.GetRawText();
            }

            call = new RoostToolCall(functionName, arguments);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private void Drain(List<OutputPart> parts, bool final)
    {
        while (true)
        {
            var s = _buffer.ToString();
            if (!_inside)
            {
                var open = s.IndexOf(Open, StringComparison.Ordinal);
                if (open >= 0)
                {
                    Text(s[..open], parts);
                    Hold(s[(open + Open.Length)..]);
                    _inside = true;
                    continue;
                }

                var keep = final ? 0 : Tags.PartialSuffix(s, Open);
                Text(s[..^keep], parts);
                Hold(s[^keep..]);
                if (final)
                {
                    EndSegment(parts);
                }

                return;
            }

            var close = s.IndexOf(Close, StringComparison.Ordinal);
            if (close >= 0)
            {
                var block = s[..close];
                Hold(s[(close + Close.Length)..]);
                _inside = false;
                if (TryReadCall(block, out var call))
                {
                    _whitespace.Clear(); // whitespace-only text before a call
                    _segmentStarted = false;
                    _sawCall = true;
                    parts.Add(OutputPart.Call(call!));
                }
                else
                {
                    Text(Open + block + Close, parts);
                }

                continue;
            }

            if (final)
            {
                _buffer.Clear();
                _inside = false;
                Text(Open + s, parts);
                EndSegment(parts);
            }

            return;
        }
    }

    private void Text(string text, List<OutputPart> parts)
    {
        if (text.Length == 0)
        {
            return;
        }

        if (_segmentStarted)
        {
            AddText(parts, text);
            return;
        }

        _whitespace.Append(text);
        if (!string.IsNullOrWhiteSpace(text))
        {
            AddText(parts, _whitespace.ToString());
            _whitespace.Clear();
            _segmentStarted = true;
        }
    }

    private void EndSegment(List<OutputPart> parts)
    {
        if (!_segmentStarted && _whitespace.Length > 0 && !_sawCall)
        {
            AddText(parts, _whitespace.ToString()); // a whitespace-only completion with no call is kept
        }

        _whitespace.Clear();
    }

    private static void AddText(List<OutputPart> parts, string text)
    {
        if (parts.Count > 0 && parts[^1].Kind == OutputPartKind.Text)
        {
            parts[^1] = OutputPart.Answer(parts[^1].Text + text);
        }
        else
        {
            parts.Add(OutputPart.Answer(text));
        }
    }

    private void Hold(string text) => _buffer.Clear().Append(text);
}

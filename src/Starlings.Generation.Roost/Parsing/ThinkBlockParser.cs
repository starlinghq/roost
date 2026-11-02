using System.Text;

namespace Starlings.Generation.Roost.Parsing;

/// <summary>
/// Splits a completion made with thinking enabled into reasoning and answer text, incrementally (R-053). The Qwen3 and Qwen3.5
/// templates open the think block in the generation prompt, so the completion starts inside it: everything up to the first
/// <c>&lt;/think&gt;</c> is reasoning and the rest is the answer. A <c>&lt;think&gt;</c> the model repeats at the start is dropped.
/// </summary>
/// <remarks>
/// Line feeds are trimmed as the Qwen3.5 template trims them when it reads a think block back from an assistant turn: leading
/// and trailing ones around the reasoning, leading ones of the answer. A completion that ends inside the block (the output limit,
/// say) is all reasoning. Streaming holds back only what may still be the start of <c>&lt;/think&gt;</c> or a line feed that
/// may be trimmed, so the parts of any split into deltas, with adjacent parts of a kind merged, equal those of the whole.
/// </remarks>
internal sealed class ThinkBlockParser
{
    private const string Open = "<think>";
    private const string Close = "</think>";

    private readonly StringBuilder _held = new();
    private State _state;

    private enum State
    {
        Start,
        ReasoningStart,
        Reasoning,
        AnswerStart,
        Answer,
    }

    /// <summary>Feeds the next delta; adds reasoning and answer parts that are final.</summary>
    public void Append(string delta, List<OutputPart> parts)
    {
        if (_state == State.Answer)
        {
            if (delta.Length > 0)
            {
                parts.Add(OutputPart.Answer(delta));
            }

            return;
        }

        _held.Append(delta);
        Drain(parts, final: false);
    }

    /// <summary>Ends the completion; adds what is held.</summary>
    public void Complete(List<OutputPart> parts) => Drain(parts, final: true);

    private void Drain(List<OutputPart> parts, bool final)
    {
        while (true)
        {
            var s = _held.ToString();
            switch (_state)
            {
                case State.Start:
                    var t = s.TrimStart('\n');
                    if (t.StartsWith(Open, StringComparison.Ordinal))
                    {
                        Hold(t[Open.Length..]);
                        _state = State.ReasoningStart;
                        continue;
                    }

                    if (!final && Open.StartsWith(t, StringComparison.Ordinal))
                    {
                        return; // empty, or still possibly a repeated <think>
                    }

                    Hold(t);
                    _state = State.Reasoning;
                    continue;
                case State.ReasoningStart:
                    var r = s.TrimStart('\n');
                    Hold(r);
                    if (r.Length == 0)
                    {
                        return; // line feeds after a repeated <think>: dropped
                    }

                    _state = State.Reasoning;
                    continue;
                case State.Reasoning:
                    var close = s.IndexOf(Close, StringComparison.Ordinal);
                    if (close >= 0)
                    {
                        Add(parts, OutputPart.Reasoning(s[..close].TrimEnd('\n')));
                        Hold(s[(close + Close.Length)..]);
                        _state = State.AnswerStart;
                        continue;
                    }

                    if (final)
                    {
                        Add(parts, OutputPart.Reasoning(s.TrimEnd('\n')));
                        _held.Clear();
                        return;
                    }

                    var end = s.Length - Tags.PartialSuffix(s, Close);
                    while (end > 0 && s[end - 1] == '\n')
                    {
                        end--; // dropped if </think> follows
                    }

                    Add(parts, OutputPart.Reasoning(s[..end]));
                    Hold(s[end..]);
                    return;
                case State.AnswerStart:
                    var answer = s.TrimStart('\n');
                    _held.Clear();
                    if (answer.Length > 0)
                    {
                        _state = State.Answer;
                        parts.Add(OutputPart.Answer(answer));
                    }

                    return;
                default:
                    if (s.Length > 0)
                    {
                        parts.Add(OutputPart.Answer(s));
                    }

                    _held.Clear();
                    return;
            }
        }
    }

    private static void Add(List<OutputPart> parts, OutputPart part)
    {
        if (part.Text.Length > 0)
        {
            parts.Add(part);
        }
    }

    private void Hold(string text) => _held.Clear().Append(text);
}

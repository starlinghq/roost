using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.Responses;

/// <summary>
/// One Responses API response: its request echo, its status (<c>queued</c>, <c>in_progress</c>, then <c>completed</c>,
/// <c>incomplete</c>, <c>failed</c> or <c>cancelled</c>), the text generated so far, usage and error. Thread-safe: a background
/// generation writes it while <c>GET /v1/responses/{id}</c> reads it.
/// </summary>
internal sealed class StoredResponse
{
    public const string Queued = "queued";
    public const string InProgress = "in_progress";
    public const string Completed = "completed";
    public const string Incomplete = "incomplete";
    public const string Failed = "failed";
    public const string Cancelled = "cancelled";

    private readonly Lock _gate = new();
    private readonly StringBuilder _text = new();
    private string _status;
    private (long Input, long Output)? _usage;
    private RoostError? _error;
    private bool _truncated;
    private ResponsesRequest _request;

    public StoredResponse(ResponsesRequest request, string model, DateTimeOffset created)
    {
        _request = request;
        Model = model;
        CreatedAt = created;
        Id = "resp_" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(16));
        MessageId = "msg_" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(16));
        _status = request.Background ? Queued : InProgress;
    }

    public string Id { get; }

    /// <summary>The id of the one output message item.</summary>
    public string MessageId { get; }

    /// <summary>The request; its messages are dropped when the response ends, because no route returns them.</summary>
    public ResponsesRequest Request
    {
        get
        {
            lock (_gate)
            {
                return _request;
            }
        }
    }

    public string Model { get; }

    public DateTimeOffset CreatedAt { get; }

    /// <summary>Cancels a background generation; null for a response generated in its request.</summary>
    public CancellationTokenSource? Cancellation { get; set; }

    /// <summary>The background generation; null for a response generated in its request.</summary>
    public Task? Run { get; set; }

    public string Status
    {
        get
        {
            lock (_gate)
            {
                return _status;
            }
        }
    }

    public bool IsTerminal
    {
        get
        {
            lock (_gate)
            {
                return IsTerminalStatus(_status);
            }
        }
    }

    public string Text
    {
        get
        {
            lock (_gate)
            {
                return _text.ToString();
            }
        }
    }

    public void Start()
    {
        lock (_gate)
        {
            if (_status == Queued)
            {
                _status = InProgress;
            }
        }
    }

    public void Append(string text)
    {
        lock (_gate)
        {
            _text.Append(text);
        }
    }

    /// <summary>Records the end of the generation: <c>incomplete</c> when the output limit or context ended it, else <c>completed</c>.</summary>
    public void Finish(ChatFinishReason? finish, UsageDetails? usage)
    {
        lock (_gate)
        {
            if (IsTerminalStatus(_status))
            {
                return;
            }

            _truncated = finish == ChatFinishReason.Length;
            _status = _truncated ? Incomplete : Completed;
            _usage = (usage?.InputTokenCount ?? 0, usage?.OutputTokenCount ?? 0);
            DropMessages();
        }
    }

    public void Fail(RoostError error)
    {
        lock (_gate)
        {
            if (!IsTerminalStatus(_status))
            {
                _status = Failed;
                _error = error;
                DropMessages();
            }
        }
    }

    public void MarkCancelled()
    {
        lock (_gate)
        {
            if (!IsTerminalStatus(_status))
            {
                _status = Cancelled;
                DropMessages();
            }
        }
    }

    private void DropMessages() => _request = _request with { Messages = [] };

    /// <summary>The response object, as it stands.</summary>
    public void Write(Utf8JsonWriter w)
    {
        lock (_gate)
        {
            var request = Request;
            var options = request.Options;
            w.WriteStartObject();
            w.WriteString("id", Id);
            w.WriteString("object", "response");
            w.WriteNumber("created_at", CreatedAt.ToUnixTimeSeconds());
            w.WriteString("status", _status);
            w.WriteBoolean("background", request.Background);
            if (_error is { } error)
            {
                w.WriteStartObject("error");
                w.WriteString("code", error.CodeName);
                w.WriteString("message", error.Message);
                w.WriteEndObject();
            }
            else
            {
                w.WriteNull("error");
            }

            if (_truncated)
            {
                w.WriteStartObject("incomplete_details");
                w.WriteString("reason", "max_output_tokens");
                w.WriteEndObject();
            }
            else
            {
                w.WriteNull("incomplete_details");
            }

            WriteStringOrNull(w, "instructions", options.Instructions);
            if (options.MaxOutputTokens is { } max)
            {
                w.WriteNumber("max_output_tokens", max);
            }
            else
            {
                w.WriteNull("max_output_tokens");
            }

            w.WriteStartObject("metadata");
            foreach (var (key, value) in request.Metadata)
            {
                w.WriteString(key, value);
            }

            w.WriteEndObject();
            w.WriteString("model", Model);
            w.WriteStartArray("output");
            if (_text.Length > 0 || _status is Completed or Incomplete)
            {
                WriteItemLocked(w);
            }

            w.WriteEndArray();
            w.WriteBoolean("parallel_tool_calls", false);
            w.WriteNull("previous_response_id");
            w.WriteBoolean("store", request.Store);
            WriteNumberOrNull(w, "temperature", options.Temperature);
            w.WriteStartObject("text");
            w.WriteStartObject("format");
            w.WriteString("type", "text");
            w.WriteEndObject();
            w.WriteEndObject();
            w.WriteString("tool_choice", "none");
            w.WriteStartArray("tools");
            w.WriteEndArray();
            WriteNumberOrNull(w, "top_p", options.TopP);
            w.WriteString("truncation", "disabled");
            if (_usage is { } usage)
            {
                w.WriteStartObject("usage");
                w.WriteNumber("input_tokens", usage.Input);
                w.WriteStartObject("input_tokens_details");
                w.WriteNumber("cached_tokens", 0);
                w.WriteEndObject();
                w.WriteNumber("output_tokens", usage.Output);
                w.WriteStartObject("output_tokens_details");
                w.WriteNumber("reasoning_tokens", 0);
                w.WriteEndObject();
                w.WriteNumber("total_tokens", usage.Input + usage.Output);
                w.WriteEndObject();
            }
            else
            {
                w.WriteNull("usage");
            }

            WriteStringOrNull(w, "user", request.User);
            w.WriteEndObject();
        }
    }

    /// <summary>The output message item, as it stands.</summary>
    public void WriteItem(Utf8JsonWriter w)
    {
        lock (_gate)
        {
            WriteItemLocked(w);
        }
    }

    /// <summary>The output text part, as it stands.</summary>
    public void WritePart(Utf8JsonWriter w)
    {
        lock (_gate)
        {
            WritePart(w, _text.ToString());
        }
    }

    /// <summary>An output text part with the given text.</summary>
    public static void WritePart(Utf8JsonWriter w, string text)
    {
        w.WriteStartObject();
        w.WriteString("type", "output_text");
        w.WriteString("text", text);
        w.WriteStartArray("annotations");
        w.WriteEndArray();
        w.WriteStartArray("logprobs");
        w.WriteEndArray();
        w.WriteEndObject();
    }

    private void WriteItemLocked(Utf8JsonWriter w)
    {
        w.WriteStartObject();
        w.WriteString("id", MessageId);
        w.WriteString("type", "message");
        w.WriteString("status", _status switch
        {
            Completed => Completed,
            Queued or InProgress => InProgress,
            _ => Incomplete,
        });
        w.WriteString("role", "assistant");
        w.WriteStartArray("content");
        if (_status is not (Queued or InProgress) || _text.Length > 0)
        {
            WritePart(w, _text.ToString());
        }

        w.WriteEndArray();
        w.WriteEndObject();
    }

    private static bool IsTerminalStatus(string status) => status is not (Queued or InProgress);

    private static void WriteStringOrNull(Utf8JsonWriter w, string name, string? value)
    {
        if (value is null)
        {
            w.WriteNull(name);
        }
        else
        {
            w.WriteString(name, value);
        }
    }

    private static void WriteNumberOrNull(Utf8JsonWriter w, string name, float? value)
    {
        if (value is { } v)
        {
            w.WriteNumber(name, v);
        }
        else
        {
            w.WriteNull(name);
        }
    }
}

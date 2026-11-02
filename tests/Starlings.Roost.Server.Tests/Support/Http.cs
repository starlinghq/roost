using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace Starlings.Roost.Server.Tests.Support;

/// <summary>Request and response helpers for the route tests.</summary>
internal static class Http
{
    public static CancellationToken Ct => TestContext.Current.CancellationToken;

    public static StringContent Json(string json) => new(json, Encoding.UTF8, new MediaTypeHeaderValue("application/json"));

    public static async Task<(HttpResponseMessage Response, JsonElement Body)> PostAsync(HttpClient http, string path, string json)
    {
        var response = await http.PostAsync(path, Json(json), Ct);
        var text = await response.Content.ReadAsStringAsync(Ct);
        using var doc = JsonDocument.Parse(text);
        return (response, doc.RootElement.Clone());
    }

    /// <summary>The <c>data:</c> payloads of a server-sent event stream, in order.</summary>
    public static List<string> Events(string stream)
    {
        var events = new List<string>();
        foreach (var block in stream.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            Assert.StartsWith("data: ", block, StringComparison.Ordinal);
            events.Add(block["data: ".Length..]);
        }

        return events;
    }

    /// <summary>The named events of a server-sent event stream (<c>event:</c> then <c>data:</c>), in order, with parsed data.</summary>
    public static List<(string Name, JsonElement Data)> NamedEvents(string stream)
    {
        var events = new List<(string, JsonElement)>();
        foreach (var block in stream.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            var lines = block.Split('\n');
            Assert.Equal(2, lines.Length);
            Assert.StartsWith("event: ", lines[0], StringComparison.Ordinal);
            Assert.StartsWith("data: ", lines[1], StringComparison.Ordinal);
            using var doc = JsonDocument.Parse(lines[1]["data: ".Length..]);
            events.Add((lines[0]["event: ".Length..], doc.RootElement.Clone()));
        }

        return events;
    }

    /// <summary>A GET whose body is JSON.</summary>
    public static async Task<(HttpResponseMessage Response, JsonElement Body)> GetAsync(HttpClient http, string path)
    {
        var response = await http.GetAsync(path, Ct);
        var text = await response.Content.ReadAsStringAsync(Ct);
        using var doc = JsonDocument.Parse(text);
        return (response, doc.RootElement.Clone());
    }

    /// <summary>Checks Anthropic's error body: <c>{"type":"error","error":{"type","message"}}</c>, the message starting with the refused field.</summary>
    public static JsonElement AssertAnthropicError(JsonElement body, string type, string? param)
    {
        Assert.Equal(["type", "error"], body.EnumerateObject().Select(p => p.Name));
        Assert.Equal("error", body.GetProperty("type").GetString());
        var error = body.GetProperty("error");
        Assert.Equal(["type", "message"], error.EnumerateObject().Select(p => p.Name));
        Assert.Equal(type, error.GetProperty("type").GetString());
        var message = error.GetProperty("message").GetString();
        Assert.False(string.IsNullOrWhiteSpace(message));
        if (param is not null)
        {
            Assert.StartsWith(param + ": ", message, StringComparison.Ordinal);
        }

        return error;
    }

    /// <summary>Checks the one error body: exactly <c>message</c>, <c>type</c>, <c>param</c> and <c>code</c> under <c>error</c>.</summary>
    public static JsonElement AssertErrorBody(JsonElement body, string code, string? param)
    {
        var error = Assert.Single(body.EnumerateObject(), p => p.Name == "error").Value;
        Assert.Single(body.EnumerateObject());
        Assert.Equal(["message", "type", "param", "code"], error.EnumerateObject().Select(p => p.Name));
        Assert.Equal("roost_error", error.GetProperty("type").GetString());
        Assert.Equal(code, error.GetProperty("code").GetString());
        Assert.False(string.IsNullOrWhiteSpace(error.GetProperty("message").GetString()));
        if (param is null)
        {
            Assert.Equal(JsonValueKind.Null, error.GetProperty("param").ValueKind);
        }
        else
        {
            Assert.Equal(param, error.GetProperty("param").GetString());
        }

        return error;
    }
}

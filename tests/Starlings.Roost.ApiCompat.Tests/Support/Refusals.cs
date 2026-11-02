using System.ClientModel;
using System.Text.Json;

namespace Starlings.Roost.ApiCompat.Tests.Support;

/// <summary>Checks a refusal as the official client surfaces it: a <see cref="ClientResultException"/> over Roost's one error body.</summary>
internal static class Refusals
{
    public static CancellationToken Ct => TestContext.Current.CancellationToken;

    /// <summary>
    /// Runs <paramref name="call"/>, expects the client to throw for HTTP <paramref name="status"/>, and checks the body is
    /// <c>{"error":{"message","type":"roost_error","param","code"}}</c> with the given <paramref name="code"/> and
    /// <paramref name="param"/>. The client's own exception message carries the server's message.
    /// </summary>
    public static async Task<JsonElement> AssertRefusedAsync(Func<Task> call, int status, string code, string? param)
    {
        var e = await Assert.ThrowsAsync<ClientResultException>(call);

        Assert.Equal(status, e.Status);
        var raw = e.GetRawResponse();
        Assert.NotNull(raw);
        using var doc = JsonDocument.Parse(raw.Content.ToString());
        var error = Assert.Single(doc.RootElement.EnumerateObject(), p => p.Name == "error").Value;
        Assert.Single(doc.RootElement.EnumerateObject());
        Assert.Equal(["message", "type", "param", "code"], error.EnumerateObject().Select(p => p.Name));
        Assert.Equal("roost_error", error.GetProperty("type").GetString());
        Assert.Equal(code, error.GetProperty("code").GetString());
        var message = error.GetProperty("message").GetString();
        Assert.False(string.IsNullOrWhiteSpace(message));
        Assert.Contains(message!, e.Message, StringComparison.Ordinal);
        if (param is null)
        {
            Assert.Equal(JsonValueKind.Null, error.GetProperty("param").ValueKind);
        }
        else
        {
            Assert.Equal(param, error.GetProperty("param").GetString());
        }

        return error.Clone();
    }
}

using System.Net;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>vLLM's utility routes (<c>/tokenize</c>, <c>/detokenize</c>, <c>/health</c>, <c>/ping</c>, <c>/version</c>), <c>/ready</c>, and the prompt-embeddings refusal.</summary>
public sealed class UtilityRoutesTests : IAsyncLifetime
{
    private ServerRig? _rig;

    private ServerRig Rig => _rig!;

    public async ValueTask InitializeAsync() => _rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

    public async ValueTask DisposeAsync()
    {
        if (_rig is not null)
        {
            await _rig.DisposeAsync();
        }
    }

    [Fact]
    public async Task HealthAndPingAnswer200WithAnEmptyBody()
    {
        foreach (var request in new[] { new HttpRequestMessage(HttpMethod.Get, "/health"), new HttpRequestMessage(HttpMethod.Get, "/ping"), new HttpRequestMessage(HttpMethod.Post, "/ping") })
        {
            using (request)
            {
                using var response = await Rig.Http.SendAsync(request, Ct);
                Assert.Equal(HttpStatusCode.OK, response.StatusCode);
                Assert.Empty(await response.Content.ReadAsByteArrayAsync(Ct));
            }
        }
    }

    [Fact]
    public async Task ReadyAndVersionAnswerJson()
    {
        var (ready, readyBody) = await GetAsync(Rig.Http, "/ready");
        Assert.Equal(HttpStatusCode.OK, ready.StatusCode);
        Assert.Equal("ready", readyBody.GetProperty("status").GetString());
        Assert.Equal(ServerRig.ModelId, readyBody.GetProperty("model").GetString());

        var (version, versionBody) = await GetAsync(Rig.Http, "/version");
        Assert.Equal(HttpStatusCode.OK, version.StatusCode);
        Assert.False(string.IsNullOrWhiteSpace(versionBody.GetProperty("version").GetString()));
    }

    [Fact]
    public async Task TokenizeAPromptAddsSpecialTokensByDefaultAsVllmDoes()
    {
        var (response, body) = await PostAsync(Rig.Http, "/tokenize", """{"model":"roost-tiny-hybrid-test","prompt":"Hello, world","return_token_strs":true}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        int[] expected = [.. ServerRig.Tokenizer.Encode("Hello, world", true).Ids];
        Assert.Equal(["count", "max_model_len", "tokens", "token_strs"], body.EnumerateObject().Select(p => p.Name));
        Assert.Equal(expected, body.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()));
        Assert.Equal(expected.Length, body.GetProperty("count").GetInt32());
        Assert.Equal(ServerRig.AttentionBlocks * 16, body.GetProperty("max_model_len").GetInt32());
        Assert.Equal(expected.Select(id => ServerRig.Tokenizer.IdToToken(id)), body.GetProperty("token_strs").EnumerateArray().Select(t => t.GetString()));

        var (_, plain) = await PostAsync(Rig.Http, "/tokenize", """{"prompt":"Hello, world","add_special_tokens":false}""");
        Assert.Equal(ServerRig.Encode("Hello, world"), plain.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()));
        Assert.Equal(JsonValueKind.Null, plain.GetProperty("token_strs").ValueKind);
    }

    [Fact]
    public async Task TokenizeMessagesRendersTheChatTemplate()
    {
        var (response, body) = await PostAsync(Rig.Http, "/tokenize",
            """{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"add_generation_prompt":true,"continue_final_message":false}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var expected = ServerRig.Encode(Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.System("Be brief."), RoostChatMessage.User("Hi")]));
        Assert.Equal(expected, body.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()));
        Assert.Empty(Rig.Executor.Steps); // tokenising runs no step
    }

    [Theory]
    [InlineData("""{"prompt":"x","frobnicate":1}""", "frobnicate")]
    [InlineData("""{"messages":[{"role":"user","content":"x"}],"add_generation_prompt":false}""", "add_generation_prompt")]
    [InlineData("""{"messages":[{"role":"user","content":"x"}],"continue_final_message":true}""", "continue_final_message")]
    [InlineData("""{"messages":[{"role":"user","content":"x"}],"chat_template":"{{x}}"}""", "chat_template")]
    [InlineData("""{"messages":[{"role":"user","content":"x"}],"tools":[]}""", "tools")]
    [InlineData("""{"messages":[{"role":"tool","content":"x"}]}""", "messages[0].role")]
    [InlineData("""{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}""", "messages[0].content[0].type")]
    [InlineData("""{"prompt":"x","messages":[{"role":"user","content":"x"}]}""", "prompt")]
    [InlineData("""{}""", "prompt")]
    [InlineData("""{"prompt":3}""", "prompt")]
    [InlineData("""{"model":"other","prompt":"x"}""", "model")]
    public async Task ATokenizeFieldIsRefusedByName(string json, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/tokenize", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
    }

    [Fact]
    public async Task DetokenizeDecodesIdsAndKeepsSpecialTokens()
    {
        var ids = ServerRig.Encode("Hello, world! 中文");
        var (response, body) = await PostAsync(Rig.Http, "/detokenize", $$"""{"model":"roost-tiny-hybrid-test","tokens":[{{string.Join(',', ids)}}]}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("Hello, world! 中文", body.GetProperty("prompt").GetString());

        var (_, special) = await PostAsync(Rig.Http, "/detokenize", $$"""{"tokens":[{{ServerRig.StopId}}]}""");
        Assert.Equal(ServerRig.Tokenizer.Decode([ServerRig.StopId], false), special.GetProperty("prompt").GetString());
        Assert.Equal("<|im_end|>", special.GetProperty("prompt").GetString());
    }

    [Theory]
    [InlineData("""{"tokens":[1,999999]}""", "tokens[1]")]
    [InlineData("""{"tokens":[-1]}""", "tokens[0]")]
    [InlineData("""{"tokens":["a"]}""", "tokens[0]")]
    [InlineData("""{"tokens":"a"}""", "tokens")]
    [InlineData("""{}""", "tokens")]
    [InlineData("""{"tokens":[1],"skip_special_tokens":true}""", "skip_special_tokens")]
    public async Task ADetokenizeFieldIsRefusedByName(string json, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/detokenize", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
    }

    [Fact]
    public async Task PromptEmbeddingsAreRefusedByName()
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/completions", """{"model":"roost-tiny-hybrid-test","prompt_embeds":"AAAA"}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = AssertErrorBody(body, "invalid_option", "prompt_embeds");
        Assert.Contains("embedding", error.GetProperty("message").GetString(), StringComparison.OrdinalIgnoreCase);
    }
}

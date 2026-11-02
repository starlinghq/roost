using System.Net;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>Every failure has the one error body, and an unsupported or unknown option is a 400 naming it.</summary>
public sealed class ErrorTests : IAsyncLifetime
{
    private const string User = """{"role":"user","content":"Hi"}""";

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

    [Theory]
    // Unknown and unsupported top-level options.
    [InlineData("\"frobnicate\":1", "frobnicate")]
    [InlineData("\"reasoning_effort\":\"low\"", "reasoning_effort")]
    [InlineData("\"modalities\":[\"text\",\"audio\"]", "modalities")]
    [InlineData("\"temperature\":0.7", "temperature")]
    [InlineData("\"top_p\":0.9", "top_p")]
    [InlineData("\"top_k\":40", "top_k")]
    [InlineData("\"frequency_penalty\":0.5", "frequency_penalty")]
    [InlineData("\"presence_penalty\":0.5", "presence_penalty")]
    [InlineData("\"n\":2", "n")]
    [InlineData("\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"strict\":true}}]", "tools[0].function.strict")]
    [InlineData("\"tools\":[{\"type\":\"custom\",\"custom\":{\"name\":\"f\"}}]", "tools[0].custom")]
    [InlineData("\"tool_choice\":\"required\"", "tool_choice")]
    [InlineData("\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}", "tool_choice")]
    [InlineData("\"chat_template_kwargs\":{\"add_generation_prompt\":false}", "chat_template_kwargs.add_generation_prompt")]
    [InlineData("\"chat_template_kwargs\":{\"enable_thinking\":\"yes\"}", "chat_template_kwargs.enable_thinking")]
    [InlineData("\"parallel_tool_calls\":false", "parallel_tool_calls")]
    [InlineData("\"response_format\":{\"type\":\"json_object\"}", "response_format")]
    [InlineData("\"logprobs\":true", "logprobs")]
    [InlineData("\"top_logprobs\":2", "top_logprobs")]
    [InlineData("\"logit_bias\":{\"50\":1}", "logit_bias")]
    [InlineData("\"store\":true", "store")]
    [InlineData("\"max_tokens\":0", "max_tokens")]
    [InlineData("\"max_tokens\":4,\"max_completion_tokens\":4", "max_completion_tokens")]
    [InlineData("\"stream_options\":{\"include_usage\":true}", "stream_options")]
    [InlineData("\"stream\":true,\"stream_options\":{\"continuous_usage_stats\":true}", "stream_options.continuous_usage_stats")]
    [InlineData("\"stop\":[\"\"]", "stop")]
    [InlineData("\"stream\":\"yes\"", "stream")]
    public async Task AChatOptionIsRefusedByName(string extra, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/chat/completions",
            $$"""{"model":"roost-tiny-hybrid-test","messages":[{{User}}],{{extra}}}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
        Assert.Empty(Rig.Executor.Steps); // refused before any work
    }

    [Theory]
    [InlineData("""{"model":"another-model","messages":[{"role":"user","content":"Hi"}]}""", "model")]
    [InlineData("""{"messages":[{"role":"user","content":"Hi"}]}""", "model")]
    [InlineData("""{"model":"roost-tiny-hybrid-test"}""", "messages")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[]}""", "messages")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}""", "messages[0].content[0].image_url")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":[{"type":"input_audio"}]}]}""", "messages[0].content[0].type")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"42","tool_call_id":"c1"}]}""", "messages[0].tool_call_id")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"tool","content":"42"}]}""", "messages[0].tool_call_id")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi","reasoning_content":"x"}]}""", "messages[0].reasoning_content")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"[1]"}}]}]}""", "messages[1].tool_calls[0].function.arguments")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{}"}}]}]}""", "messages[1].tool_calls[0].id")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"oracle","content":"42"}]}""", "messages[0].role")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi","name":"ann"}]}""", "messages[0].name")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"},{"content":"no role"}]}""", "messages[1].role")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"x","tool_calls":[]}]}""", "messages[0].tool_calls")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}]}""", "model")]
    public async Task AMessageOrRequiredFieldProblemIsRefusedByName(string json, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/chat/completions", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
    }

    [Theory]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":"x","suffix":"y"}""", "suffix")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":"x","echo":true}""", "echo")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":"x","best_of":2}""", "best_of")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":"x","logprobs":1}""", "logprobs")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":[1,2,3]}""", "prompt")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":["a","b"]}""", "prompt")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":""}""", "prompt")]
    [InlineData("""{"model":"roost-tiny-hybrid-test"}""", "prompt")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":"x","messages":[]}""", "messages")]
    [InlineData("""{"model":"roost-tiny-hybrid-test","prompt":"x","temperature":1.0}""", "temperature")]
    public async Task ACompletionOptionIsRefusedByName(string json, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/completions", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
    }

    [Theory]
    [InlineData("{not json")]
    [InlineData("[1,2]")]
    [InlineData("")]
    public async Task ABodyThatIsNotAJsonObjectIsA400WithTheErrorBody(string json)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/chat/completions", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", null);
    }

    [Fact]
    public async Task AnUnknownRouteIsA404WithTheErrorBody()
    {
        var response = await Rig.Http.GetAsync("/v1/embeddings", Ct);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        AssertErrorBody(doc.RootElement, "not_found", null);
    }

    [Fact]
    public async Task AWrongMethodIsA405WithTheErrorBody()
    {
        var response = await Rig.Http.GetAsync("/v1/chat/completions", Ct);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal(HttpStatusCode.MethodNotAllowed, response.StatusCode);
        AssertErrorBody(doc.RootElement, "method_not_allowed", null);
    }
}

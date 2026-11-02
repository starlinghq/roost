using System.Net;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Plans;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;
using static Starlings.Roost.Server.Tests.Support.Prometheus;

namespace Starlings.Roost.Server.Tests;

/// <summary><c>GET /metrics</c> (R-038): the Roost families and their vLLM aliases, in Prometheus text format.</summary>
public sealed class MetricsTests
{
    private const string Reply = "Hello, world!";
    private const string Hi = """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}]""";

    private static readonly (string Roost, string[] Vllm, string Type)[] s_families =
    [
        ("roost_requests_total", ["vllm:request_success_total"], "counter"),
        ("roost_prompt_tokens_total", ["vllm:prompt_tokens_total"], "counter"),
        ("roost_generation_tokens_total", ["vllm:generation_tokens_total"], "counter"),
        ("roost_request_prompt_tokens", ["vllm:request_prompt_tokens"], "histogram"),
        ("roost_request_generation_tokens", ["vllm:request_generation_tokens"], "histogram"),
        ("roost_time_to_first_token_seconds", ["vllm:time_to_first_token_seconds"], "histogram"),
        ("roost_inter_token_latency_seconds", ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"], "histogram"),
        ("roost_e2e_request_latency_seconds", ["vllm:e2e_request_latency_seconds"], "histogram"),
        ("roost_requests_running", ["vllm:num_requests_running"], "gauge"),
        ("roost_requests_waiting", ["vllm:num_requests_waiting"], "gauge"),
        ("roost_kv_cache_usage_ratio", ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"], "gauge"),
    ];

    private static readonly (string, string) Model = ("model_name", ServerRig.ModelId);

    private static int PromptTokens(string user) =>
        ServerRig.Encode(Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User(user)])).Length;

    private static async Task<(List<PromSample> Samples, Dictionary<string, string> Types)> ScrapeAsync(ServerRig rig)
    {
        var response = await rig.Http.GetAsync("/metrics", Ct);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/plain", response.Content.Headers.ContentType?.MediaType);
        Assert.Contains(response.Content.Headers.ContentType!.Parameters, p => p.Name == "version" && p.Value == "0.0.4");
        return Parse(await response.Content.ReadAsStringAsync(Ct));
    }

    [Fact]
    public async Task BeforeAnyRequestEveryFamilyIsDeclaredUnderBothNamesAndCountsZero()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (samples, types) = await ScrapeAsync(rig);

        foreach (var (roost, vllm, type) in s_families)
        {
            Assert.Equal(type, types[roost]);
            Assert.All(vllm, name => Assert.Equal(type, types[name]));
        }

        Assert.All(samples, s => Assert.Equal(ServerRig.ModelId, s.Labels["model_name"]));
        foreach (var reason in RoostTelemetry.FinishReasons)
        {
            Assert.Equal(0, Value(samples, "roost_requests_total", Model, ("finish_reason", reason)));
        }

        // vLLM's success counter has no error series and names the label finished_reason.
        Assert.Equal(["abort", "length", "stop"], samples.Where(s => s.Name == "vllm:request_success_total").Select(s => s.Labels["finished_reason"]).Order());
        Assert.Equal(0, Value(samples, "vllm:num_requests_running", Model));
        Assert.Equal(0, Value(samples, "vllm:num_requests_waiting", Model));
        Assert.Equal(0, Value(samples, "roost_prompt_tokens_total", Model));
    }

    [Fact]
    public async Task AChatCompletionIsCountedTimedAndAliased()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });
        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + "}");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        int prompt = body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();
        int completion = body.GetProperty("usage").GetProperty("completion_tokens").GetInt32();
        Assert.Equal(PromptTokens("Hi"), prompt);

        var (samples, _) = await ScrapeAsync(rig);

        Assert.Equal(1, Value(samples, "roost_requests_total", ("finish_reason", "stop")));
        Assert.Equal(1, Value(samples, "vllm:request_success_total", Model, ("finished_reason", "stop")));
        Assert.Equal(0, Value(samples, "roost_requests_total", ("finish_reason", "length")));
        foreach (var name in new[] { "roost_prompt_tokens_total", "vllm:prompt_tokens_total" })
        {
            Assert.Equal(prompt, Value(samples, name));
        }

        foreach (var name in new[] { "roost_generation_tokens_total", "vllm:generation_tokens_total" })
        {
            Assert.Equal(completion, Value(samples, name));
        }

        foreach (var name in new[] { "roost_time_to_first_token_seconds", "vllm:time_to_first_token_seconds", "vllm:e2e_request_latency_seconds", "vllm:request_prompt_tokens" })
        {
            AssertHistogram(samples, name, expectedCount: 1);
        }

        Assert.Equal(prompt, Value(samples, "vllm:request_prompt_tokens_sum"));
        Assert.Equal(completion, Value(samples, "vllm:request_generation_tokens_sum"));

        // One interval per token after the first event (the scripted worker emits one token a step).
        var itl = AssertHistogram(samples, "vllm:inter_token_latency_seconds", expectedCount: null);
        Assert.InRange(itl, 1, completion - 1);
        Assert.Equal(itl, AssertHistogram(samples, "roost_inter_token_latency_seconds", expectedCount: null));
        Assert.Equal(itl, AssertHistogram(samples, "vllm:time_per_output_token_seconds", expectedCount: null));

        Assert.Equal(0, Value(samples, "vllm:num_requests_running"));
        Assert.Equal(0, Value(samples, "vllm:num_requests_waiting"));
        Assert.InRange(Value(samples, "vllm:kv_cache_usage_perc"), 0, 1);
        Assert.Equal(Value(samples, "roost_kv_cache_usage_ratio"), Value(samples, "vllm:gpu_cache_usage_perc"));
    }

    [Fact]
    public async Task LengthAndAbortAreCountedApart()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("abcdef"), NeverStop = true });
        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + ""","max_tokens":2}""");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());

        // Abandoning the enumeration after the first update is an abort.
        await foreach (var update in rig.Client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "Hi")], new ChatOptions { MaxOutputTokens = 100 }, Ct))
        {
            Assert.NotEmpty(update.Text);
            Assert.Equal(1, rig.Client.Telemetry.RunningRequests);
            break;
        }

        var (samples, _) = await ScrapeAsync(rig);
        Assert.Equal(1, Value(samples, "roost_requests_total", ("finish_reason", "length")));
        Assert.Equal(1, Value(samples, "vllm:request_success_total", ("finished_reason", "length")));
        Assert.Equal(1, Value(samples, "roost_requests_total", ("finish_reason", "abort")));
        Assert.Equal(1, Value(samples, "vllm:request_success_total", ("finished_reason", "abort")));
        Assert.Equal(0, Value(samples, "roost_requests_total", ("finish_reason", "stop")));
        Assert.Equal(0, Value(samples, "roost_requests_running"));
        Assert.Equal(0, Value(samples, "roost_requests_waiting"));
        AssertHistogram(samples, "roost_e2e_request_latency_seconds", expectedCount: 2);
    }

    [Fact]
    public async Task AFailedRequestIsAnErrorThatVllmsSuccessCounterLeavesOut()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply), FaultAtStep = 2, Fault = FaultClass.Recoverable });
        var (response, _) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + "}");
        Assert.NotEqual(HttpStatusCode.OK, response.StatusCode);

        var (samples, _) = await ScrapeAsync(rig);
        Assert.Equal(1, Value(samples, "roost_requests_total", ("finish_reason", "error")));
        Assert.DoesNotContain(samples, s => s.Name == "vllm:request_success_total" && s.Labels["finished_reason"] == "error");
        Assert.All(samples.Where(s => s.Name == "vllm:request_success_total"), s => Assert.Equal(0, s.Value));
        Assert.Equal(PromptTokens("Hi"), Value(samples, "roost_prompt_tokens_total"));
        Assert.Equal(0, Value(samples, "roost_requests_waiting"));
        Assert.Equal(0, Value(samples, "roost_requests_running"));
    }

    [Fact]
    public async Task RefusedRequestsRecordNothing()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });
        var (response, _) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + ""","temperature":0.7}""");
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var (samples, _) = await ScrapeAsync(rig);
        Assert.All(samples.Where(s => s.Name == "roost_requests_total"), s => Assert.Equal(0, s.Value));
        AssertHistogram(samples, "roost_e2e_request_latency_seconds", expectedCount: 0);
    }

    /// <summary>Checks a histogram's buckets (cumulative, ending in +Inf equal to the count) and returns its count.</summary>
    private static double AssertHistogram(List<PromSample> samples, string name, int? expectedCount)
    {
        var buckets = samples.Where(s => s.Name == name + "_bucket").ToList();
        Assert.True(buckets.Count > 1, $"{name} has no buckets");
        var bounds = buckets.Select(b => b.Labels["le"]).ToList();
        Assert.Equal("+Inf", bounds[^1]);
        for (var i = 1; i < buckets.Count; i++)
        {
            Assert.True(buckets[i].Value >= buckets[i - 1].Value, $"{name} buckets are not cumulative");
        }

        double count = Value(samples, name + "_count");
        Assert.Equal(count, buckets[^1].Value);
        Assert.True(Value(samples, name + "_sum") >= 0);
        if (expectedCount is { } expected)
        {
            Assert.Equal(expected, count);
        }

        return count;
    }
}

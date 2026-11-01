using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Starlings.Generation.Roost;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>R-038: one activity per request with its id and token counts, and no prompt or output text in any log.</summary>
public sealed class TracesAndLogsTests
{
    // Distinctive strings that must never reach a log or a tag: the prompt and the scripted reply.
    private const string Secret = "zebra-marker-7781";
    private const string Reply = "quokka-reply-5521";
    private const string Model = "\"model\":\"roost-tiny-hybrid-test\"";

    [Fact]
    public async Task EveryRequestRecordsOneActivityWithItsIdAndTokenCountsAndNoText()
    {
        var stopped = new ConcurrentQueue<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = source => source.Name == RoostTelemetry.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = stopped.Enqueue,
        };
        ActivitySource.AddActivityListener(listener);
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (chat, chatBody) = await PostAsync(rig.Http, "/v1/chat/completions", $$"""{{{Model}},"messages":[{"role":"user","content":"{{Secret}}"}]}""");
        var (completion, completionBody) = await PostAsync(rig.Http, "/v1/completions", $$"""{{{Model}},"prompt":"{{Secret}}","max_tokens":3}""");
        Assert.Equal(HttpStatusCode.OK, chat.StatusCode);
        Assert.Equal(HttpStatusCode.OK, completion.StatusCode);

        var activities = stopped.Where(a => a.OperationName == RoostTelemetry.RequestActivityName).ToList();
        Assert.Equal(2, activities.Count);
        AssertActivity(activities[0], chatBody, "chatcmpl-", "stop");
        AssertActivity(activities[1], completionBody, "cmpl-", "length");
        Assert.All(activities.SelectMany(a => a.TagObjects), t => Assert.DoesNotContain(Secret, Convert.ToString(t.Value, System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty, StringComparison.Ordinal));
        Assert.All(activities.SelectMany(a => a.TagObjects), t => Assert.DoesNotContain(Reply[..6], Convert.ToString(t.Value, System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty, StringComparison.Ordinal));
    }

    [Fact]
    public async Task NoPromptOrCompletionTextReachesTheLogsAtAnyLevel()
    {
        var logs = new CapturingLoggerProvider();
        await using (var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) }, configureWebHost: web => web.ConfigureServices(services => services.AddLogging(l =>
        {
            l.SetMinimumLevel(LogLevel.Trace);
            l.AddProvider(logs);
        }))))
        {
            string user = $$"""[{"role":"user","content":"{{Secret}}"}]""";
            await AssertOkAsync(rig, "/v1/chat/completions", $$"""{{{Model}},"messages":{{user}}}""");
            await AssertOkAsync(rig, "/v1/chat/completions", $$"""{{{Model}},"messages":{{user}},"stream":true}""", json: false);
            await AssertOkAsync(rig, "/v1/completions", $$"""{{{Model}},"prompt":"{{Secret}}","max_tokens":30}""");
            await AssertOkAsync(rig, "/v1/responses", $$"""{{{Model}},"input":"{{Secret}}"}""");
            await AssertOkAsync(rig, "/v1/messages", $$"""{{{Model}},"max_tokens":30,"messages":{{user}}}""");
            await AssertOkAsync(rig, "/v1/messages", $$"""{{{Model}},"max_tokens":30,"messages":{{user}},"stream":true}""", json: false);
            await AssertOkAsync(rig, "/tokenize", $$"""{{{Model}},"prompt":"{{Secret}}"}""");
            await AssertOkAsync(rig, "/detokenize", $$"""{{{Model}},"tokens":[{{string.Join(',', ServerRig.Encode(Secret))}}]}""");

            // Refusals and failures: an unknown field whose value is the secret, a refused option, a body that is not JSON.
            var (unknown, _) = await PostAsync(rig.Http, "/v1/chat/completions", $$"""{{{Model}},"messages":{{user}},"{{Secret}}":"{{Secret}}"}""");
            Assert.Equal(HttpStatusCode.BadRequest, unknown.StatusCode);
            var (refused, _) = await PostAsync(rig.Http, "/v1/chat/completions", $$"""{{{Model}},"messages":{{user}},"temperature":0.5}""");
            Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
            var notJson = await rig.Http.PostAsync("/v1/chat/completions", Json("{" + Secret), Ct);
            Assert.Equal(HttpStatusCode.BadRequest, notJson.StatusCode);
            Assert.Equal(HttpStatusCode.OK, (await rig.Http.GetAsync("/metrics", Ct)).StatusCode);
        }

        var entries = logs.Entries.ToList();
        Assert.NotEmpty(entries); // the capture works: ASP.NET Core's hosting logs every request
        Assert.Contains(entries, e => e.Contains("/v1/chat/completions", StringComparison.Ordinal));
        Assert.All(entries, e =>
        {
            Assert.DoesNotContain(Secret, e, StringComparison.Ordinal);
            Assert.DoesNotContain(Reply, e, StringComparison.Ordinal);
            Assert.DoesNotContain("quokka", e, StringComparison.Ordinal); // a partial reply chunk
        });
    }

    private static async Task AssertOkAsync(ServerRig rig, string path, string body, bool json = true)
    {
        var response = await rig.Http.PostAsync(path, Json(body), Ct);
        var text = await response.Content.ReadAsStringAsync(Ct);
        Assert.True(response.StatusCode == HttpStatusCode.OK, $"{path}: {(int)response.StatusCode} {text}");
        if (json)
        {
            using var _ = JsonDocument.Parse(text);
        }
    }

    private static void AssertActivity(Activity activity, JsonElement body, string idPrefix, string finish)
    {
        var usage = body.GetProperty("usage");
        Assert.Equal(body.GetProperty("id").GetString(), idPrefix + activity.GetTagItem(RoostTelemetry.RequestIdTag));
        Assert.Equal(ServerRig.ModelId, activity.GetTagItem("gen_ai.request.model"));
        Assert.Equal(usage.GetProperty("prompt_tokens").GetInt32(), activity.GetTagItem("gen_ai.usage.input_tokens"));
        Assert.Equal(usage.GetProperty("completion_tokens").GetInt32(), activity.GetTagItem("gen_ai.usage.output_tokens"));
        Assert.Equal(finish, activity.GetTagItem("gen_ai.response.finish_reasons"));
        Assert.Equal(ActivityStatusCode.Unset, activity.Status);
    }

    /// <summary>Captures every log entry (message, state and exception text) of every category.</summary>
    private sealed class CapturingLoggerProvider : ILoggerProvider
    {
        public ConcurrentQueue<string> Entries { get; } = new();

        public ILogger CreateLogger(string categoryName) => new Logger(categoryName, Entries);

        public void Dispose()
        {
        }

        private sealed class Logger(string category, ConcurrentQueue<string> entries) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state)
                where TState : notnull
            {
                entries.Enqueue(category + " scope: " + state);
                return null;
            }

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                var structured = state is IEnumerable<KeyValuePair<string, object?>> pairs ? string.Join(';', pairs.Select(p => p.Key + "=" + p.Value)) : string.Empty;
                entries.Enqueue($"{category} {logLevel} {formatter(state, exception)} {structured} {exception}");
            }
        }
    }
}

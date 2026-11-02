using Starlings.Generation.Roost;
using Starlings.Roost.Server.Anthropic;
using Starlings.Roost.Server.Metrics;
using Starlings.Roost.Server.OpenAi;
using Starlings.Roost.Server.Pooling;
using Starlings.Roost.Server.Responses;
using Starlings.Roost.Server.Utility;

namespace Starlings.Roost.Server;

/// <summary>
/// Roost's HTTP server (R-037, R-074): an ASP.NET Core minimal API over a <see cref="RoostChatClient"/>, with the OpenAI routes
/// <c>POST /v1/chat/completions</c>, <c>POST /v1/completions</c>, <c>GET /v1/models</c> and the Responses API
/// (<c>POST /v1/responses</c>, <c>GET /v1/responses/{id}</c>, <c>POST /v1/responses/{id}/cancel</c>), the Anthropic Messages API
/// (<c>POST /v1/messages</c>, <c>POST /v1/messages/count_tokens</c>), and vLLM's utility routes (<c>POST /tokenize</c>,
/// <c>POST /detokenize</c>, <c>GET /health</c>, <c>/ping</c>, <c>GET /version</c>) with Roost's <c>GET /ready</c>, and
/// <c>GET /metrics</c> in Prometheus text format with vLLM's metric names as aliases (R-038), and the pooling routes
/// <c>POST /v1/embeddings</c> and <c>POST /v1/extract</c> over <see cref="RoostServerOptions.Pooling"/>, with <c>POST /v1/rerank</c>
/// and <c>POST /v1/classify</c> answering <c>profile_unsupported</c> (R-052). It listens on
/// loopback unless non-loopback listening is allowed explicitly (<see cref="RoostServerOptions.AllowNonLoopback"/>). Every
/// failure, including an unknown route, answers with one error body, <c>{"error":{"message","type","param","code"}}</c>, except
/// on the Anthropic routes, which answer with Anthropic's; an unsupported or unknown request option is a 400 that names it.
/// With <see cref="RoostServerOptions.EnableSleepMode"/> it also maps the sleep-mode routes (<see cref="AdminRoutes"/>, R-100),
/// which answer loopback clients only.
/// </summary>
public static class RoostServer
{
    /// <summary>Builds the server; start it with <c>RunAsync</c> or <c>StartAsync</c>. The caller owns the client.</summary>
    /// <param name="client">The chat client that serves the model.</param>
    /// <param name="options">Where to listen; null means loopback on the default port.</param>
    /// <param name="configureWebHost">Further web host settings (tests replace the server with an in-memory one).</param>
    /// <returns>The application.</returns>
    /// <exception cref="ArgumentException">The listen address is invalid, or not loopback without <see cref="RoostServerOptions.AllowNonLoopback"/>,
    /// or two pooling models of one kind share an id.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The response store's capacity is below 1 or its expiry is not positive.</exception>
    public static WebApplication Create(RoostChatClient client, RoostServerOptions? options = null, Action<IWebHostBuilder>? configureWebHost = null)
    {
        ArgumentNullException.ThrowIfNull(client);
        options ??= new RoostServerOptions();
        var address = options.ResolveAddress();
        PoolingRoutes.Validate(options.Pooling);
        var store = new ResponseStore(options.ResponseStoreCapacity, options.ResponseStoreTtl, options.TimeProvider);

        // The slim builder (no reflection-heavy defaults) is the one S1 published with Native AOT. No command-line or URL
        // configuration reaches Kestrel's endpoints: the only listener is the one checked above.
        var builder = WebApplication.CreateSlimBuilder(new WebApplicationOptions { Args = [] });
        builder.WebHost.ConfigureKestrel(kestrel => kestrel.Listen(address, options.Port));
        configureWebHost?.Invoke(builder.WebHost);

        var app = builder.Build();
        app.UseStatusCodePages(ErrorBodies.StatusCodePageAsync);
        var routes = new OpenAiRoutes(client, DateTimeOffset.UtcNow, app.Logger);
        app.MapPost("/v1/chat/completions", (RequestDelegate)routes.ChatCompletionsAsync);
        app.MapPost("/v1/completions", (RequestDelegate)routes.CompletionsAsync);
        app.MapGet("/v1/models", (RequestDelegate)routes.ModelsAsync);

        var stopping = app.Lifetime.ApplicationStopping;
        var responses = new ResponsesRoutes(client, store, app.Logger, stopping);
        app.MapPost("/v1/responses", (RequestDelegate)responses.CreateAsync);
        app.MapGet("/v1/responses/{id}", (RequestDelegate)responses.RetrieveAsync);
        app.MapPost("/v1/responses/{id}/cancel", (RequestDelegate)responses.CancelAsync);

        var messages = new MessagesRoutes(client, app.Logger);
        app.MapPost("/v1/messages", (RequestDelegate)messages.MessagesAsync);
        app.MapPost("/v1/messages/count_tokens", (RequestDelegate)messages.CountTokensAsync);

        var utility = new UtilityRoutes(client, app.Logger, stopping);
        app.MapPost("/tokenize", (RequestDelegate)utility.TokenizeAsync);
        app.MapPost("/detokenize", (RequestDelegate)utility.DetokenizeAsync);
        app.MapGet("/health", (RequestDelegate)UtilityRoutes.HealthAsync);
        app.MapMethods("/ping", [HttpMethods.Get, HttpMethods.Post], (RequestDelegate)UtilityRoutes.HealthAsync);
        app.MapGet("/version", (RequestDelegate)utility.VersionAsync);
        app.MapGet("/ready", (RequestDelegate)utility.ReadyAsync);

        var pooling = new PoolingRoutes(options.Pooling, app.Logger);
        app.MapPost("/v1/embeddings", (RequestDelegate)pooling.EmbeddingsAsync);
        app.MapPost("/v1/extract", (RequestDelegate)pooling.ExtractAsync);
        app.MapPost("/v1/rerank", (RequestDelegate)pooling.RerankAsync);
        app.MapPost("/v1/classify", (RequestDelegate)pooling.ClassifyAsync);

        var metrics = new PrometheusExporter(client.Telemetry, client.Model.ModelId, options.ColdStart);
        _ = app.Lifetime.ApplicationStopped.Register(metrics.Dispose);
        app.MapGet("/metrics", (RequestDelegate)metrics.ScrapeAsync);

        if (options.EnableSleepMode)
        {
            new AdminRoutes(client, app.Logger).Map(app);
        }

        return app;
    }
}

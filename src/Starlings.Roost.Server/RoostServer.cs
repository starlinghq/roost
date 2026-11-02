using Starlings.Generation.Roost;
using Starlings.Roost.Server.OpenAi;

namespace Starlings.Roost.Server;

/// <summary>
/// Roost's HTTP server (R-037): an ASP.NET Core minimal API over a <see cref="RoostChatClient"/>, with the OpenAI routes
/// <c>POST /v1/chat/completions</c>, <c>POST /v1/completions</c> and <c>GET /v1/models</c>. It listens on loopback unless
/// non-loopback listening is allowed explicitly (<see cref="RoostServerOptions.AllowNonLoopback"/>). Every failure, including
/// an unknown route, answers with one error body, <c>{"error":{"message","type","param","code"}}</c>, and an unsupported or
/// unknown request option is a 400 whose <c>param</c> names it. With <see cref="RoostServerOptions.EnableSleepMode"/> it also maps
/// the sleep-mode routes (<see cref="AdminRoutes"/>, R-100), which answer loopback clients only.
/// </summary>
public static class RoostServer
{
    /// <summary>Builds the server; start it with <c>RunAsync</c> or <c>StartAsync</c>. The caller owns the client.</summary>
    /// <param name="client">The chat client that serves the model.</param>
    /// <param name="options">Where to listen; null means loopback on the default port.</param>
    /// <param name="configureWebHost">Further web host settings (tests replace the server with an in-memory one).</param>
    /// <returns>The application.</returns>
    /// <exception cref="ArgumentException">The listen address is invalid, or not loopback without <see cref="RoostServerOptions.AllowNonLoopback"/>.</exception>
    public static WebApplication Create(RoostChatClient client, RoostServerOptions? options = null, Action<IWebHostBuilder>? configureWebHost = null)
    {
        ArgumentNullException.ThrowIfNull(client);
        options ??= new RoostServerOptions();
        var address = options.ResolveAddress();

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
        if (options.EnableSleepMode)
        {
            new AdminRoutes(client, app.Logger).Map(app);
        }

        return app;
    }
}

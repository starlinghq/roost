using OpenAI.Embeddings;
using OpenAI.Models;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;
using static Starlings.Roost.ApiCompat.Tests.Support.Refusals;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary><c>GET /v1/models</c> through the official client, and routes Roost does not serve, as the client sees them.</summary>
public sealed class ModelsAndErrorsCompatTests : IAsyncLifetime
{
    private ApiCompatRig? _rig;

    private ApiCompatRig Rig => _rig!;

    public async ValueTask InitializeAsync() => _rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("ok") });

    public async ValueTask DisposeAsync()
    {
        if (_rig is not null)
        {
            await _rig.DisposeAsync();
        }
    }

    [Fact]
    public async Task ModelsListsTheServedModel()
    {
        OpenAIModelCollection models = await Rig.Models.GetModelsAsync(Ct);

        var model = Assert.Single(models);
        Assert.Equal(ApiCompatRig.ModelId, model.Id);
        Assert.Equal("roost", model.OwnedBy);
        Assert.True(model.CreatedAt > DateTimeOffset.UnixEpoch);
    }

    [Fact]
    public async Task RoutesRoostDoesNotServeAre404WithTheErrorBody()
    {
        // GET /v1/models/{id} is not built (the list is), nor are embeddings (R-052).
        await AssertRefusedAsync(() => Rig.Models.GetModelAsync(ApiCompatRig.ModelId, Ct), 404, "not_found", null);
        await AssertRefusedAsync(() => Rig.OpenAi.GetEmbeddingClient(ApiCompatRig.ModelId).GenerateEmbeddingAsync("Hi", cancellationToken: Ct), 404, "not_found", null);
    }

    [Fact]
    public async Task ARequestThatIsNotAJsonObjectIsA400ThroughTheClientsPipeline()
    {
        // The client's protocol method sends a body as given: Roost answers it with the one error body.
        var e = await AssertRefusedAsync(
            () => Rig.Chat.CompleteChatAsync(System.ClientModel.BinaryContent.Create(BinaryData.FromString("[1,2]")), new System.ClientModel.Primitives.RequestOptions { CancellationToken = Ct }),
            400,
            "invalid_option",
            null);
        Assert.Contains("object", e.GetProperty("message").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AnUnknownFieldIsRefusedByName()
    {
        var options = new OpenAI.Chat.ChatCompletionOptions();
        options.Patch.Set("$.frobnicate"u8, 1);

        await AssertRefusedAsync(() => Rig.Chat.CompleteChatAsync([new OpenAI.Chat.UserChatMessage("Hi")], options, Ct), 400, "invalid_option", "frobnicate");
    }
}

using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Starlings.Roost.Server.OpenAi;

/// <summary>
/// A function tool from a request body (R-053): its name, description and parameters schema for the chat template. The server
/// never invokes it; the caller runs the calls the model returns.
/// </summary>
internal sealed class DeclaredFunction : AIFunction
{
    private readonly string _name;
    private readonly string? _description;
    private readonly JsonElement _schema;

    public DeclaredFunction(string name, string? description, JsonElement schema)
    {
        _name = name;
        _description = description;
        _schema = schema;
    }

    /// <inheritdoc />
    public override string Name => _name;

    /// <inheritdoc />
    public override string Description => _description ?? string.Empty;

    /// <inheritdoc />
    public override JsonElement JsonSchema => _schema;

    /// <inheritdoc />
    protected override ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken) =>
        throw new NotSupportedException("A declared function is not invoked by the server; the caller runs the tool calls.");
}

using Microsoft.CodeAnalysis;

namespace Starlings.Roost.ModelCompiler;

/// <summary>The generator's diagnostics. Every message states what was found and where to look.</summary>
internal static class Diagnostics
{
    private const string Category = "Roost";

    private static DiagnosticDescriptor Error(string id, string title) =>
        new DiagnosticDescriptor(id, title, "{0}", Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    public static readonly DiagnosticDescriptor UnknownOp = Error("ROOST001", "Op is not a member of IRoostOps");

    public static readonly DiagnosticDescriptor UnknownValue = Error("ROOST002", "Name does not resolve to a value");

    public static readonly DiagnosticDescriptor UnknownField = Error("ROOST003", "Field is not a member of the op's argument struct");

    public static readonly DiagnosticDescriptor BadBinding = Error("ROOST004", "Argument binding is not valid for the field");

    public static readonly DiagnosticDescriptor OpsNotFound = Error("ROOST005", "IRoostOps is not referenced");

    public static readonly DiagnosticDescriptor BadShape = Error("ROOST006", "Shape or dimension is not valid");

    public static readonly DiagnosticDescriptor Duplicate = Error("ROOST007", "Name is declared twice");

    public static readonly DiagnosticDescriptor BadDefinition = Error("ROOST008", "Model definition is not valid");

    public static readonly DiagnosticDescriptor InPlaceHazard = Error("ROOST009", "Value is read after an op overwrote it");

    public static readonly DiagnosticDescriptor BadLayers = Error("ROOST010", "Layer set or placeholder is not valid");

    public static readonly DiagnosticDescriptor BadStateGroup = Error("ROOST011", "State group is not valid");

    public static readonly DiagnosticDescriptor NoDecomposition = Error("ROOST012", "Fused op has no known unfused form");

    public static readonly DiagnosticDescriptor MissingContractType = Error("ROOST013", "A contract type the generated code needs is not referenced");

    public static readonly DiagnosticDescriptor UnusedWeight = new DiagnosticDescriptor(
        "ROOST014", "Weight slot is never read", "{0}", Category, DiagnosticSeverity.Warning, isEnabledByDefault: true);
}

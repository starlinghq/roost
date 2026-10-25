// Clean consumer of the release drill (tools/release-drill.sh): restored from a local feed of the packed packages only,
// compiled outside the checkout. It checks that each library package's assembly loads and exports types, then runs a
// typed call per package it knows. A package added later (for example Starlings.Generation.Roost or a worker executable
// package, R-062) is restored and loaded by the generic part without a change here; add a typed block for it.
using System.Reflection;

var failures = new List<string>();

foreach (var name in AppContext.GetData("RoostDrillAssemblies") as string is { } list ? list.Split(';', StringSplitOptions.RemoveEmptyEntries) : [])
{
    try
    {
        var assembly = Assembly.Load(new AssemblyName(name));
        var types = assembly.GetExportedTypes().Length;
        Console.WriteLine($"ok:   {name} {assembly.GetName().Version} loads, {types} exported types");
        if (types == 0)
        {
            failures.Add($"{name} exports no types");
        }
    }
    catch (Exception e) when (e is FileNotFoundException or FileLoadException or BadImageFormatException)
    {
        failures.Add($"{name} does not load: {e.Message}");
    }
}

#if HAS_STARLINGS_ROOST_MODELS_QWEN35
// The model package ships the code the generator produced, so a consumer needs no generator (R-035).
var slots = Starlings.Roost.Models.Qwen35.Qwen35WeightMap.All();
Console.WriteLine($"ok:   Starlings.Roost.Models.Qwen35: {slots.Length} weight slots mapped to checkpoint tensors");
if (slots.Length == 0)
{
    failures.Add("Qwen35WeightMap.All() is empty");
}
#endif

#if HAS_STARLINGS_ROOST_ABSTRACTIONS
var error = new Starlings.Roost.Requests.RoostError(Starlings.Roost.Requests.RoostErrorCode.ProfileUnsupported, "drill");
Console.WriteLine($"ok:   Starlings.Roost.Abstractions: {error.Code} is a typed C6 error");
#endif

foreach (var failure in failures)
{
    Console.WriteLine("FAIL: " + failure);
}

Console.WriteLine(failures.Count == 0 ? "RESULT: PASS" : "RESULT: FAIL");
return failures.Count == 0 ? 0 : 1;

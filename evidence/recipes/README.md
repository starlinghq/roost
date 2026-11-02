# Roost recipes (roost-v1)

One JSON file per onboarded model, named `<id>.json`, in the strict roost-v1 schema
(`evidence/pipeline/onboarding/Starlings.Roost.Onboarding/Schemas/roost-recipe-v1.schema.json`). Stream I, build plan R-055.

A recipe pins the checkpoint by full commit SHA, records what `roost-onboard inspect` reported for its config, names the model
definition that implements it and the serving settings its evidence was produced under, and carries an explicit status with
the same meaning as in [docs/capabilities.md](../../docs/capabilities.md): nothing is `supported` before its fixtures and
tolerances are in `evidence/protected/`, and an SM without hardware evidence is `untested`.

```sh
dotnet run --project evidence/pipeline/onboarding/Starlings.Roost.Onboarding -- validate-recipe --root . evidence/recipes
```

`tests/Starlings.Roost.Onboarding.Tests` validates every file here in CI.

| Recipe | Status | Note |
| --- | --- | --- |
| `qwen3.5-0.8b.json` | `experimental` | The M0 profile (decision 15), written as the first recipe. Not one of the M3 gate's three unseen models (R-057), which are open. |

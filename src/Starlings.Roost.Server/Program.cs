// roost (R-037, R-046): `roost serve` starts the GPU worker for a model profile and serves the OpenAI routes on loopback;
// `roost doctor` checks the machine and the assets before a start.
var sinceProcessStart = Starlings.Roost.Server.ColdStart.SinceProcessStart();
return await Starlings.Roost.Server.RoostCli.RunAsync(args, Console.Out, Console.Error, sinceProcessStart: sinceProcessStart).ConfigureAwait(false);

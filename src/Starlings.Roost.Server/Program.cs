// roost (R-037): `roost serve` starts the GPU worker for a model profile and serves the OpenAI routes on loopback.
return await Starlings.Roost.Server.RoostCli.RunAsync(args, Console.Error).ConfigureAwait(false);

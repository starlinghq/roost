using System.Diagnostics.Metrics;
using System.Globalization;
using System.Text;
using Starlings.Generation.Roost;

namespace Starlings.Roost.Server.Metrics;

/// <summary>
/// <c>GET /metrics</c> (R-038): a <see cref="MeterListener"/> over one client's <see cref="RoostTelemetry.Meter"/> that keeps
/// counter sums, histogram buckets and gauge readings, and writes them in the Prometheus text exposition format (version 0.0.4).
/// Each instrument is exported under a Roost name (<c>roost_...</c>) and under vLLM's Prometheus names (<c>vllm:...</c>) so that
/// dashboards built for vLLM work unchanged; every series carries vLLM's <c>model_name</c> label. No exporter package is used.
/// </summary>
internal sealed class PrometheusExporter : IDisposable
{
    /// <summary>The content type of the text exposition format.</summary>
    public const string ContentType = "text/plain; version=0.0.4; charset=utf-8";

    // Instrument name, Roost family, type, help, vLLM aliases. vLLM names its finish label `finished_reason` and counts only
    // stop, length and abort in vllm:request_success_total; Roost's own family also counts `error`.
    private static readonly Family[] s_families =
    [
        new(RoostTelemetry.RequestsInstrument, "roost_requests_total", Kind.Counter, "Finished requests by finish reason.", ["vllm:request_success_total"], SeedFinishReasons: true),
        new(RoostTelemetry.PromptTokensInstrument, "roost_prompt_tokens_total", Kind.Counter, "Prompt tokens of finished requests.", ["vllm:prompt_tokens_total"]),
        new(RoostTelemetry.GenerationTokensInstrument, "roost_generation_tokens_total", Kind.Counter, "Generated tokens of finished requests.", ["vllm:generation_tokens_total"]),
        new(RoostTelemetry.RequestPromptTokensInstrument, "roost_request_prompt_tokens", Kind.Histogram, "Prompt tokens per finished request.", ["vllm:request_prompt_tokens"]),
        new(RoostTelemetry.RequestGenerationTokensInstrument, "roost_request_generation_tokens", Kind.Histogram, "Generated tokens per finished request.", ["vllm:request_generation_tokens"]),
        new(RoostTelemetry.TimeToFirstTokenInstrument, "roost_time_to_first_token_seconds", Kind.Histogram, "Time to first token in seconds.", ["vllm:time_to_first_token_seconds"]),
        new(RoostTelemetry.InterTokenLatencyInstrument, "roost_inter_token_latency_seconds", Kind.Histogram, "Inter-token latency in seconds.", ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"]),
        new(RoostTelemetry.E2eRequestLatencyInstrument, "roost_e2e_request_latency_seconds", Kind.Histogram, "End-to-end request latency in seconds.", ["vllm:e2e_request_latency_seconds"]),
        new(RoostTelemetry.RunningRequestsInstrument, "roost_requests_running", Kind.Gauge, "Requests generating.", ["vllm:num_requests_running"]),
        new(RoostTelemetry.WaitingRequestsInstrument, "roost_requests_waiting", Kind.Gauge, "Requests waiting for their first token.", ["vllm:num_requests_waiting"]),
        new(RoostTelemetry.KvCacheUsageInstrument, "roost_kv_cache_usage_ratio", Kind.Gauge, "Share of the KV cache's attention blocks in use (1 is full).", ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]),
    ];

    private readonly Lock _gate = new();
    private readonly Meter _meter;
    private readonly string _modelName;
    private readonly MeterListener _listener;
    private readonly Dictionary<string, State> _states = new(StringComparer.Ordinal);

    public PrometheusExporter(RoostTelemetry telemetry, string modelName)
    {
        ArgumentNullException.ThrowIfNull(telemetry);
        _meter = telemetry.Meter;
        _modelName = modelName;
        foreach (var family in s_families)
        {
            var state = new State(family);
            if (family.SeedFinishReasons)
            {
                foreach (var reason in RoostTelemetry.FinishReasons)
                {
                    _ = state.Get([new(RoostTelemetry.FinishReasonTag, reason)]);
                }
            }
            else if (family.Kind != Kind.Gauge)
            {
                _ = state.Get([]); // an untagged counter or histogram reads 0 before its first measurement
            }

            _states.Add(family.Instrument, state);
        }

        _listener = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (ReferenceEquals(instrument.Meter, _meter) && _states.TryGetValue(instrument.Name, out var state))
                {
                    lock (_gate)
                    {
                        state.Bounds = Bounds(instrument);
                    }

                    listener.EnableMeasurementEvents(instrument, state);
                }
            },
        };
        _listener.SetMeasurementEventCallback<long>((_, value, tags, s) => Record((State)s!, value, tags));
        _listener.SetMeasurementEventCallback<int>((_, value, tags, s) => Record((State)s!, value, tags));
        _listener.SetMeasurementEventCallback<double>((_, value, tags, s) => Record((State)s!, value, tags));
        _listener.Start();
    }

    /// <summary>Answers <c>GET /metrics</c> with the current values.</summary>
    public async Task ScrapeAsync(HttpContext context)
    {
        var body = Encoding.UTF8.GetBytes(Collect());
        context.Response.StatusCode = StatusCodes.Status200OK;
        context.Response.ContentType = ContentType;
        await context.Response.Body.WriteAsync(body, context.RequestAborted).ConfigureAwait(false);
    }

    /// <summary>Reads the gauges and writes every family in the text exposition format.</summary>
    public string Collect()
    {
        lock (_gate)
        {
            foreach (var state in _states.Values)
            {
                if (state.Family.Kind == Kind.Gauge)
                {
                    state.Series.Clear(); // a gauge with no reading (KV usage before the first step) has no sample
                }
            }
        }

        _listener.RecordObservableInstruments();

        var text = new StringBuilder();
        lock (_gate)
        {
            foreach (var family in s_families)
            {
                var state = _states[family.Instrument];
                Write(text, family.Name, family, state, renameFinish: false, skipError: false);
                foreach (var alias in family.Aliases)
                {
                    Write(text, alias, family, state, renameFinish: true, skipError: true);
                }
            }
        }

        return text.ToString();
    }

    public void Dispose() => _listener.Dispose();

    private static double[] Bounds(Instrument instrument) => instrument switch
    {
        Histogram<double> h => [.. h.Advice?.HistogramBucketBoundaries ?? []],
        Histogram<long> h => [.. (h.Advice?.HistogramBucketBoundaries ?? []).Select(b => (double)b)],
        _ => [],
    };

    private void Record(State state, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
    {
        lock (_gate)
        {
            var series = state.Get(Tags(tags));
            switch (state.Family.Kind)
            {
                case Kind.Counter:
                    series.Value += value;
                    break;
                case Kind.Gauge:
                    series.Value = value;
                    break;
                default:
                    series.Count++;
                    series.Sum += value;
                    series.Buckets ??= new long[state.Bounds.Length];
                    for (var i = 0; i < state.Bounds.Length; i++)
                    {
                        if (value <= state.Bounds[i])
                        {
                            series.Buckets[i]++;
                        }
                    }

                    break;
            }
        }
    }

    private static KeyValuePair<string, string>[] Tags(ReadOnlySpan<KeyValuePair<string, object?>> tags)
    {
        var result = new KeyValuePair<string, string>[tags.Length];
        for (var i = 0; i < tags.Length; i++)
        {
            result[i] = new(tags[i].Key, Convert.ToString(tags[i].Value, CultureInfo.InvariantCulture) ?? string.Empty);
        }

        Array.Sort(result, (a, b) => string.CompareOrdinal(a.Key, b.Key));
        return result;
    }

    private void Write(StringBuilder text, string name, Family family, State state, bool renameFinish, bool skipError)
    {
        string type = family.Kind switch
        {
            Kind.Counter => "counter",
            Kind.Gauge => "gauge",
            _ => "histogram",
        };

        // Counters keep the _total sample name on their HELP and TYPE lines, as Python's prometheus_client writes vLLM's.
        text.Append("# HELP ").Append(name).Append(' ').Append(EscapeHelp(family.Help)).Append('\n');
        text.Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n');
        foreach (var series in state.Series.Values)
        {
            if (skipError && series.Tags.Any(t => t.Key == RoostTelemetry.FinishReasonTag && t.Value == "error"))
            {
                continue;
            }

            var labels = Labels(series.Tags, renameFinish);
            if (family.Kind != Kind.Histogram)
            {
                Sample(text, name, labels, null, series.Value);
                continue;
            }

            for (var i = 0; i < state.Bounds.Length; i++)
            {
                Sample(text, name + "_bucket", labels, Number(state.Bounds[i]), series.Buckets?[i] ?? 0);
            }

            Sample(text, name + "_bucket", labels, "+Inf", series.Count);
            Sample(text, name + "_sum", labels, null, series.Sum);
            Sample(text, name + "_count", labels, null, series.Count);
        }
    }

    private string Labels(KeyValuePair<string, string>[] tags, bool renameFinish)
    {
        var labels = new StringBuilder();
        labels.Append("model_name=\"").Append(EscapeLabel(_modelName)).Append('"');
        foreach (var tag in tags)
        {
            string key = renameFinish && tag.Key == RoostTelemetry.FinishReasonTag ? "finished_reason" : tag.Key;
            labels.Append(',').Append(key).Append("=\"").Append(EscapeLabel(tag.Value)).Append('"');
        }

        return labels.ToString();
    }

    private static void Sample(StringBuilder text, string name, string labels, string? le, double value)
    {
        text.Append(name).Append('{').Append(labels);
        if (le is not null)
        {
            text.Append(",le=\"").Append(le).Append('"');
        }

        text.Append("} ").Append(Number(value)).Append('\n');
    }

    private static string Number(double value) => value switch
    {
        double.PositiveInfinity => "+Inf",
        double.NegativeInfinity => "-Inf",
        _ when double.IsNaN(value) => "NaN",
        _ => value.ToString("R", CultureInfo.InvariantCulture),
    };

    private static string EscapeHelp(string help) => help.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\n", "\\n", StringComparison.Ordinal);

    private static string EscapeLabel(string value) =>
        value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal).Replace("\n", "\\n", StringComparison.Ordinal);

    private enum Kind
    {
        Counter,
        Gauge,
        Histogram,
    }

    private sealed record Family(string Instrument, string Name, Kind Kind, string Help, string[] Aliases, bool SeedFinishReasons = false);

    private sealed class Series(KeyValuePair<string, string>[] tags)
    {
        public KeyValuePair<string, string>[] Tags { get; } = tags;

        public double Value { get; set; }

        public long[]? Buckets { get; set; }

        public double Sum { get; set; }

        public long Count { get; set; }
    }

    private sealed class State(Family family)
    {
        public Family Family { get; } = family;

        public double[] Bounds { get; set; } = [];

        public SortedDictionary<string, Series> Series { get; } = new(StringComparer.Ordinal);

        public Series Get(KeyValuePair<string, string>[] tags)
        {
            var key = string.Join('\u0001', tags.Select(t => t.Key + "=" + t.Value));
            if (!Series.TryGetValue(key, out var series))
            {
                series = new Series(tags);
                Series.Add(key, series);
            }

            return series;
        }
    }
}

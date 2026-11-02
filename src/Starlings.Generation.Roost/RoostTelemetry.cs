using System.Diagnostics;
using System.Diagnostics.Metrics;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost;

/// <summary>
/// The metrics and traces of a <see cref="RoostChatClient"/> (R-038), on <c>System.Diagnostics.Metrics</c> and
/// <c>System.Diagnostics.Activity</c>, with no exporter dependency: any <see cref="MeterListener"/> or OpenTelemetry SDK can read
/// them, and Roost's server exports them on <c>GET /metrics</c> in Prometheus text format. Each client owns one
/// <see cref="Meter"/> named <see cref="MeterName"/> (so two clients in one process stay apart); the per-request activities come
/// from the shared <see cref="ActivitySourceName"/> source. No instrument, tag or activity carries prompt or output text.
/// </summary>
/// <remarks>
/// <para>A request is counted from the moment it is submitted to the backend: a request refused while it is read (an unsupported
/// option, an empty or too-long prompt) records nothing. It is <b>waiting</b> until its first token arrives (queued or in
/// prefill) and <b>running</b> from then until it ends. Its finish reason is <c>stop</c>, <c>length</c>, <c>abort</c> (cancelled
/// or abandoned by its caller) or <c>error</c> (failed with a <see cref="RoostChatException"/>).</para>
/// <para>Times are measured in the host from the start of tokenisation: time to first token to the first token event, inter-token
/// latency between token events (an event of <c>n</c> tokens after the first records <c>n</c> intervals of a <c>1/n</c> share; tokens
/// that arrive with the first one record none), and end-to-end
/// latency to the end of the request.</para>
/// </remarks>
public sealed class RoostTelemetry : IDisposable
{
    /// <summary>The name of every client's <see cref="Meter"/>.</summary>
    public const string MeterName = "Starlings.Roost";

    /// <summary>The name of the <see cref="ActivitySource"/> of the per-request activities.</summary>
    public const string ActivitySourceName = "Starlings.Roost";

    /// <summary>The operation name of the one activity each request records.</summary>
    public const string RequestActivityName = "roost.request";

    /// <summary>Counter of finished requests, tagged <see cref="FinishReasonTag"/>.</summary>
    public const string RequestsInstrument = "roost.requests";

    /// <summary>Counter of prompt tokens of finished requests.</summary>
    public const string PromptTokensInstrument = "roost.prompt_tokens";

    /// <summary>Counter of generated tokens of finished requests (a stop token counts, as in usage).</summary>
    public const string GenerationTokensInstrument = "roost.generation_tokens";

    /// <summary>Histogram of the prompt tokens per finished request.</summary>
    public const string RequestPromptTokensInstrument = "roost.request.prompt_tokens";

    /// <summary>Histogram of the generated tokens per finished request.</summary>
    public const string RequestGenerationTokensInstrument = "roost.request.generation_tokens";

    /// <summary>Histogram of the time to first token, in seconds.</summary>
    public const string TimeToFirstTokenInstrument = "roost.time_to_first_token";

    /// <summary>Histogram of the inter-token latency, in seconds.</summary>
    public const string InterTokenLatencyInstrument = "roost.inter_token_latency";

    /// <summary>Histogram of the end-to-end request latency, in seconds.</summary>
    public const string E2eRequestLatencyInstrument = "roost.e2e_request_latency";

    /// <summary>Gauge of the requests generating (first token received, not finished).</summary>
    public const string RunningRequestsInstrument = "roost.requests.running";

    /// <summary>Gauge of the requests submitted and waiting for their first token.</summary>
    public const string WaitingRequestsInstrument = "roost.requests.waiting";

    /// <summary>Gauge of the share of the KV cache's attention blocks in use, 0 to 1, as of the last planned step.</summary>
    public const string KvCacheUsageInstrument = "roost.kv_cache.usage";

    /// <summary>The finish reason tag of <see cref="RequestsInstrument"/> (<c>stop</c>, <c>length</c>, <c>abort</c>, <c>error</c>).</summary>
    public const string FinishReasonTag = "finish_reason";

    /// <summary>The activity tag of the request id (the response id the routes return, <c>roost-&lt;n&gt;</c>).</summary>
    public const string RequestIdTag = "roost.request_id";

    /// <summary>The finish reasons a request ends with.</summary>
    public static IReadOnlyList<string> FinishReasons { get; } = ["stop", "length", "abort", "error"];

    // Bucket boundaries, in seconds, chosen to span the ranges of vLLM's histograms of the same names.
    private static readonly double[] s_ttftBuckets = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20, 40, 80, 160, 640, 2560];
    private static readonly double[] s_itlBuckets = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20, 40, 80];
    private static readonly double[] s_e2eBuckets = [0.3, 0.5, 0.8, 1, 1.5, 2, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 120, 240, 480, 960, 1920, 7680];
    private static readonly long[] s_tokenBuckets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000];

    private static readonly ActivitySource s_source = new(ActivitySourceName);

    private readonly string _modelId;
    private readonly Counter<long> _requests;
    private readonly Counter<long> _promptTokens;
    private readonly Counter<long> _generationTokens;
    private readonly Histogram<long> _requestPromptTokens;
    private readonly Histogram<long> _requestGenerationTokens;
    private readonly Histogram<double> _timeToFirstToken;
    private readonly Histogram<double> _interTokenLatency;
    private readonly Histogram<double> _e2eRequestLatency;
    private int _running;
    private int _waiting;

    internal RoostTelemetry(string modelId, Func<double?> kvCacheUsage)
    {
        _modelId = modelId;
        Meter = new Meter(MeterName);
        _requests = Meter.CreateCounter<long>(RequestsInstrument, "{request}", "Finished requests by finish reason.");
        _promptTokens = Meter.CreateCounter<long>(PromptTokensInstrument, "{token}", "Prompt tokens of finished requests.");
        _generationTokens = Meter.CreateCounter<long>(GenerationTokensInstrument, "{token}", "Generated tokens of finished requests.");
        _requestPromptTokens = Meter.CreateHistogram(RequestPromptTokensInstrument, "{token}", "Prompt tokens per finished request.", null, new InstrumentAdvice<long> { HistogramBucketBoundaries = s_tokenBuckets });
        _requestGenerationTokens = Meter.CreateHistogram(RequestGenerationTokensInstrument, "{token}", "Generated tokens per finished request.", null, new InstrumentAdvice<long> { HistogramBucketBoundaries = s_tokenBuckets });
        _timeToFirstToken = Meter.CreateHistogram(TimeToFirstTokenInstrument, "s", "Time to first token.", null, new InstrumentAdvice<double> { HistogramBucketBoundaries = s_ttftBuckets });
        _interTokenLatency = Meter.CreateHistogram(InterTokenLatencyInstrument, "s", "Inter-token latency.", null, new InstrumentAdvice<double> { HistogramBucketBoundaries = s_itlBuckets });
        _e2eRequestLatency = Meter.CreateHistogram(E2eRequestLatencyInstrument, "s", "End-to-end request latency.", null, new InstrumentAdvice<double> { HistogramBucketBoundaries = s_e2eBuckets });
        _ = Meter.CreateObservableGauge(RunningRequestsInstrument, () => Volatile.Read(ref _running), "{request}", "Requests generating.");
        _ = Meter.CreateObservableGauge(WaitingRequestsInstrument, () => Volatile.Read(ref _waiting), "{request}", "Requests waiting for their first token.");
        _ = Meter.CreateObservableGauge(KvCacheUsageInstrument, () => kvCacheUsage() is { } usage ? [new Measurement<double>(usage)] : Array.Empty<Measurement<double>>(), "1", "Share of the KV cache's attention blocks in use.");
    }

    /// <summary>This client's meter; it is disposed with the client.</summary>
    public Meter Meter { get; }

    /// <summary>The requests generating now.</summary>
    public int RunningRequests => Volatile.Read(ref _running);

    /// <summary>The requests waiting for their first token now.</summary>
    public int WaitingRequests => Volatile.Read(ref _waiting);

    /// <summary>Disposes the meter; listeners stop receiving its measurements.</summary>
    public void Dispose() => Meter.Dispose();

    /// <summary>Starts the telemetry of a request that is being submitted: it is waiting from now, and its activity starts.</summary>
    internal RequestTelemetry Start(string requestId, long startTimestamp, int promptTokens)
    {
        _ = Interlocked.Increment(ref _waiting);
        var activity = s_source.StartActivity(RequestActivityName, ActivityKind.Internal);
        if (activity is not null)
        {
            _ = activity.SetTag(RequestIdTag, requestId);
            _ = activity.SetTag("gen_ai.request.model", _modelId);
            _ = activity.SetTag("gen_ai.usage.input_tokens", promptTokens);
        }

        return new RequestTelemetry(this, activity, startTimestamp);
    }

    /// <summary>The telemetry of one request; <see cref="End"/> records it once.</summary>
    internal sealed class RequestTelemetry
    {
        private readonly RoostTelemetry _owner;
        private readonly Activity? _activity;
        private readonly long _start;
        private long _last;
        private bool _running;
        private bool _ended;

        public RequestTelemetry(RoostTelemetry owner, Activity? activity, long start)
        {
            _owner = owner;
            _activity = activity;
            _start = start;
        }

        /// <summary>Records the arrival of <paramref name="count"/> tokens.</summary>
        public void Tokens(int count)
        {
            if (count <= 0 || _ended)
            {
                return;
            }

            long now = Stopwatch.GetTimestamp();
            if (!_running)
            {
                _running = true;
                _ = Interlocked.Decrement(ref _owner._waiting);
                _ = Interlocked.Increment(ref _owner._running);
                _owner._timeToFirstToken.Record(Stopwatch.GetElapsedTime(_start, now).TotalSeconds);
            }
            else
            {
                // The interval since the previous token event is shared by the tokens of this one.
                double share = Stopwatch.GetElapsedTime(_last, now).TotalSeconds / count;
                for (var i = 0; i < count; i++)
                {
                    _owner._interTokenLatency.Record(share);
                }
            }

            _last = now;
        }

        /// <summary>Records the end of the request: its finish reason, token counts and latency; ends its activity.</summary>
        public void End(string finishReason, int promptTokens, int completionTokens, RoostError? error)
        {
            if (_ended)
            {
                return;
            }

            _ended = true;
            if (_running)
            {
                _ = Interlocked.Decrement(ref _owner._running);
            }
            else
            {
                _ = Interlocked.Decrement(ref _owner._waiting);
            }

            var reason = new KeyValuePair<string, object?>(FinishReasonTag, finishReason);
            _owner._requests.Add(1, reason);
            _owner._promptTokens.Add(promptTokens);
            _owner._generationTokens.Add(completionTokens);
            _owner._requestPromptTokens.Record(promptTokens);
            _owner._requestGenerationTokens.Record(completionTokens);
            _owner._e2eRequestLatency.Record(Stopwatch.GetElapsedTime(_start).TotalSeconds);

            if (_activity is not null)
            {
                _ = _activity.SetTag("gen_ai.usage.input_tokens", promptTokens);
                _ = _activity.SetTag("gen_ai.usage.output_tokens", completionTokens);
                _ = _activity.SetTag("gen_ai.response.finish_reasons", finishReason);
                if (error is not null)
                {
                    // The C6 code name only: a message may quote request fields.
                    _ = _activity.SetTag("error.type", error.CodeName);
                    _ = _activity.SetStatus(ActivityStatusCode.Error, error.CodeName);
                }

                _activity.Dispose();
            }
        }
    }
}

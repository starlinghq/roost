"""Roost's serving-load harness for the section 14.3 scorecard (R-047 groundwork, stream I).

Drives any OpenAI-compatible endpoint and computes TTFT, ITL, throughput, goodput under SLO and saturation.
Standard library only, so it runs on a client host or rented pod with nothing but Python 3.11+.
"""

__version__ = "0.1.0"

# Interim per-request trace format. Not the Roost raw-results schema, which must follow Starling's
# starlings-bench-raw/1 and is still open under R-047.
TRACE_FORMAT = "roost-bench-trace/0"

"""Prometheus metrics for the JetPilotGuard serving path.

Teaching note
-------------
Prometheus works by *scraping*: our service exposes a text page at /metrics,
and Prometheus fetches it every few seconds and stores the numbers as
time-series. We just have to declare the metrics here and update them whenever
we process a packet.

Three metric *types* matter, and picking the right one is the skill:

* Counter  -- only goes up; for counting events (overrides fired, packets seen).
             You look at its *rate* over time ("overrides per minute").
* Gauge    -- goes up and down; for a current value (latest impairment
             probability, current time-to-impact).
* Histogram-- records a distribution; for things like latency, where you care
             about p50/p95/p99, not just the average.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Counters: cumulative event tallies -----------------------------------
PACKETS_TOTAL = Counter(
    "jetpilotguard_packets_total",
    "Total telemetry packets processed, labelled by resulting status.",
    ["status"],
)
OVERRIDES_TOTAL = Counter(
    "jetpilotguard_overrides_total",
    "Total Auto-GCAS overrides commanded by the deterministic watchdog.",
)
ADVISORIES_TOTAL = Counter(
    "jetpilotguard_advisories_total",
    "Total ML advisories raised (impairment probability above threshold).",
)

# --- Gauges: latest instantaneous values ----------------------------------
IMPAIRMENT_PROBABILITY = Gauge(
    "jetpilotguard_impairment_probability",
    "Most recent ML impairment probability (0-1).",
)
TIME_TO_IMPACT = Gauge(
    "jetpilotguard_time_to_impact_seconds",
    "Most recent time-to-impact estimate (seconds); large value means safe.",
)

# --- Histogram: latency distribution --------------------------------------
PROCESS_LATENCY = Histogram(
    "jetpilotguard_process_latency_seconds",
    "Per-packet engine processing latency (seconds).",
    # Buckets tuned for a sub-10ms real-time path.
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)


def record_assessment(status: str, impairment: float, tti_seconds: float) -> None:
    """Update all metrics for one processed packet.

    Called by the service after each engine.process() call. ``tti_seconds`` may
    be infinite (no descent); we clamp it to a large finite number so the gauge
    stays plottable.
    """
    PACKETS_TOTAL.labels(status=status).inc()
    IMPAIRMENT_PROBABILITY.set(impairment)
    TIME_TO_IMPACT.set(min(tti_seconds, 9999.0))
    if status == "CRITICAL_OVERRIDE":
        OVERRIDES_TOTAL.inc()
    elif status == "ADVISORY":
        ADVISORIES_TOTAL.inc()

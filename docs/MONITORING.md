# Monitoring & Observability

JetPilotGuard ships with a production-style monitoring stack: the service exposes
Prometheus metrics, Prometheus scrapes and stores them, and Grafana visualises
them live. This document explains what's measured and why.

## The stack

```
  load_generator ──POST /assess──> JetPilotGuard service ──exposes /metrics──┐
                                                                          │ scrape (5s)
                                            ┌──────── Prometheus <────────┘
                                            │ query
                                            ▼
                                         Grafana  (dashboards)
```

Run it with `docker compose up --build`, then `python -m scripts.load_generator`
to create traffic. See `docs/GETTING_STARTED.md` section 9.

## What's measured, and why each metric type was chosen

| Metric | Type | Why this type |
|--------|------|---------------|
| `jetpilotguard_packets_total{status}` | Counter | Counting events; you graph its *rate* (packets/sec) and split by status. |
| `jetpilotguard_overrides_total` | Counter | Safety-critical event count; alert on any increase. |
| `jetpilotguard_advisories_total` | Counter | Advisory volume over time. |
| `jetpilotguard_impairment_probability` | Gauge | A current value that moves up and down. |
| `jetpilotguard_time_to_impact_seconds` | Gauge | Current instantaneous estimate. |
| `jetpilotguard_process_latency_seconds` | Histogram | Distributions matter here — you want p95/p99, not just the mean. |

Choosing the right metric type is the actual skill. Counters answer "how often";
gauges answer "what is it now"; histograms answer "what's the spread".

## The dashboard panels

The provisioned dashboard (**JetPilotGuard Live Monitoring**) has three rows:

1. **Current state** — live impairment probability, time-to-impact, and running
   totals of overrides and advisories.
2. **Throughput & rates** — packets/sec broken down by status, and override
   rate per minute. This is where you'd *see* a spike in dangerous conditions.
3. **Latency percentiles** — p50/p95/p99 processing latency, computed from the
   histogram with `histogram_quantile`. This substantiates the "real-time"
   claim with live evidence, not a one-off benchmark.

## How this connects to ML monitoring

Watching `jetpilotguard_impairment_probability` over time is the seed of **model
drift detection**: if the distribution of predictions shifts unexpectedly, that
is an early signal something changed (in inputs or in the world) and the model
may need retraining. This project demonstrates the *instrumentation* that makes
such monitoring possible.

## Honest scope

Prometheus/Grafana monitor the **operational** behaviour of the running service.
They do **not** evaluate model quality on labelled data — that is what
`scripts/evaluate_system.py` and the held-out metrics are for. The two are
complementary: offline evaluation tells you if the model is good; online
monitoring tells you if the running system is healthy and behaving as expected.

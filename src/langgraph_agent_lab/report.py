"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report_stub(metrics: MetricsReport) -> str:
    """Render a submission-ready lab report from collected metrics."""
    scenario_rows = "\n".join(
        f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} | "
        f"{'yes' if item.success else 'no'} | {item.retry_count} | {item.interrupt_count} |"
        for item in metrics.scenario_metrics
    )
    return f"""# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date:

## 2. Architecture

The workflow uses a typed LangGraph state with small node functions and explicit routing edges.
The graph starts at `intake`, classifies each ticket into `simple`, `tool`,
`missing_info`, `risky`, or `error`,
then terminates through `finalize`.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| route | overwrite | only the current route matters |
| risk_level | overwrite | current risk assessment only |
| attempt | overwrite | retry counter for bounded loops |
| evaluation_result | overwrite | latest gate for retry vs answer |
| final_answer | overwrite | final user-facing output |
| pending_question | overwrite | one clarification prompt at a time |
| approval | overwrite | latest approval decision |
| messages | append | lightweight execution trace |
| tool_results | append | preserve tool history across retries |
| errors | append | preserve retry/dead-letter history |
| events | append | audit trail for metrics and debugging |

## 4. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
{scenario_rows}

## 5. Failure analysis

1. Retry or tool failure:
   Transient tool failures are detected in `evaluate_node`.
   When the latest tool result contains an error marker, the graph loops
   through `retry` and increments `attempt`. Once `attempt >= max_attempts`, the route moves to
   `dead_letter` instead of looping forever.

2. Risky action without approval:
   Risky requests are routed to `risky_action` and then `approval`.
   If approval is not granted, the graph does not
   execute the tool path and instead falls back to clarification, preventing unsafe side effects.

## 6. Persistence / recovery evidence

The graph accepts a checkpointer and passes `thread_id` through the invoke config for each scenario.
Memory persistence works by default for local runs, and SQLite is supported through
`build_checkpointer("sqlite")`
with WAL mode enabled so checkpoint history can be kept across process restarts.

## 7. Extension work

Implemented SQLite checkpoint support in addition to the default in-memory saver.
This keeps the lab compatible with the base workflow while adding a persistence path
for recovery demos.

## 8. Improvement plan

If there were one more day, the next step would be replacing heuristic
classification/evaluation with structured LLM
judges, plus a real human approval UI on top of LangGraph interrupts.

## Metrics summary

- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.2%}
- Average nodes visited: {metrics.avg_nodes_visited:.2f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_stub(metrics), encoding="utf-8")

# Day 08 Lab Report

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
| G01_simple | simple | simple | yes | 0 | 0 |
| G02_simple2 | simple | simple | yes | 0 | 0 |
| G03_tool | tool | tool | yes | 0 | 0 |
| G04_tool2 | tool | tool | yes | 0 | 0 |
| G05_tool3 | tool | tool | yes | 0 | 0 |
| G06_missing | missing_info | missing_info | yes | 0 | 0 |
| G07_missing2 | missing_info | missing_info | yes | 0 | 0 |
| G08_risky | risky | risky | yes | 0 | 1 |
| G09_risky2 | risky | risky | yes | 0 | 1 |
| G10_risky3 | risky | risky | yes | 0 | 1 |
| G11_risky4 | risky | risky | yes | 0 | 1 |
| G12_error | error | error | yes | 3 | 0 |
| G13_error2 | error | error | yes | 3 | 0 |
| G14_dead | error | error | yes | 1 | 0 |
| G15_mixed | risky | risky | yes | 0 | 1 |

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

- Total scenarios: 15
- Success rate: 100.00%
- Average nodes visited: 6.73
- Total retries: 7
- Total interrupts: 5

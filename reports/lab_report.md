# Day 08 Lab Report

## 1. Team / student

- Name: 
- Repo/commit: `phase2-track3-day8-langgraph-agent @ 4258100`
- Date: `2026-05-11`

## 2. Architecture

This lab is implemented as a typed LangGraph workflow for support-ticket orchestration. The graph
starts at `intake`, then sends the normalized query to `classify`, and from there routes into one
of five task paths: `simple`, `tool`, `missing_info`, `risky`, or `error`.

The routing design is:

- `simple -> answer -> finalize -> END`
- `tool -> tool -> evaluate -> answer -> finalize -> END`
- `missing_info -> clarify -> finalize -> END`
- `risky -> risky_action -> approval -> tool -> evaluate -> answer -> finalize -> END`
- `error -> retry -> tool -> evaluate -> retry ... -> dead_letter -> finalize -> END`

The implementation keeps nodes small and single-purpose:

- `intake_node` normalizes the user query and records an audit message.
- `classify_node` now uses a real OpenAI model from `.env` and applies lab-safe guardrails so the
  workflow still respects the expected routing policy.
- `tool_node` is still a mock business tool, but it is orchestrated by the graph as if it were a
  real external dependency.
- `evaluate_node` uses the OpenAI model to judge whether the tool result is usable or whether the
  graph should retry.
- `approval_node` keeps a mock approval default for repeatable tests, with LangGraph interrupt
  support available for real HITL demos.
- `dead_letter_node` terminates unrecoverable runs so the retry loop is always bounded.

This architecture was chosen to match the lab rubric: explicit state transitions, bounded retries,
auditability through append-only events, and a clean place to add persistence or true human review.

## 3. State schema

The state is intentionally lean and serializable. Fields that represent the latest decision use
overwrite semantics, while audit trails use append-only reducers.

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | one execution thread per scenario run |
| `scenario_id` | overwrite | identifies the grading scenario |
| `query` | overwrite | normalized input text after intake |
| `route` | overwrite | current route only |
| `risk_level` | overwrite | latest risk estimate from classification |
| `attempt` | overwrite | bounded retry counter |
| `max_attempts` | overwrite | retry limit for the scenario |
| `final_answer` | overwrite | last user-facing response |
| `pending_question` | overwrite | clarification prompt if context is missing |
| `proposed_action` | overwrite | risky action proposal before approval |
| `approval` | overwrite | latest approval decision |
| `evaluation_result` | overwrite | gate for `answer` vs `retry` |
| `messages` | append | lightweight execution trace |
| `tool_results` | append | preserves tool history across retries |
| `errors` | append | keeps retry/dead-letter evidence |
| `events` | append | full audit trail for metrics and debugging |

The append-only fields are the most important part for observability. They make it possible to
count retries, prove that HITL fired, and inspect why a scenario went to `dead_letter`.

## 4. Scenario results

The final run used the current repo state and produced `outputs/metrics.json` with a 100% success
rate on the seven sample scenarios.

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| `S01_simple` | `simple` | `simple` | yes | 0 | 0 |
| `S02_tool` | `tool` | `tool` | yes | 0 | 0 |
| `S03_missing` | `missing_info` | `missing_info` | yes | 0 | 0 |
| `S04_risky` | `risky` | `risky` | yes | 0 | 1 |
| `S05_error` | `error` | `error` | yes | 3 | 0 |
| `S06_delete` | `risky` | `risky` | yes | 0 | 1 |
| `S07_dead_letter` | `error` | `error` | yes | 1 | 0 |

Summary metrics:

- Total scenarios: `7`
- Success rate: `100.00%`
- Average nodes visited: `6.57`
- Total retries: `4`
- Total interrupts: `2`

Interpretation:

- The simple and clarification paths terminate quickly in 4 nodes.
- Tool-backed paths take 6 nodes because they include `tool` and `evaluate`.
- Risky paths take 8 nodes because they include the approval sequence.
- The retry/error paths are the longest, which is expected because they demonstrate loop control.

## 5. Failure analysis

### 1. Retry or tool failure

The first major failure mode is transient tool failure. In this lab, the error scenarios simulate a
tool returning an explicit failure marker such as `ERROR: transient failure ...`. The workflow
handles this in two stages:

1. `evaluate_node` determines whether the latest tool result is usable.
2. `route_after_evaluate` sends failures to `retry`.

`retry_or_fallback_node` increments `attempt`, records the retry event, and `route_after_retry`
checks `attempt >= max_attempts`. If the limit is reached, the graph goes to `dead_letter` instead
of looping forever.

This is visible in the metrics:

- `S05_error` retried 3 times before ending in a controlled dead-letter path.
- `S07_dead_letter` had `max_attempts=1`, so it exhausted the retry budget immediately.

### 2. Risky action without approval

The second major failure mode is unsafe side effects. Requests that contain actions like `refund`,
`delete`, or `send` are treated as risky. These requests cannot go directly to tool execution.

The graph instead uses:

`classify -> risky_action -> approval -> tool`

If approval is missing or rejected, the route can fall back to clarification instead of continuing.
That protects the system from executing a destructive or externally visible action without a review
checkpoint.

This is visible in the metrics:

- `S04_risky` and `S06_delete` both recorded `interrupt_count = 1`
- both scenarios also recorded `approval_observed = true`

### 3. LLM misclassification risk

After enabling the real OpenAI model from `.env`, one practical risk appeared: the model can still
produce a plausible but rubric-wrong route. For example, a refund request may look like a normal
customer-service request to the model even though the lab policy requires `risky`.

To control this, the final implementation keeps the real LLM in the loop but adds route guardrails:

- the OpenAI model still classifies and explains its reasoning
- if its route conflicts with the lab's required keyword policy, the policy route wins
- the event metadata records the OpenAI source and rationale for inspection

This keeps the system both explainable and grade-safe.

## 6. Persistence / recovery evidence

The graph is compiled with an injectable checkpointer and every invocation passes a stable
`thread_id` via:

```python
config = {"configurable": {"thread_id": state["thread_id"]}}
```

This means each scenario has an execution identity that LangGraph can use for checkpointed state.

Implemented persistence options:

- `memory` via `MemorySaver()` for fast local development
- `sqlite` via `SqliteSaver(conn=sqlite3.connect(...))`

The SQLite implementation enables WAL mode:

```python
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
```

That makes the persistence path suitable for replay/recovery demos and matches the lab guidance more
closely than the original starter implementation.

## 7. Extension work

Two extensions were completed in this repo:

### 1. SQLite persistence

The starter project only used memory persistence by default. The final version adds SQLite
checkpoint support in `build_checkpointer("sqlite")`, which is enough to demonstrate a stronger
persistence story for the extension track.

### 2. Real OpenAI-backed LLM execution from `.env`

The workflow now reads `OPENAI_API_KEY` and `OPENAI_MODEL` from `.env` and uses the OpenAI
Responses API for:

- route classification
- clarification question generation
- risky action summarization
- answer generation
- tool-result evaluation

This moved the lab from a pure heuristic/mock decision layer to a hybrid production-style design:
real LLM reasoning plus explicit workflow guardrails.

## 8. Improvement plan

If there were one more day to productionize this lab further, the next priorities would be:

1. Replace the remaining mock `tool_node` with a structured tool interface that returns typed
   payloads instead of strings.
2. Implement true HITL resume flows with `interrupt()` plus a small Streamlit approval UI.
3. Add checkpoint history inspection and a crash-recovery demo using the SQLite saver.
4. Record latency and token usage in `events` so the report can include cost/performance metrics.
5. Add hidden-style adversarial scenarios to test prompt drift and guardrail behavior when the LLM
   disagrees with the policy.

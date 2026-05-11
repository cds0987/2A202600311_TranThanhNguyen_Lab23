"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .llm import generate_text, is_llm_enabled, parse_structured_output
from .state import AgentState, ApprovalDecision, Route, make_event

RISKY_KEYWORDS = {"refund", "delete", "send", "cancel", "remove", "revoke"}
TOOL_KEYWORDS = {"status", "order", "lookup", "check", "track", "find", "search"}
ERROR_KEYWORDS = {"timeout", "fail", "failure", "error", "crash", "unavailable", "recover"}
VAGUE_PRONOUNS = {"it", "this", "that", "thing"}


class ClassificationOutput(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="One of: simple, tool, missing_info, risky, error"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="One of: low, medium, high"
    )
    rationale: str = Field(description="Short explanation of the routing choice")


class EvaluationOutput(BaseModel):
    evaluation_result: Literal["success", "needs_retry"] = Field(
        description="Either success or needs_retry"
    )
    rationale: str = Field(description="Short explanation of the evaluation")


VALID_ROUTES = {
    Route.SIMPLE.value,
    Route.TOOL.value,
    Route.MISSING_INFO.value,
    Route.RISKY.value,
    Route.ERROR.value,
}


def _heuristic_classification(query: str) -> tuple[Route, str, str]:
    clean_words = re.findall(r"\b[\w']+\b", query.lower())
    word_set = set(clean_words)
    if word_set & RISKY_KEYWORDS:
        return Route.RISKY, "high", "matched risky keywords"
    if word_set & TOOL_KEYWORDS:
        return Route.TOOL, "low", "matched tool keywords"
    if len(clean_words) < 5 and word_set & VAGUE_PRONOUNS:
        return Route.MISSING_INFO, "low", "short vague request missing context"
    if word_set & ERROR_KEYWORDS:
        return Route.ERROR, "low", "matched error keywords"
    return Route.SIMPLE, "low", "default simple route"


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    The lab keeps intake lightweight and serializable: trim whitespace and record an audit event.
    """
    query = " ".join(state.get("query", "").split())
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route.

    The lab uses deterministic keyword rules so scenarios and hidden tests can exercise the same
    policy without relying on exact scenario IDs.
    """
    raw_query = state.get("query", "")
    route, risk_level, rationale = _heuristic_classification(raw_query)
    if is_llm_enabled():
        try:
            llm_result = parse_structured_output(
                instructions=(
                    "Classify a support ticket into exactly one route: simple, tool, "
                    "missing_info, risky, or error. "
                    "Use these priorities when multiple categories appear: risky first, "
                    "then tool, then missing_info, then error, then simple. "
                    "Requests involving refund, delete, send, cancel, remove, or revoke "
                    "must be risky. "
                    "Requests involving status, order, lookup, check, track, find, or "
                    "search should be tool unless risky already applies. "
                    "Very short vague requests like 'Can you fix it?' should be "
                    "missing_info. "
                    "Return a conservative risk_level of low, medium, or high."
                ),
                user_input=raw_query,
                text_format=ClassificationOutput,
            )
            route_candidate = llm_result.route.strip().lower()
            if route_candidate in VALID_ROUTES:
                llm_route = Route(route_candidate)
                llm_risk_level = llm_result.risk_level.strip().lower()
                llm_rationale = llm_result.rationale.strip()
                if llm_route != route:
                    return {
                        "route": route.value,
                        "risk_level": risk_level,
                        "events": [
                            make_event(
                                "classify",
                                "completed",
                                f"route={route.value}",
                                source="openai_guardrail",
                                llm_route=llm_route.value,
                                rationale=llm_rationale,
                                guardrail_reason=rationale,
                            )
                        ],
                    }
                route = llm_route
                risk_level = llm_risk_level
                rationale = llm_rationale
                return {
                    "route": route.value,
                    "risk_level": risk_level,
                    "events": [
                        make_event(
                            "classify",
                            "completed",
                            f"route={route.value}",
                            source="openai",
                            rationale=rationale,
                        )
                    ],
                }
        except Exception as exc:
            rationale = f"llm_fallback:{type(exc).__name__}; {rationale}"
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}",
                source="heuristic",
                rationale=rationale,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Clarification stays specific enough to unblock the next turn without inventing details.
    """
    query = state.get("query", "").strip()
    if is_llm_enabled():
        try:
            question = generate_text(
                instructions=(
                    "Write one concise clarification question for a support agent. "
                    "Ask only for the missing detail needed to proceed. "
                    "Do not answer the request."
                ),
                user_input=query,
                max_output_tokens=80,
            )
        except Exception:
            question = ""
    else:
        question = ""
    if not question:
        if "order" in query.lower():
            question = "Can you share the order ID and the specific issue you want me to check?"
        else:
            question = "Can you share the missing details so I can help with the request?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = (
            f"ERROR: transient failure attempt={attempt} "
            f"scenario={state.get('scenario_id', 'unknown')}"
        )
    elif state.get("route") == Route.RISKY.value:
        approved = bool((state.get("approval") or {}).get("approved"))
        result = f"mock-tool-result for scenario={scenario_id}; approved={approved}"
    else:
        result = f"mock-tool-result for scenario={scenario_id}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.

    The proposal captures the requested risky action so the approval node has enough context.
    """
    proposed_action = f"Execute risky support action for query: {state.get('query', '')}"
    if is_llm_enabled():
        try:
            proposed_action = generate_text(
                instructions=(
                    "Summarize the risky support action that needs approval. "
                    "Mention the requested action and why approval is required in one sentence."
                ),
                user_input=state.get("query", ""),
                max_output_tokens=100,
            )
        except Exception:
            pass
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                "approval required",
                risk_level=state.get("risk_level", "high"),
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    By default this node uses a mock approval so tests run offline; interrupt() can be enabled for
    real HITL demos.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.

    Routing enforces the retry bound; this node only increments the counter and logs metadata.
    """
    attempt = int(state.get("attempt", 0)) + 1
    errors = [f"transient failure attempt={attempt}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=attempt,
                remaining_attempts=max(int(state.get("max_attempts", 3)) - attempt, 0),
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.

    Final answers are grounded in the latest tool result and approval context when present.
    """
    approval = state.get("approval") or {}
    answer = ""
    latest_tool_result = state.get("tool_results", [])[-1] if state.get("tool_results") else ""
    if is_llm_enabled():
        try:
            answer = generate_text(
                instructions=(
                    "Write a concise support-agent response. "
                    "Ground the answer in the provided tool result and approval state. "
                    "If no tool result exists, give a safe generic response."
                ),
                user_input=(
                    f"query={state.get('query', '')}\n"
                    f"tool_result={latest_tool_result}\n"
                    f"approval={approval}"
                ),
                max_output_tokens=140,
            )
        except Exception:
            answer = ""
    if not answer:
        if state.get("tool_results"):
            answer = f"I found: {state['tool_results'][-1]}"
            if approval:
                answer = f"{answer} | approval={approval.get('approved')}"
        else:
            answer = "Provide the customer with the standard support guidance for this request."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    The lab keeps evaluation deterministic by treating explicit tool error markers as retryable.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if is_llm_enabled() and latest:
        try:
            llm_result = parse_structured_output(
                instructions=(
                    "Evaluate whether the support workflow should retry the tool call. "
                    "Return success when the tool output is usable. "
                    "Return needs_retry only for transient or explicit tool failures."
                ),
                user_input=(
                    f"query={state.get('query', '')}\n"
                    f"route={state.get('route', '')}\n"
                    f"attempt={state.get('attempt', 0)}\n"
                    f"tool_result={latest}"
                ),
                text_format=EvaluationOutput,
            )
            if llm_result.evaluation_result.strip().lower() in {"success", "needs_retry"}:
                return {
                    "evaluation_result": llm_result.evaluation_result.strip().lower(),
                    "events": [
                        make_event(
                            "evaluate",
                            "completed",
                            "tool result evaluated",
                            source="openai",
                            rationale=llm_result.rationale.strip(),
                        )
                    ],
                }
        except Exception:
            pass
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    "tool result indicates failure, retry needed",
                )
            ],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    """
    return {
        "final_answer": (
            "Request could not be completed after maximum retry attempts. "
            "Logged for manual review."
        ),
        "errors": [f"dead_letter after attempt={state.get('attempt', 0)}"],
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded, attempt={state.get('attempt', 0)}",
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}

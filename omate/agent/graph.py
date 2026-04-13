"""
Omate LangGraph Agent Graph.

Supervisor + specialist agent pattern with human-in-the-loop.

Nodes:
  signal_analyst  — load FHIR context, check medications, compute risk score
  rag_reporter    — generate clinical report with hallucination guards
  escalation      — alert on-call physician for high-risk cases
  hitl_review     — human-in-the-loop checkpoint (interrupt here)

The graph uses interrupt_before=["hitl_review"] so execution pauses
for physician input. After approval/rejection, resume() continues.

Note: This implementation works WITHOUT LangGraph installed by using
a simple state machine fallback. Install langgraph for full functionality.
"""

from dataclasses import dataclass, field
from typing import Any

from .state import OmateState
from .tools import OmateTools
from ..rag.engine import ClinicalRAGEngine
from ..fhir.store import FHIRStore


ESCALATION_THRESHOLD = float(__import__("os").getenv("ESCALATION_THRESHOLD", "0.90"))


# ---------------------------------------------------------------------------
# Node implementations (pure functions on OmateState)
# ---------------------------------------------------------------------------

def node_signal_analyst(state: OmateState, tools: OmateTools) -> OmateState:
    """
    Load FHIR context, check QT medications, compute composite risk score.
    This is the agent that 'reasons across time' — something a pure pipeline cannot do.
    """
    patient_id = state["patient_id"]

    # Load FHIR context
    fhir_context = tools.store.get_patient_context(patient_id)

    # Check QT-prolonging medications
    qt_meds = tools.check_qt_prolonging_medications(patient_id)

    # Count recurrence in history
    recurrence = tools.count_anomaly_events(patient_id)

    # Compute composite risk
    risk_score = tools.compute_risk_score(
        signal_events=state.get("signal_events", []),
        recurrence_count=recurrence,
        qt_medications=qt_meds,
    )

    steps = list(state.get("steps_taken", []))
    steps.append(
        f"signal_analyst: risk={risk_score:.3f}, "
        f"recurrence={recurrence}, qt_meds={qt_meds}"
    )

    return {
        **state,
        "fhir_context": fhir_context,
        "qt_medications": qt_meds,
        "recurrence_count": recurrence,
        "risk_score": risk_score,
        "requires_escalation": risk_score >= ESCALATION_THRESHOLD,
        "action": "escalate" if risk_score >= ESCALATION_THRESHOLD else "report",
        "steps_taken": steps,
    }


def node_rag_reporter(state: OmateState, rag_engine: ClinicalRAGEngine) -> OmateState:
    """Generate clinical report with hallucination guards."""
    result = rag_engine.generate_report(
        patient_id=state["patient_id"],
        signal_events=state.get("signal_events", []),
    )
    steps = list(state.get("steps_taken", []))
    steps.append(
        f"rag_reporter: status={result.status}, confidence={result.confidence:.3f}"
    )
    return {
        **state,
        "report_draft": result.report,
        "report_confidence": result.confidence,
        "guard_result": {
            "passed": result.guard_result.passed if result.guard_result else False,
            "confidence": result.guard_result.confidence if result.guard_result else 0.0,
            "citation_score": result.guard_result.citation_score if result.guard_result else 0.0,
            "consistency_score": result.guard_result.consistency_score if result.guard_result else 0.0,
            "flagged_sentences": result.guard_result.flagged_sentences if result.guard_result else [],
            "reason": result.message,
        } if result.guard_result else None,
        "steps_taken": steps,
    }


def node_escalation(state: OmateState, tools: OmateTools) -> OmateState:
    """Alert on-call physician. Bypasses report generation for urgent cases."""
    # Find most severe anomaly class
    events = state.get("signal_events", [])
    anomaly_class = (
        max(events, key=lambda e: e.get("risk_score", 0)).get("anomaly_class", "Unknown")
        if events else "Unknown"
    )
    tools.alert_on_call_physician(
        patient_id=state["patient_id"],
        risk_score=state.get("risk_score", 0.0),
        anomaly_class=anomaly_class,
    )
    steps = list(state.get("steps_taken", []))
    steps.append(f"escalation: on-call alerted for {anomaly_class}")
    return {
        **state,
        "action": "done",
        "steps_taken": steps,
    }


def node_hitl_review(state: OmateState, tools: OmateTools) -> OmateState:
    """
    Human-in-the-loop checkpoint.
    In LangGraph: graph.compile(interrupt_before=["hitl_review"])
    In this POC: implemented as an interactive prompt in demo mode,
    or skipped with a default decision in test mode.
    """
    decision = state.get("physician_decision")

    # If no decision yet, pause here (in real LangGraph this is an interrupt)
    if decision is None:
        return {**state, "action": "waiting_for_physician"}

    steps = list(state.get("steps_taken", []))
    steps.append(f"hitl_review: physician_decision={decision}")

    if decision == "approved":
        # Write final report to FHIR store
        cited = state.get("guard_result", {}).get("cited_observations", []) or []
        report_id = tools.write_diagnostic_report(
            patient_id=state["patient_id"],
            report_text=state.get("report_draft", ""),
            confidence=state.get("report_confidence", 0.0),
            cited_observations=cited,
        )
        steps.append(f"hitl_review: DiagnosticReport written → {report_id}")
        return {**state, "action": "done", "steps_taken": steps}

    elif decision == "escalated":
        steps.append("hitl_review: physician escalated — routing to escalation")
        return {
            **state,
            "action": "escalate",
            "requires_escalation": True,
            "steps_taken": steps,
        }

    else:  # rejected — loop back for retry
        steps.append("hitl_review: report rejected — reanalyzing")
        return {
            **state,
            "physician_decision": None,
            "report_draft": None,
            "action": "analyze",
            "steps_taken": steps,
        }


# ---------------------------------------------------------------------------
# Simple state machine runner (works without LangGraph installed)
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    final_state: OmateState
    steps: list[str]
    outcome: str    # "report_approved" | "escalated" | "rejected" | "error"


class OmateAgentGraph:
    """
    Omate agent graph runner.

    Uses LangGraph if available, falls back to a simple state machine loop.
    Behavior is equivalent — LangGraph adds persistence, streaming, and
    proper interrupt/resume for async physician review.
    """

    def __init__(self, fhir_store: FHIRStore, rag_engine: ClinicalRAGEngine):
        self.tools = OmateTools(fhir_store)
        self.rag_engine = rag_engine
        self._try_build_langgraph()

    def _try_build_langgraph(self):
        """Attempt to build LangGraph graph. Silently falls back if not installed."""
        self._lg_app = None
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.checkpoint.memory import MemorySaver

            graph = StateGraph(OmateState)

            graph.add_node("signal_analyst",
                           lambda s: node_signal_analyst(s, self.tools))
            graph.add_node("rag_reporter",
                           lambda s: node_rag_reporter(s, self.rag_engine))
            graph.add_node("escalation",
                           lambda s: node_escalation(s, self.tools))
            graph.add_node("hitl_review",
                           lambda s: node_hitl_review(s, self.tools))

            graph.set_entry_point("signal_analyst")

            def route_after_analyst(state):
                if state.get("requires_escalation"):
                    return "escalation"
                return "rag_reporter"

            def route_after_hitl(state):
                decision = state.get("physician_decision")
                action = state.get("action")
                if action == "done":
                    return END
                if action == "escalate":
                    return "escalation"
                if action == "analyze":
                    return "signal_analyst"
                return END

            graph.add_conditional_edges("signal_analyst", route_after_analyst)
            graph.add_edge("rag_reporter", "hitl_review")
            graph.add_conditional_edges("hitl_review", route_after_hitl)
            graph.add_edge("escalation", END)

            self._lg_app = graph.compile(
                checkpointer=MemorySaver(),
                interrupt_before=["hitl_review"],
            )
        except ImportError:
            pass

    def run(self, patient_id: str, signal_events: list[dict],
             physician_decision: str = "approved",
             physician_notes: str = "") -> AgentRunResult:
        """
        Run the full agent graph for a patient.

        Args:
            patient_id: FHIR patient ID
            signal_events: list of SignalEvent dicts
            physician_decision: "approved" | "rejected" | "escalated"
                                 (simulates physician review in POC)
            physician_notes: optional notes from physician

        Returns:
            AgentRunResult with final state and outcome
        """
        initial_state: OmateState = {
            "patient_id": patient_id,
            "signal_events": signal_events,
            "fhir_context": {},
            "qt_medications": [],
            "risk_score": 0.0,
            "recurrence_count": 0,
            "report_draft": None,
            "report_confidence": 0.0,
            "guard_result": None,
            "action": "analyze",
            "requires_escalation": False,
            "physician_decision": None,
            "physician_notes": physician_notes,
            "steps_taken": [],
        }

        # Use LangGraph if available, otherwise simple state machine
        if self._lg_app is not None:
            return self._run_with_langgraph(initial_state, physician_decision)
        else:
            return self._run_state_machine(initial_state, physician_decision)

    def _run_state_machine(self, state: OmateState,
                             physician_decision: str) -> AgentRunResult:
        """Simple state machine fallback (no LangGraph required)."""
        max_steps = 10
        for _ in range(max_steps):
            action = state.get("action", "analyze")

            if action in ("analyze", "report") and not state.get("fhir_context"):
                state = node_signal_analyst(state, self.tools)
            elif action == "report" or (
                state.get("fhir_context") and not state.get("report_draft")
                and not state.get("requires_escalation")
            ):
                state = node_rag_reporter(state, self.rag_engine)
                # Inject physician decision for HITL
                state = {**state, "physician_decision": physician_decision}
                state = node_hitl_review(state, self.tools)
            elif state.get("requires_escalation") or action == "escalate":
                state = node_escalation(state, self.tools)
                break
            elif action == "done":
                break
            else:
                break

        # Determine outcome
        if state.get("requires_escalation"):
            outcome = "escalated"
        elif state.get("physician_decision") == "approved":
            outcome = "report_approved"
        elif state.get("physician_decision") == "rejected":
            outcome = "rejected"
        else:
            outcome = "completed"

        return AgentRunResult(
            final_state=state,
            steps=state.get("steps_taken", []),
            outcome=outcome,
        )

    def _run_with_langgraph(self, state: OmateState,
                              physician_decision: str) -> AgentRunResult:
        """Run using LangGraph with proper interrupt/resume."""
        config = {"configurable": {"thread_id": state["patient_id"]}}

        # Phase 1: Run until interrupt (before hitl_review)
        for _ in self._lg_app.stream(state, config=config):
            pass

        # Check if interrupted at hitl_review
        current = self._lg_app.get_state(config)
        if "hitl_review" in (current.next or []):
            # Resume with physician decision
            self._lg_app.update_state(
                config,
                {"physician_decision": physician_decision},
                as_node="hitl_review",
            )
            for _ in self._lg_app.stream(None, config=config):
                pass

        final = self._lg_app.get_state(config).values

        outcome = "escalated" if final.get("requires_escalation") else (
            "report_approved" if final.get("physician_decision") == "approved"
            else "completed"
        )

        return AgentRunResult(
            final_state=final,
            steps=final.get("steps_taken", []),
            outcome=outcome,
        )

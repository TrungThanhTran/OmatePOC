"""OmateState TypedDict for the LangGraph agent graph."""

from typing import TypedDict, Literal


class OmateState(TypedDict):
    # Input
    patient_id: str
    signal_events: list        # list of SignalEvent dicts

    # Signal analysis
    fhir_context: dict         # loaded from FHIR store
    qt_medications: list       # QT-prolonging drugs found
    risk_score: float          # composite 0–1 risk score
    recurrence_count: int      # number of anomaly events in last 24h

    # Report generation
    report_draft: str | None
    report_confidence: float
    guard_result: dict | None

    # Routing
    action: Literal["analyze", "report", "escalate", "done"]
    requires_escalation: bool

    # Human-in-the-loop
    physician_decision: Literal["approved", "rejected", "escalated"] | None
    physician_notes: str | None

    # Audit
    steps_taken: list[str]     # log of agent actions for audit trail

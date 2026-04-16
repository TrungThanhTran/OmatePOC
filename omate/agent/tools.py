"""
FHIR query tools for the Omate agent.

In production these would be LangChain @tool decorated functions
calling AWS HealthLake APIs. Here they query the mock FHIR store.
"""

from ..fhir.store import FHIRStore, QT_PROLONGING_DRUGS


class OmateTools:
    """Tool collection backed by a FHIRStore instance."""

    def __init__(self, fhir_store: FHIRStore):
        self.store = fhir_store
        # Escalation alerts are buffered so demo output can print them
        # in the right order (after result tables, not mid-loop).
        self.escalation_alerts: list[str] = []

    def get_patient_signal_history(self, patient_id: str,
                                    last_n: int = 10) -> list[dict]:
        """
        Query the FHIR store for recent ECG observations for a patient.
        Production: queries HealthLake with date filter & LOINC 11524-6.
        """
        all_obs = self.store.search("Observation")
        patient_obs = [
            o for o in all_obs
            if o.get("subject", {}).get("reference") == f"Patient/{patient_id}"
            and any(
                c.get("code") == "11524-6"
                for c in o.get("code", {}).get("coding", [])
            )
        ]
        return patient_obs[-last_n:]

    def check_qt_prolonging_medications(self, patient_id: str) -> list[str]:
        """
        Check whether patient is on any QT-prolonging medications.
        Cross-references MedicationStatement against CredibleMeds database.
        """
        meds = self.store.search("MedicationStatement")
        patient_meds = [
            m for m in meds
            if m.get("subject", {}).get("reference") == f"Patient/{patient_id}"
            and m.get("status") == "active"
        ]
        return [
            m["medicationCodeableConcept"]["text"]
            for m in patient_meds
            if m.get("medicationCodeableConcept", {}).get("text", "").lower()
            in QT_PROLONGING_DRUGS
        ]

    def count_anomaly_events(self, patient_id: str) -> int:
        """Count anomaly flags for patient (proxy for 24h recurrence)."""
        flags = self.store.search("Flag")
        return sum(
            1 for f in flags
            if f.get("subject", {}).get("reference") == f"Patient/{patient_id}"
            and f.get("status") == "active"
        )

    def compute_risk_score(self, signal_events: list[dict],
                            recurrence_count: int,
                            qt_medications: list[str]) -> float:
        """
        Composite risk score 0–1.

        Components:
        - base_score: max anomaly risk_score from signal events
        - recurrence: capped at 1.0 after 5 events in 24h
        - medication_risk: 0.3 if any QT-prolonging drug
        """
        if not signal_events:
            return 0.0
        base_score = max(
            (e.get("risk_score", 0.0) for e in signal_events), default=0.0
        )
        recurrence = min(recurrence_count / 5.0, 1.0)
        med_risk = 0.30 if qt_medications else 0.0
        score = base_score * 0.50 + recurrence * 0.20 + med_risk * 0.30
        return round(min(score, 1.0), 4)

    def alert_on_call_physician(self, patient_id: str,
                                  risk_score: float,
                                  anomaly_class: str) -> str:
        """
        Escalation action — notify on-call physician.
        Production: sends page via hospital paging system / PagerDuty.
        POC: prints alert and returns confirmation.
        """
        msg = (
            f"ESCALATION ALERT\n"
            f"   Patient:  {patient_id}\n"
            f"   Anomaly:  {anomaly_class}\n"
            f"   Risk:     {risk_score:.2f}\n"
            f"   Action:   Immediate physician review required"
        )
        # Buffer the alert — callers print via escalation_alerts after tables
        self.escalation_alerts.append(msg)
        return f"On-call physician notified for patient {patient_id}"

    def write_diagnostic_report(self, patient_id: str,
                                  report_text: str,
                                  confidence: float,
                                  cited_observations: list[str]) -> str:
        """Write approved report back to FHIR store as DiagnosticReport."""
        from ..fhir.store import build_diagnostic_report
        resource_id = self.store.create_resource(
            "DiagnosticReport",
            build_diagnostic_report(
                patient_id=patient_id,
                summary=report_text,
                confidence=confidence,
                status="final",
                cited_observations=cited_observations,
            ),
        )
        return resource_id

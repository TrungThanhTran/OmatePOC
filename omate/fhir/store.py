"""
Lightweight FHIR R4 mock store.

Implements the key resource types used in Omate:
  Patient, Observation, Flag, DiagnosticReport, Provenance, MedicationStatement

No external dependencies — pure Python dict store.
Swap with AWS HealthLake client for production.
"""

import uuid
from datetime import datetime, timezone
from typing import Any
from dataclasses import dataclass, field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


class FHIRStore:
    """
    In-memory FHIR R4 resource store.

    Mimics the key HealthLake API operations:
    - create_resource(resource_type, data) -> resource_id
    - get_resource(resource_type, resource_id) -> dict
    - search(resource_type, params) -> list[dict]
    - resource_exists(resource_id) -> bool
    """

    def __init__(self):
        self._store: dict[str, dict[str, dict]] = {}
        self._all: dict[str, dict] = {}  # flat lookup by ID

    def create_resource(self, resource_type: str, data: dict) -> str:
        """Create a FHIR resource. Returns the generated resource ID."""
        resource_id = f"{resource_type.lower()}-{_new_id()}"
        resource = {
            "resourceType": resource_type,
            "id": resource_id,
            "meta": {"lastUpdated": _now_iso()},
            **data,
        }
        if resource_type not in self._store:
            self._store[resource_type] = {}
        self._store[resource_type][resource_id] = resource
        self._all[resource_id] = resource
        return resource_id

    def get_resource(self, resource_type: str, resource_id: str) -> dict | None:
        return self._store.get(resource_type, {}).get(resource_id)

    def resource_exists(self, resource_id: str) -> bool:
        return resource_id in self._all

    def search(self, resource_type: str, params: dict | None = None) -> list[dict]:
        """Simple search — filters by top-level field equality."""
        resources = list(self._store.get(resource_type, {}).values())
        if not params:
            return resources
        results = []
        for r in resources:
            if all(r.get(k) == v for k, v in params.items()):
                results.append(r)
        return results

    def get_patient_context(self, patient_id: str) -> dict:
        """
        Aggregate all relevant context for a patient.
        Returns a dict that the RAG engine uses as context.
        """
        def _pid_match(r):
            if r.get("id") == patient_id:
                return True
            ids = r.get("identifier", [])
            if isinstance(ids, list):
                return any(i.get("value") == patient_id for i in ids)
            return False
        patient = next((r for r in self.search("Patient") if _pid_match(r)), None)
        observations = [
            r for r in self.search("Observation")
            if r.get("subject", {}).get("reference") == f"Patient/{patient_id}"
        ]
        flags = [
            r for r in self.search("Flag")
            if r.get("subject", {}).get("reference") == f"Patient/{patient_id}"
        ]
        medications = [
            r for r in self.search("MedicationStatement")
            if r.get("subject", {}).get("reference") == f"Patient/{patient_id}"
        ]
        reports = [
            r for r in self.search("DiagnosticReport")
            if r.get("subject", {}).get("reference") == f"Patient/{patient_id}"
        ]
        return {
            "patient": patient,
            "observations": observations[-10:],   # last 10
            "flags": flags[-5:],
            "medications": medications,
            "recent_reports": reports[-3:],       # last 3
        }


# ---------------------------------------------------------------------------
# FHIR resource builders — helpers to create well-formed resources
# ---------------------------------------------------------------------------

def build_patient(name: str, birth_date: str, gender: str = "unknown",
                   patient_id: str | None = None) -> dict:
    return {
        "identifier": [{"value": patient_id or _new_id()}],
        "name": [{"text": name}],
        "birthDate": birth_date,
        "gender": gender,
    }


def build_ecg_observation(patient_id: str, value_note: str,
                            timestamp: str | None = None,
                            anomaly_class: str = "Normal",
                            risk_score: float = 0.0) -> dict:
    """LOINC 11524-6 = ECG study."""
    return {
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "11524-6",
                         "display": "EKG study"}]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp or _now_iso(),
        "valueString": value_note,
        "component": [
            {"code": {"text": "anomaly_class"}, "valueString": anomaly_class},
            {"code": {"text": "risk_score"}, "valueDecimal": risk_score},
        ],
    }


def build_glucose_observation(patient_id: str, value_mg_dl: float,
                                timestamp: str | None = None) -> dict:
    """LOINC 14743-9 = Glucose [Moles/volume] in Capillary blood."""
    return {
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "14743-9",
                         "display": "Glucose capillary"}]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp or _now_iso(),
        "valueQuantity": {
            "value": value_mg_dl,
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL",
        },
    }


def build_anomaly_flag(patient_id: str, anomaly_class: str,
                        risk_score: float) -> dict:
    severity = "urgent" if risk_score >= 0.9 else "warning"
    return {
        "status": "active",
        "code": {
            "coding": [{"system": "http://omate.io/flags",
                         "code": anomaly_class.lower().replace(" ", "_")}],
            "text": anomaly_class,
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": _now_iso()},
        "extension": [
            {"url": "http://omate.io/risk_score", "valueDecimal": risk_score},
            {"url": "http://omate.io/severity", "valueString": severity},
        ],
    }


def build_diagnostic_report(patient_id: str, summary: str,
                              confidence: float, status: str = "preliminary",
                              cited_observations: list[str] | None = None) -> dict:
    return {
        "status": status,
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "59776-5",
                         "display": "Procedure findings Narrative"}]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "issued": _now_iso(),
        "conclusion": summary,
        "extension": [
            {"url": "http://omate.io/confidence", "valueDecimal": confidence},
            {"url": "http://omate.io/cited_observations",
             "valueString": ",".join(cited_observations or [])},
        ],
    }


def build_medication_statement(patient_id: str, drug_name: str,
                                 status: str = "active") -> dict:
    return {
        "status": status,
        "medicationCodeableConcept": {"text": drug_name},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectivePeriod": {"start": _now_iso()},
    }


# ---------------------------------------------------------------------------
# Demo data loader
# ---------------------------------------------------------------------------

QT_PROLONGING_DRUGS = {
    "amiodarone", "sotalol", "dofetilide", "quinidine",
    "haloperidol", "chlorpromazine", "azithromycin", "ciprofloxacin",
    "fluconazole", "ondansetron", "methadone", "citalopram",
}


def seed_demo_store(store: FHIRStore) -> dict[str, str]:
    """
    Populate the FHIR store with demo patients and observations.
    Returns mapping of {patient_name: patient_id}.
    """
    ids = {}

    # Patient A — normal ECG, stable glucose
    p_id = store.create_resource("Patient", build_patient(
        "Maria Santos", "1965-03-14", "female", "patient-A"))
    ids["Maria Santos"] = "patient-A"
    store.create_resource("Observation", build_ecg_observation(
        "patient-A", "Normal sinus rhythm", risk_score=0.05))
    store.create_resource("Observation", build_glucose_observation(
        "patient-A", 98.0))

    # Patient B — AFib detected, on QT-prolonging medication
    store.create_resource("Patient", build_patient(
        "João Costa", "1952-08-22", "male", "patient-B"))
    ids["João Costa"] = "patient-B"
    store.create_resource("Observation", build_ecg_observation(
        "patient-B", "Irregular R-R intervals, possible AFib",
        anomaly_class="Atrial Fibrillation", risk_score=0.78))
    store.create_resource("Observation", build_ecg_observation(
        "patient-B", "Continued irregularity",
        anomaly_class="Atrial Fibrillation", risk_score=0.82))
    store.create_resource("Observation", build_glucose_observation(
        "patient-B", 142.0))
    store.create_resource("MedicationStatement",
                           build_medication_statement("patient-B", "amiodarone"))
    store.create_resource("Flag", build_anomaly_flag(
        "patient-B", "Atrial Fibrillation", 0.82))

    # Patient C — ST elevation, urgent
    store.create_resource("Patient", build_patient(
        "Ana Lima", "1978-11-05", "female", "patient-C"))
    ids["Ana Lima"] = "patient-C"
    store.create_resource("Observation", build_ecg_observation(
        "patient-C", "ST elevation in leads II, III, aVF",
        anomaly_class="ST Elevation", risk_score=0.93))
    store.create_resource("Flag", build_anomaly_flag(
        "patient-C", "ST Elevation", 0.93))

    return ids

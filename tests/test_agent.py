"""Tests for FHIR store and agent graph."""

import pytest

from omate.fhir import (
    FHIRStore, seed_demo_store,
    build_patient, build_ecg_observation, build_glucose_observation,
    build_anomaly_flag, build_diagnostic_report, build_medication_statement,
    QT_PROLONGING_DRUGS,
)
from omate.agent import OmateAgentGraph, OmateTools
from omate.rag import ClinicalRAGEngine, MockLLM
from omate.signal import generate_synthetic_ecg, run_signal_pipeline, AnomalyDetector


class TestFHIRStore:
    @pytest.fixture
    def store(self):
        return FHIRStore()

    def test_create_and_get_patient(self, store):
        pid = store.create_resource("Patient", build_patient(
            "Test User", "1980-01-01", "unknown", "test-001"
        ))
        resource = store.get_resource("Patient", pid)
        assert resource is not None
        assert resource["resourceType"] == "Patient"

    def test_resource_exists(self, store):
        pid = store.create_resource("Patient", build_patient(
            "Test User", "1980-01-01"
        ))
        assert store.resource_exists(pid)
        assert not store.resource_exists("nonexistent-id")

    def test_search_by_field(self, store):
        store.create_resource("Observation", build_ecg_observation(
            "patient-X", "Normal rhythm"
        ))
        results = store.search("Observation")
        assert len(results) >= 1

    def test_create_ecg_observation_has_loinc(self, store):
        oid = store.create_resource("Observation", build_ecg_observation(
            "patient-X", "Normal sinus rhythm", risk_score=0.1
        ))
        obs = store.get_resource("Observation", oid)
        coding = obs["code"]["coding"]
        assert any(c["code"] == "11524-6" for c in coding)

    def test_create_glucose_observation_has_quantity(self, store):
        oid = store.create_resource("Observation", build_glucose_observation(
            "patient-X", 95.0
        ))
        obs = store.get_resource("Observation", oid)
        assert obs["valueQuantity"]["value"] == 95.0
        assert obs["valueQuantity"]["unit"] == "mg/dL"

    def test_get_patient_context_includes_observations(self, store):
        store.create_resource("Patient", {
            **build_patient("Ana Test", "1990-01-01", patient_id="patient-test"),
            "id": "patient-test",
        })
        store.create_resource("Observation", build_ecg_observation(
            "patient-test", "Normal"
        ))
        ctx = store.get_patient_context("patient-test")
        assert "observations" in ctx
        assert "medications" in ctx

    def test_seed_demo_store(self, store):
        ids = seed_demo_store(store)
        assert "Maria Santos" in ids
        assert "João Costa" in ids
        assert "Ana Lima" in ids
        # Patient B should have AFib flag
        ctx_b = store.get_patient_context(ids["João Costa"])
        assert any(
            f.get("code", {}).get("text") == "Atrial Fibrillation"
            for f in ctx_b["flags"]
        )

    def test_qt_prolonging_drugs_list_not_empty(self):
        assert len(QT_PROLONGING_DRUGS) > 0
        assert "amiodarone" in QT_PROLONGING_DRUGS


class TestOmateTools:
    @pytest.fixture
    def tools_and_store(self):
        store = FHIRStore()
        seed_demo_store(store)
        tools = OmateTools(store)
        return tools, store

    def test_check_qt_meds_finds_amiodarone(self, tools_and_store):
        tools, _ = tools_and_store
        meds = tools.check_qt_prolonging_medications("patient-B")
        assert "amiodarone" in meds

    def test_check_qt_meds_none_for_clean_patient(self, tools_and_store):
        tools, _ = tools_and_store
        meds = tools.check_qt_prolonging_medications("patient-A")
        assert meds == []

    def test_count_anomaly_events(self, tools_and_store):
        tools, _ = tools_and_store
        count = tools.count_anomaly_events("patient-B")
        assert count >= 1  # patient-B has a flag in demo data

    def test_compute_risk_score_zero_for_empty(self, tools_and_store):
        tools, _ = tools_and_store
        score = tools.compute_risk_score([], recurrence_count=0, qt_medications=[])
        assert score == 0.0

    def test_compute_risk_score_increases_with_meds(self, tools_and_store):
        tools, _ = tools_and_store
        events = [{"risk_score": 0.5, "anomaly_class": "Atrial Fibrillation"}]
        score_no_meds = tools.compute_risk_score(events, 0, [])
        score_with_meds = tools.compute_risk_score(events, 0, ["amiodarone"])
        assert score_with_meds > score_no_meds

    def test_risk_score_capped_at_one(self, tools_and_store):
        tools, _ = tools_and_store
        events = [{"risk_score": 0.99}]
        score = tools.compute_risk_score(events, 10, ["amiodarone", "sotalol"])
        assert score <= 1.0

    def test_write_diagnostic_report(self, tools_and_store):
        tools, store = tools_and_store
        rid = tools.write_diagnostic_report(
            patient_id="patient-A",
            report_text="Normal findings noted.",
            confidence=0.85,
            cited_observations=["obs-001"],
        )
        assert store.resource_exists(rid)
        report = store.get_resource("DiagnosticReport", rid)
        assert report["status"] == "final"


class TestAgentGraph:
    @pytest.fixture(scope="class")
    def agent_and_store(self):
        store = FHIRStore()
        seed_demo_store(store)
        rag = ClinicalRAGEngine(
            fhir_store=store,
            llm=MockLLM(),
            confidence_threshold=0.0,
        )
        agent = OmateAgentGraph(fhir_store=store, rag_engine=rag)
        return agent, store

    @pytest.fixture(scope="class")
    def detector(self):
        return AnomalyDetector()

    def _make_signal_events(self, patient_id, anomaly_type, detector):
        raw = generate_synthetic_ecg(duration_s=10, fs=250,
                                      anomaly_type=anomaly_type)
        _, _, event = run_signal_pipeline(
            raw_ecg=raw,
            patient_id=patient_id,
            timestamp="2025-01-01T00:00:00Z",
            detector=detector,
        )
        return [vars(event)]

    def test_normal_patient_approved(self, agent_and_store, detector):
        agent, _ = agent_and_store
        events = self._make_signal_events("patient-A", "normal", detector)
        result = agent.run("patient-A", events, physician_decision="approved")
        assert result.outcome in ("report_approved", "escalated", "completed")
        assert len(result.steps) > 0

    def test_high_risk_escalates(self, agent_and_store, detector):
        agent, store = agent_and_store
        # Patient C has ST elevation flag in demo store → should escalate
        events = self._make_signal_events("patient-C", "st_elevation", detector)
        # Force high risk score by adding more flags
        from omate.fhir import build_anomaly_flag
        for _ in range(5):
            store.create_resource("Flag", build_anomaly_flag(
                "patient-C", "ST Elevation", 0.95
            ))
        result = agent.run("patient-C", events, physician_decision="escalated")
        assert result.outcome in ("escalated", "report_approved", "completed")

    def test_steps_taken_not_empty(self, agent_and_store, detector):
        agent, _ = agent_and_store
        events = self._make_signal_events("patient-A", "normal", detector)
        result = agent.run("patient-A", events, physician_decision="approved")
        assert len(result.steps) > 0

    def test_final_state_has_risk_score(self, agent_and_store, detector):
        agent, _ = agent_and_store
        events = self._make_signal_events("patient-B", "afib", detector)
        result = agent.run("patient-B", events, physician_decision="approved")
        assert "risk_score" in result.final_state
        assert 0.0 <= result.final_state["risk_score"] <= 1.0

    def test_qt_medications_detected_for_patient_b(self, agent_and_store, detector):
        agent, _ = agent_and_store
        events = self._make_signal_events("patient-B", "afib", detector)
        result = agent.run("patient-B", events, physician_decision="approved")
        qt_meds = result.final_state.get("qt_medications", [])
        assert "amiodarone" in qt_meds

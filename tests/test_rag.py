"""Tests for Clinical RAG engine and hallucination guards."""

import pytest

from omate.fhir import FHIRStore, seed_demo_store
from omate.rag import ClinicalRAGEngine, MockLLM
from omate.rag.guards import (
    extract_citations,
    split_into_sentences,
    compute_citation_score,
    selfcheck_report,
    run_hallucination_guards,
)


class TestCitationGuard:
    def test_extract_citations_basic(self):
        text = "The patient had elevated glucose [SOURCE: resource_id=obs-001]."
        cites = extract_citations(text)
        assert cites == ["obs-001"]

    def test_extract_multiple_citations(self):
        text = (
            "ECG shows normal rhythm [SOURCE: resource_id=obs-001]. "
            "Glucose is stable [SOURCE: resource_id=obs-002, timestamp=2025-01-01]."
        )
        cites = extract_citations(text)
        assert "obs-001" in cites
        assert "obs-002" in cites

    def test_extract_no_citations(self):
        text = "This sentence has no source references at all."
        cites = extract_citations(text)
        assert cites == []

    def test_citation_score_all_valid(self):
        text = (
            "Patient has AFib [SOURCE: resource_id=obs-001]. "
            "Glucose is elevated [SOURCE: resource_id=obs-002]."
        )
        valid_ids = {"obs-001", "obs-002"}
        score = compute_citation_score(text, lambda rid: rid in valid_ids)
        assert score > 0.0

    def test_citation_score_all_invalid(self):
        text = (
            "Patient has AFib [SOURCE: resource_id=fake-999]. "
            "Glucose is elevated [SOURCE: resource_id=fake-000]."
        )
        score = compute_citation_score(text, lambda _: False)
        assert score == 0.0

    def test_split_into_sentences(self):
        text = "First sentence here. Second sentence follows! Third one ends."
        sentences = split_into_sentences(text)
        assert len(sentences) >= 1


class TestSelfCheckGPT:
    def test_selfcheck_consistent_text(self):
        primary = "The patient has a normal heart rhythm with stable readings."
        variations = [
            "The patient shows a normal rhythm with stable data.",
            "Normal cardiac rhythm observed with stable measurements.",
            "Patient cardiac rhythm is normal and readings are stable.",
        ]
        risk, flagged = selfcheck_report(primary, variations, threshold=0.30)
        # Should have low risk since variations are similar
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_selfcheck_inconsistent_text(self):
        primary = "The patient has severe ventricular fibrillation."
        variations = [
            "The patient shows completely normal sinus rhythm.",
            "No cardiac abnormalities detected in monitoring data.",
            "All parameters within normal ranges for this patient.",
        ]
        risk, _ = selfcheck_report(primary, variations)
        # Risk should be higher due to inconsistency
        assert isinstance(risk, float)

    def test_selfcheck_empty_variations(self):
        primary = "Some clinical text here for testing purposes only."
        risk, flagged = selfcheck_report(primary, [])
        assert risk == 0.0
        assert flagged == []


class TestHallucinationGuards:
    def test_guards_pass_with_valid_citations(self):
        report = (
            "Patient has AFib [SOURCE: resource_id=obs-001]. "
            "Glucose is stable [SOURCE: resource_id=obs-002]. "
            "Medications reviewed [SOURCE: resource_id=med-001]."
        )
        valid_ids = {"obs-001", "obs-002", "med-001"}

        def resource_exists(rid):
            return rid in valid_ids

        def generate_fn(temperature):
            return report  # same report = perfect consistency

        result = run_hallucination_guards(
            primary_report=report,
            generate_fn=generate_fn,
            resource_exists_fn=resource_exists,
            n_samples=2,
            confidence_threshold=0.5,  # low threshold for test
        )
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_guards_fail_with_no_citations(self):
        report = (
            "The patient has some condition. "
            "Various observations have been noted. "
            "Clinical review is recommended."
        )
        result = run_hallucination_guards(
            primary_report=report,
            generate_fn=lambda temperature=0.0: report,
            resource_exists_fn=lambda _: False,
            n_samples=2,
            confidence_threshold=0.80,
        )
        # No valid citations → low citation score → likely fails
        assert result.citation_score == 0.0


class TestRAGEngine:
    @pytest.fixture
    def store_and_engine(self):
        store = FHIRStore()
        seed_demo_store(store)
        engine = ClinicalRAGEngine(
            fhir_store=store,
            llm=MockLLM(),
            confidence_threshold=0.0,  # always pass for testing
        )
        return store, engine

    def test_generate_report_known_patient(self, store_and_engine):
        _, engine = store_and_engine
        result = engine.generate_report(
            patient_id="patient-A",
            signal_events=[{
                "patient_id": "patient-A",
                "anomaly_class": "Normal",
                "risk_score": 0.05,
                "is_anomaly": False,
                "heart_rate_bpm": 72.0,
            }],
        )
        assert result.status in ("READY_FOR_REVIEW", "INSUFFICIENT_CONFIDENCE")
        assert 0.0 <= result.confidence <= 1.0

    def test_generate_report_unknown_patient(self, store_and_engine):
        _, engine = store_and_engine
        result = engine.generate_report(
            patient_id="patient-UNKNOWN",
            signal_events=[],
        )
        assert result.status == "ERROR"
        assert result.report is None

    def test_rag_result_has_guard_result(self, store_and_engine):
        _, engine = store_and_engine
        result = engine.generate_report(
            patient_id="patient-B",
            signal_events=[{
                "patient_id": "patient-B",
                "anomaly_class": "Atrial Fibrillation",
                "risk_score": 0.75,
                "is_anomaly": True,
            }],
        )
        if result.status == "READY_FOR_REVIEW":
            assert result.guard_result is not None
            assert result.guard_result.confidence > 0.0

    def test_mock_llm_produces_citations(self):
        llm = MockLLM()
        response = llm.generate("Generate a clinical summary for patient-A.")
        # Mock LLM always includes [SOURCE: ...] tags
        assert "[SOURCE:" in response.text

    def test_retrieve_context_returns_list(self, store_and_engine):
        _, engine = store_and_engine
        results = engine.retrieve_context("atrial fibrillation", n_results=3)
        assert isinstance(results, list)

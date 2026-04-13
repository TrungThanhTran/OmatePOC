"""
Tests for omate/monitoring — MLflow tracker and signal drift detector.
"""
from __future__ import annotations

import numpy as np
import pytest

from omate.monitoring.mlflow_tracker import (
    MockMLflowTracker,
    MLflowTracker,
    RunMetrics,
    get_mlflow_tracker,
)
from omate.monitoring.drift_detector import (
    DriftDetector,
    DriftReport,
    SignalStats,
    compute_signal_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    patient_id: str = "patient-A",
    anomaly: str = "Normal",
    outcome: str = "report_approved",
    risk: float = 0.10,
    confidence: float = 0.85,
) -> RunMetrics:
    return RunMetrics(
        run_id="test-run",
        patient_id=patient_id,
        timestamp="2026-01-01T00:00:00Z",
        anomaly_class=anomaly,
        risk_score=risk,
        confidence=confidence,
        citation_score=0.90,
        consistency_score=0.88,
        agent_outcome=outcome,
        latency_signal_ms=15.0,
        latency_rag_ms=200.0,
        model_version="test-v1",
        backend="mock",
    )


def _normal_signal(seed: int = 0, n: int = 2500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


# ---------------------------------------------------------------------------
# MockMLflowTracker
# ---------------------------------------------------------------------------

class TestMockMLflowTracker:

    def test_empty_summary(self):
        t = MockMLflowTracker()
        s = t.get_summary()
        assert s["total_runs"] == 0
        assert s["escalations"] == 0

    def test_log_single_run(self):
        t = MockMLflowTracker()
        t.log_pipeline_run(_make_metrics())
        s = t.get_summary()
        assert s["total_runs"] == 1
        assert s["approvals"] == 1

    def test_log_multiple_outcomes(self):
        t = MockMLflowTracker()
        t.log_pipeline_run(_make_metrics("A", outcome="report_approved"))
        t.log_pipeline_run(_make_metrics("B", outcome="escalated", risk=0.95))
        t.log_pipeline_run(_make_metrics("C", outcome="rejected"))
        s = t.get_summary()
        assert s["total_runs"] == 3
        assert s["approvals"] == 1
        assert s["escalations"] == 1
        assert s["rejections"] == 1

    def test_anomaly_distribution(self):
        t = MockMLflowTracker()
        t.log_pipeline_run(_make_metrics(anomaly="AFib"))
        t.log_pipeline_run(_make_metrics(anomaly="AFib"))
        t.log_pipeline_run(_make_metrics(anomaly="Normal"))
        s = t.get_summary()
        assert s["anomaly_distribution"]["AFib"] == 2
        assert s["anomaly_distribution"]["Normal"] == 1

    def test_avg_confidence(self):
        t = MockMLflowTracker()
        t.log_pipeline_run(_make_metrics(confidence=0.80))
        t.log_pipeline_run(_make_metrics(confidence=0.90))
        s = t.get_summary()
        assert abs(s["avg_confidence"] - 0.85) < 0.01

    def test_get_recent_runs_limit(self):
        t = MockMLflowTracker()
        for i in range(10):
            t.log_pipeline_run(_make_metrics(f"patient-{i}"))
        recent = t.get_recent_runs(3)
        assert len(recent) == 3

    def test_get_recent_runs_order(self):
        t = MockMLflowTracker()
        for i in range(5):
            t.log_pipeline_run(_make_metrics(f"patient-{i}"))
        recent = t.get_recent_runs(2)
        assert recent[-1].patient_id == "patient-4"

    def test_start_end_run_interface(self):
        t = MockMLflowTracker()
        run_id = t.start_run("test-run", {"tag": "value"})
        assert run_id.startswith("mock-run-")
        t.log_param("key", "value")
        t.log_metric("score", 0.9)
        t.log_metrics({"a": 1.0, "b": 2.0})
        t.set_tag("status", "ok")
        t.end_run()
        assert t._active is None

    def test_tracking_uri_is_mock(self):
        t = MockMLflowTracker()
        assert "mock" in t.tracking_uri.lower()


# ---------------------------------------------------------------------------
# MLflowTracker (falls back to mock when mlflow not installed)
# ---------------------------------------------------------------------------

class TestMLflowTracker:

    def test_falls_back_to_mock_when_not_installed(self):
        t = MLflowTracker(experiment_name="test-experiment")
        # MLflow not installed in test env → mock should be set
        assert t._mock is not None

    def test_log_and_summarise(self):
        t = MLflowTracker(experiment_name="test-experiment")
        t.log_pipeline_run(_make_metrics())
        s = t.get_summary()
        assert s["total_runs"] == 1

    def test_get_recent_runs(self):
        t = MLflowTracker(experiment_name="test-experiment")
        for i in range(5):
            t.log_pipeline_run(_make_metrics(f"patient-{i}"))
        assert len(t.get_recent_runs(3)) == 3


def test_get_mlflow_tracker_returns_instance():
    tracker = get_mlflow_tracker("test-omate")
    assert tracker is not None


# ---------------------------------------------------------------------------
# compute_signal_stats
# ---------------------------------------------------------------------------

class TestComputeSignalStats:

    def test_returns_signal_stats(self):
        sig = _normal_signal()
        stats = compute_signal_stats(sig, timestamp="2026-01-01T00:00:00Z")
        assert isinstance(stats, SignalStats)

    def test_mean_near_zero_for_normal(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(10_000).astype(np.float32)
        stats = compute_signal_stats(sig)
        assert abs(stats.mean) < 0.05

    def test_std_near_one_for_standard_normal(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(10_000).astype(np.float32)
        stats = compute_signal_stats(sig)
        assert 0.95 < stats.std < 1.05

    def test_rms_positive(self):
        sig = _normal_signal()
        stats = compute_signal_stats(sig)
        assert stats.rms > 0

    def test_amplitude_range_positive(self):
        sig = _normal_signal()
        stats = compute_signal_stats(sig)
        assert stats.amplitude_range > 0

    def test_zero_crossing_rate_bounded(self):
        sig = _normal_signal()
        stats = compute_signal_stats(sig)
        assert 0.0 <= stats.zero_crossing_rate <= 1.0

    def test_timestamp_preserved(self):
        sig = _normal_signal()
        stats = compute_signal_stats(sig, timestamp="2026-01-01T12:00:00Z")
        assert stats.timestamp == "2026-01-01T12:00:00Z"


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:

    def test_baseline_not_ready_returns_no_drift(self):
        d = DriftDetector(min_baseline_size=10)
        sig = _normal_signal()
        report = d.check(sig)
        assert isinstance(report, DriftReport)
        assert report.drift_detected is False
        assert report.drift_score == 0.0

    def test_baseline_not_ready_message(self):
        d = DriftDetector(min_baseline_size=10)
        report = d.check(_normal_signal())
        assert "0/" in report.recommendation or "building" in report.recommendation.lower()

    def test_stable_distribution_no_drift(self):
        rng = np.random.default_rng(1)
        d = DriftDetector(min_baseline_size=5, p_threshold=0.01)
        # Build baseline from N(0, 0.1)
        for _ in range(15):
            d.add_to_baseline(rng.normal(0, 0.1, 2500).astype(np.float32))
        # Add production samples from same distribution
        for _ in range(5):
            d.check(rng.normal(0, 0.1, 2500).astype(np.float32))
        report = d.check(rng.normal(0, 0.1, 2500).astype(np.float32))
        # Same distribution should not trigger catastrophic drift
        assert report.drift_score < 0.7

    def test_large_shift_detected(self):
        rng = np.random.default_rng(2)
        d = DriftDetector(min_baseline_size=5, p_threshold=0.05)
        # Baseline: N(0, 0.05) — tiny amplitude
        for _ in range(15):
            d.add_to_baseline(rng.normal(0, 0.05, 2500).astype(np.float32))
        # Production: N(10, 3) — massive shift
        for _ in range(10):
            d.check(rng.normal(10, 3, 2500).astype(np.float32))
        report = d.check(rng.normal(10, 3, 2500).astype(np.float32))
        assert report.drift_detected is True
        assert len(report.drifted_features) > 0

    def test_drifted_features_are_valid(self):
        rng = np.random.default_rng(3)
        d = DriftDetector(min_baseline_size=5, p_threshold=0.05)
        for _ in range(15):
            d.add_to_baseline(rng.normal(0, 0.05, 2500).astype(np.float32))
        for _ in range(10):
            d.check(rng.normal(10, 3, 2500).astype(np.float32))
        report = d.check(rng.normal(10, 3, 2500).astype(np.float32))
        valid_features = {
            "mean", "std", "rms", "skewness",
            "kurtosis", "amplitude_range", "zero_crossing_rate",
        }
        for feat in report.drifted_features:
            assert feat in valid_features

    def test_p_values_returned(self):
        rng = np.random.default_rng(4)
        d = DriftDetector(min_baseline_size=5)
        for _ in range(10):
            d.add_to_baseline(rng.normal(0, 1, 2500).astype(np.float32))
        for _ in range(5):
            d.check(rng.normal(0, 1, 2500).astype(np.float32))
        report = d.check(rng.normal(0, 1, 2500).astype(np.float32))
        assert isinstance(report.p_values, dict)

    def test_drift_score_bounded(self):
        rng = np.random.default_rng(5)
        d = DriftDetector(min_baseline_size=5)
        for _ in range(10):
            d.add_to_baseline(rng.standard_normal(2500).astype(np.float32))
        for _ in range(5):
            d.check(rng.normal(100, 50, 2500).astype(np.float32))
        report = d.check(rng.normal(100, 50, 2500).astype(np.float32))
        assert 0.0 <= report.drift_score <= 1.0

    def test_add_to_baseline_increments(self):
        d = DriftDetector()
        assert d.baseline_size == 0
        d.add_to_baseline(_normal_signal())
        assert d.baseline_size == 1
        d.add_to_baseline(_normal_signal(seed=1))
        assert d.baseline_size == 2

    def test_reset_baseline(self):
        d = DriftDetector()
        for i in range(5):
            d.add_to_baseline(_normal_signal(seed=i))
        assert d.baseline_size == 5
        d.reset_baseline()
        assert d.baseline_size == 0
        assert d.history_size == 0

    def test_baseline_size_reported_correctly(self):
        d = DriftDetector(min_baseline_size=5)
        for i in range(7):
            d.add_to_baseline(_normal_signal(seed=i))
        for i in range(5):
            d.check(_normal_signal(seed=i + 100))
        report = d.check(_normal_signal(seed=200))
        assert report.baseline_size == 7

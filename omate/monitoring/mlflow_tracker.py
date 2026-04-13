"""
MLflow experiment tracker for the Omate pipeline.

Logs every inference run, model version, and clinical outcome.
Falls back to an in-memory store when MLflow is not installed.

Production setup:
  pip install mlflow
  mlflow server --host 0.0.0.0 --port 5000 \\
    --backend-store-uri postgresql://omate:pass@db:5432/mlflow \\
    --default-artifact-root s3://omate-mlflow-artifacts/
  export MLFLOW_TRACKING_URI=http://mlflow.internal:5000

Tracked per run:
  params  : patient_id, anomaly_class, agent_outcome, model_version, backend
  metrics : risk_score, confidence, citation_score, consistency_score,
            latency_signal_ms, latency_rag_ms
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RunMetrics:
    """All metrics captured for a single end-to-end pipeline run."""
    run_id: str
    patient_id: str
    timestamp: str
    anomaly_class: str
    risk_score: float
    confidence: float
    citation_score: float
    consistency_score: float
    agent_outcome: str
    latency_signal_ms: float
    latency_rag_ms: float
    model_version: str
    backend: str


# ---------------------------------------------------------------------------
# Mock tracker (no MLflow dependency)
# ---------------------------------------------------------------------------

class MockMLflowTracker:
    """
    In-memory experiment tracker.

    Stores runs in a plain list. Provides the same interface as the real
    MLflowTracker so the pipeline is backend-agnostic.
    """

    def __init__(self, experiment_name: str = "omate-poc"):
        self.experiment_name = experiment_name
        self._runs: list[RunMetrics] = []
        self._run_counter = 0
        self._active: dict[str, Any] | None = None

    # --- MLflow-compatible interface ---

    def start_run(self, run_name: str = "",
                  tags: dict[str, str] | None = None) -> str:
        self._run_counter += 1
        run_id = f"mock-run-{self._run_counter:04d}"
        self._active = {
            "run_id": run_id,
            "run_name": run_name,
            "metrics": {},
            "params": {},
            "tags": tags or {},
            "start_time": time.time(),
        }
        return run_id

    def log_param(self, key: str, value: Any) -> None:
        if self._active:
            self._active["params"][key] = value

    def log_metric(self, key: str, value: float,
                   step: int | None = None) -> None:
        if self._active:
            self._active["metrics"][key] = value

    def log_metrics(self, metrics: dict[str, float],
                    step: int | None = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def set_tag(self, key: str, value: str) -> None:
        if self._active:
            self._active["tags"][key] = value

    def end_run(self, status: str = "FINISHED") -> None:
        self._active = None

    # --- Omate-specific helpers ---

    def log_pipeline_run(self, metrics: RunMetrics) -> None:
        """Append a complete pipeline run to the in-memory store."""
        self._runs.append(metrics)

    def get_summary(self) -> dict[str, Any]:
        """Aggregate statistics across all logged runs."""
        runs = self._runs
        if not runs:
            return {
                "total_runs": 0,
                "avg_confidence": 0.0,
                "avg_risk_score": 0.0,
                "escalations": 0,
                "approvals": 0,
                "rejections": 0,
                "anomaly_distribution": {},
                "avg_latency_signal_ms": 0.0,
                "avg_latency_rag_ms": 0.0,
                "drift_alert": None,
            }
        outcomes = [r.agent_outcome for r in runs]
        anomaly_dist: dict[str, int] = {}
        for r in runs:
            anomaly_dist[r.anomaly_class] = (
                anomaly_dist.get(r.anomaly_class, 0) + 1
            )
        return {
            "total_runs": len(runs),
            "avg_confidence": round(
                sum(r.confidence for r in runs) / len(runs), 3
            ),
            "avg_risk_score": round(
                sum(r.risk_score for r in runs) / len(runs), 3
            ),
            "escalations": outcomes.count("escalated"),
            "approvals": outcomes.count("report_approved"),
            "rejections": outcomes.count("rejected"),
            "anomaly_distribution": anomaly_dist,
            "avg_latency_signal_ms": round(
                sum(r.latency_signal_ms for r in runs) / len(runs), 1
            ),
            "avg_latency_rag_ms": round(
                sum(r.latency_rag_ms for r in runs) / len(runs), 1
            ),
            "drift_alert": None,
        }

    def get_recent_runs(self, n: int = 5) -> list[RunMetrics]:
        return self._runs[-n:]

    @property
    def tracking_uri(self) -> str:
        return "in-memory (mock)"


# ---------------------------------------------------------------------------
# Real MLflow tracker
# ---------------------------------------------------------------------------

class MLflowTracker:
    """
    MLflow-backed experiment tracker.

    Automatically falls back to MockMLflowTracker when MLflow is not
    installed (import error) or MLFLOW_TRACKING_URI is not set.
    """

    def __init__(self, experiment_name: str = "omate-poc",
                 tracking_uri: str | None = None):
        self.experiment_name = experiment_name
        self._mock: MockMLflowTracker | None = None
        self._mlflow = None
        self._runs: list[RunMetrics] = []

        try:
            import mlflow
            self._mlflow = mlflow
            uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
        except ImportError:
            self._mock = MockMLflowTracker(experiment_name)

    # --- MLflow-compatible interface ---

    def start_run(self, run_name: str = "",
                  tags: dict[str, str] | None = None) -> str:
        if self._mock:
            return self._mock.start_run(run_name, tags)
        run = self._mlflow.start_run(run_name=run_name, tags=tags or {})
        return run.info.run_id

    def log_param(self, key: str, value: Any) -> None:
        if self._mock:
            return self._mock.log_param(key, value)
        self._mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float,
                   step: int | None = None) -> None:
        if self._mock:
            return self._mock.log_metric(key, value, step)
        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float],
                    step: int | None = None) -> None:
        if self._mock:
            return self._mock.log_metrics(metrics, step)
        self._mlflow.log_metrics(metrics, step=step)

    def set_tag(self, key: str, value: str) -> None:
        if self._mock:
            return self._mock.set_tag(key, value)
        self._mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._mock:
            return self._mock.end_run(status)
        self._mlflow.end_run(status=status)

    # --- Omate-specific helpers ---

    def log_pipeline_run(self, metrics: RunMetrics) -> None:
        """Log a complete pipeline run (params + metrics in one call)."""
        if self._mock:
            self._mock.log_pipeline_run(metrics)
            return
        self._runs.append(metrics)
        self.start_run(
            run_name=f"{metrics.patient_id}-{metrics.anomaly_class}",
            tags={"patient_id": metrics.patient_id, "backend": metrics.backend},
        )
        self.log_metrics({
            "risk_score":         metrics.risk_score,
            "confidence":         metrics.confidence,
            "citation_score":     metrics.citation_score,
            "consistency_score":  metrics.consistency_score,
            "latency_signal_ms":  metrics.latency_signal_ms,
            "latency_rag_ms":     metrics.latency_rag_ms,
        })
        self.log_param("anomaly_class",  metrics.anomaly_class)
        self.log_param("agent_outcome",  metrics.agent_outcome)
        self.log_param("model_version",  metrics.model_version)
        self.end_run()

    def get_summary(self) -> dict[str, Any]:
        if self._mock:
            return self._mock.get_summary()
        runs = self._runs
        if not runs:
            return {"total_runs": 0}
        outcomes = [r.agent_outcome for r in runs]
        anomaly_dist: dict[str, int] = {}
        for r in runs:
            anomaly_dist[r.anomaly_class] = (
                anomaly_dist.get(r.anomaly_class, 0) + 1
            )
        return {
            "total_runs": len(runs),
            "avg_confidence": round(
                sum(r.confidence for r in runs) / len(runs), 3
            ),
            "avg_risk_score": round(
                sum(r.risk_score for r in runs) / len(runs), 3
            ),
            "escalations": outcomes.count("escalated"),
            "approvals": outcomes.count("report_approved"),
            "rejections": outcomes.count("rejected"),
            "anomaly_distribution": anomaly_dist,
            "avg_latency_signal_ms": round(
                sum(r.latency_signal_ms for r in runs) / len(runs), 1
            ),
            "avg_latency_rag_ms": round(
                sum(r.latency_rag_ms for r in runs) / len(runs), 1
            ),
            "drift_alert": None,
        }

    def get_recent_runs(self, n: int = 5) -> list[RunMetrics]:
        if self._mock:
            return self._mock.get_recent_runs(n)
        return self._runs[-n:]

    @property
    def tracking_uri(self) -> str:
        if self._mock:
            return "in-memory (mock)"
        try:
            return self._mlflow.get_tracking_uri()
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_mlflow_tracker(experiment: str = "omate-poc") -> MLflowTracker:
    """
    Return a tracker configured from environment variables.
    Falls back to in-memory mock if MLflow is not installed.

    Env var: MLFLOW_TRACKING_URI (e.g. http://mlflow.internal:5000)
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    return MLflowTracker(
        experiment_name=experiment,
        tracking_uri=uri or None,
    )

"""
Demo 3: Live Terminal Dashboard

Full end-to-end pipeline with a real-time Rich dashboard.

Shows all components updating simultaneously:
  Signal intelligence → FHIR → Triton serving → RAG → Agent → MLflow

Run:
  make demo-dashboard                   # mock LLM (no API key needed)
  make demo-dashboard-openai            # with OPENAI_API_KEY
  make demo-dashboard-anthropic         # with ANTHROPIC_API_KEY
  python -m omate.demo_dashboard        # interactive backend chooser
  python -m omate.demo_dashboard --no-interactive  # skip backend prompt
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

# Load .env before any omate imports so env vars are available
try:
    from dotenv import load_dotenv
    load_dotenv(".env", override=False)
    load_dotenv(".env.example", override=False)
except ImportError:
    pass

from omate.config import configure_llm_interactive
from omate.signal import generate_synthetic_ecg, run_signal_pipeline, AnomalyDetector
from omate.fhir import FHIRStore, seed_demo_store, build_ecg_observation, build_anomaly_flag
from omate.rag import ClinicalRAGEngine
from omate.agent import OmateAgentGraph
from omate.serving.triton_client import get_triton_client
from omate.monitoring.mlflow_tracker import get_mlflow_tracker, RunMetrics
from omate.monitoring.drift_detector import DriftDetector
from omate.dashboard.terminal import OmateDashboard, DashboardState


# ---------------------------------------------------------------------------
# Patient scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "patient_id":   "patient-A",
        "name":         "Maria Santos",
        "anomaly_type": "normal",
        "decision":     "approved",
    },
    {
        "patient_id":   "patient-B",
        "name":         "João Costa",
        "anomaly_type": "afib",
        "decision":     "approved",
    },
    {
        "patient_id":   "patient-C",
        "name":         "Ana Lima",
        "anomaly_type": "st_elevation",
        "decision":     "escalated",
    },
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hms() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_dashboard_demo(interactive: bool = True) -> None:
    # 1. LLM backend selection -----------------------------------------------
    if interactive and sys.stdin.isatty():
        configure_llm_interactive()

    # 2. Infrastructure setup ------------------------------------------------
    store   = FHIRStore()
    _       = seed_demo_store(store)
    rag     = ClinicalRAGEngine(fhir_store=store)
    agent   = OmateAgentGraph(fhir_store=store, rag_engine=rag)
    detector = AnomalyDetector()
    triton  = get_triton_client()
    tracker = get_mlflow_tracker("omate-poc")
    drift   = DriftDetector(min_baseline_size=3)   # low for demo speed

    # 3. Initial dashboard state ---------------------------------------------
    state = DashboardState(
        llm_backend=os.getenv("LLM_BACKEND", "mock"),
        status_message="Starting up...",
        patients=[
            {
                "patient_id":     sc["patient_id"],
                "name":           sc["name"],
                "risk_score":     0.0,
                "anomaly_class":  "—",
                "heart_rate_bpm": "—",
                "qt_meds":        "—",
            }
            for sc in SCENARIOS
        ],
    )

    dashboard = OmateDashboard(state)

    with dashboard.live():
        # 4. Build drift baseline from synthetic normal signals --------------
        state.status_message = "Building signal drift baseline..."
        dashboard.update()

        for _ in range(5):
            sig = generate_synthetic_ecg(
                duration_s=10.0, fs=250, anomaly_type="normal"
            )
            drift.add_to_baseline(sig)

        # 5. Show initial Triton status -------------------------------------
        state.triton_metrics = triton.get_metrics()
        state.status_message = "Infrastructure ready. Starting pipeline..."
        dashboard.update()
        time.sleep(0.6)

        # 6. Process each patient — two full cycles (simulates streaming) ---
        for cycle in range(2):
            for sc in SCENARIOS:
                pid  = sc["patient_id"]
                name = sc["name"]

                # ── Signal pipeline ─────────────────────────────────────
                state.status_message = f"[{cycle+1}/2] Signal analysis — {name}..."
                dashboard.update()
                time.sleep(0.2)

                t_sig = time.perf_counter()
                raw_ecg = generate_synthetic_ecg(
                    duration_s=10.0, fs=250,
                    heart_rate=75, noise_level=0.04,
                    anomaly_type=sc["anomaly_type"],
                )
                denoised, anomaly, event = run_signal_pipeline(
                    raw_ecg=raw_ecg,
                    patient_id=pid,
                    timestamp=_ts(),
                    detector=detector,
                )
                latency_signal_ms = (time.perf_counter() - t_sig) * 1000

                # Triton inference (mock if server not running)
                triton.infer(denoised.denoised)
                state.triton_metrics = triton.get_metrics()

                # Drift check
                drift_report = drift.check(raw_ecg, timestamp=_ts())
                state.drift_reports.append({
                    "drift_detected":  drift_report.drift_detected,
                    "drift_score":     drift_report.drift_score,
                    "drifted_features": drift_report.drifted_features,
                    "recommendation":  drift_report.recommendation,
                    "baseline_size":   drift_report.baseline_size,
                })

                # Update signal event feed
                state.signal_events.append({
                    "time":          _hms(),
                    "patient_id":    pid,
                    "anomaly_class": anomaly.predicted_class,
                    "confidence":    anomaly.confidence,
                    "risk_score":    event.risk_score,
                })

                # Update patient row
                for p in state.patients:
                    if p["patient_id"] == pid:
                        p["risk_score"]     = event.risk_score
                        p["anomaly_class"]  = anomaly.predicted_class
                        p["heart_rate_bpm"] = f"{event.heart_rate_bpm:.0f}"

                dashboard.update()
                time.sleep(0.3)

                # ── FHIR write ───────────────────────────────────────────
                store.create_resource("Observation", build_ecg_observation(
                    patient_id=pid,
                    value_note=(
                        f"{anomaly.predicted_class} "
                        f"(conf={anomaly.confidence:.2f})"
                    ),
                    timestamp=event.timestamp,
                    anomaly_class=anomaly.predicted_class,
                    risk_score=event.risk_score,
                ))
                store.create_resource("Flag", build_anomaly_flag(
                    patient_id=pid,
                    anomaly_class=anomaly.predicted_class,
                    risk_score=event.risk_score,
                ))

                # ── RAG report ───────────────────────────────────────────
                state.status_message = f"[{cycle+1}/2] Generating RAG report — {name}..."
                dashboard.update()

                t_rag = time.perf_counter()
                rag_result = rag.generate_report(
                    patient_id=pid,
                    signal_events=[vars(event)],
                )
                latency_rag_ms = (time.perf_counter() - t_rag) * 1000

                if rag_result.report:
                    state.last_report         = rag_result.report
                    state.last_report_patient = name

                dashboard.update()
                time.sleep(0.3)

                # ── Agent graph ──────────────────────────────────────────
                state.status_message = f"[{cycle+1}/2] Agent routing — {name}..."
                dashboard.update()

                agent_result = agent.run(
                    patient_id=pid,
                    signal_events=[vars(event)],
                    physician_decision=sc["decision"],
                )

                qt_meds = agent_result.final_state.get("qt_medications", [])
                for p in state.patients:
                    if p["patient_id"] == pid:
                        p["qt_meds"] = (
                            ", ".join(qt_meds) if qt_meds else "none"
                        )

                state.agent_decisions.append({
                    "patient_id": f"{pid} ({name})",
                    "outcome":    agent_result.outcome,
                    "risk_score": agent_result.final_state.get("risk_score", 0.0),
                    "steps":      len(agent_result.steps),
                })

                # ── MLflow logging ───────────────────────────────────────
                tracker.log_pipeline_run(RunMetrics(
                    run_id=f"run-{pid}-cycle{cycle}",
                    patient_id=pid,
                    timestamp=_ts(),
                    anomaly_class=anomaly.predicted_class,
                    risk_score=event.risk_score,
                    confidence=anomaly.confidence,
                    citation_score=(
                        rag_result.guard_result.citation_score
                        if rag_result.guard_result else 0.0
                    ),
                    consistency_score=(
                        rag_result.guard_result.consistency_score
                        if rag_result.guard_result else 0.0
                    ),
                    agent_outcome=agent_result.outcome,
                    latency_signal_ms=round(latency_signal_ms, 1),
                    latency_rag_ms=round(latency_rag_ms, 1),
                    model_version="patchtst-v1",
                    backend=os.getenv("LLM_BACKEND", "mock"),
                ))
                state.mlflow_summary = tracker.get_summary()

                # Update drift alert in MLflow summary
                if drift_report.drift_detected:
                    state.mlflow_summary["drift_alert"] = (
                        drift_report.drifted_features[:2]
                    )

                dashboard.update()
                time.sleep(0.7)

        # 7. Done — keep dashboard visible until Ctrl-C ---------------------
        state.status_message = (
            f"Pipeline complete — {len(state.agent_decisions)} runs. "
            "Ctrl-C to exit."
        )
        dashboard.update()

        try:
            while True:
                time.sleep(1)
                dashboard.update()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    interactive = "--no-interactive" not in sys.argv
    run_dashboard_demo(interactive=interactive)

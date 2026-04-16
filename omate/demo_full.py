"""
Demo 2: Full Omate Pipeline

Shows:
  - FHIR mock store with demo patients
  - Signal pipeline → FHIR Observation
  - Clinical RAG with hallucination guards
  - LangGraph agent with HITL simulation

Run:
  python -m omate.demo_full
  python -m omate.demo_full --no-warnings           # suppress library noise
  python -m omate.demo_full --data mitbih --record 202   # real AFib ECG
  python -m omate.demo_full --data mitbih --record 100   # real normal ECG
"""

import sys
import warnings
if "--no-warnings" in sys.argv:
    warnings.filterwarnings("ignore")

import json
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from omate.signal import generate_synthetic_ecg, run_signal_pipeline, AnomalyDetector
from omate.fhir import FHIRStore, seed_demo_store, build_ecg_observation, build_anomaly_flag
from omate.rag import ClinicalRAGEngine
from omate.agent import OmateAgentGraph

console = Console()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def section(title: str):
    console.print()
    console.print(Rule(f"[bold yellow]{title}[/bold yellow]", style="yellow"))


def _parse_args() -> dict:
    args = sys.argv[1:]
    result = {"data": None, "record": None, "start_s": None,
              "duration_s": 10.0, "classifier": "clinical"}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--data" and i + 1 < len(args):
            result["data"] = args[i + 1]; i += 2
        elif a == "--record" and i + 1 < len(args):
            result["record"] = args[i + 1]; i += 2
        elif a == "--start" and i + 1 < len(args):
            result["start_s"] = float(args[i + 1]); i += 2
        elif a == "--duration" and i + 1 < len(args):
            result["duration_s"] = float(args[i + 1]); i += 2
        elif a == "--classifier" and i + 1 < len(args):
            result["classifier"] = args[i + 1]; i += 2
        else:
            i += 1
    return result


def _load_real_ecg(record_id: str, start_s=None, duration_s=10.0):
    """Load MIT-BIH record; return (signal, fs, record_obj)."""
    from omate.signal import load_mitbih_record
    rec = load_mitbih_record(record_id, start_s=start_s, duration_s=duration_s)
    return rec.signal, rec.fs, rec


def run_demo():
    opts = _parse_args()
    use_mitbih = opts["data"] == "mitbih"
    mitbih_record = opts.get("record") or "202"  # default to AFib record

    title_suffix = (
        f"\n[dim]ECG source: MIT-BIH record {mitbih_record} "
        f"(PhysioNet · 360 Hz · real data)[/dim]"
        if use_mitbih else
        "\n[dim]ECG source: synthetic generator[/dim]"
    )
    console.print(Panel.fit(
        "[bold cyan]Omate POC — Full Pipeline Demo[/bold cyan]\n"
        f"Signal → FHIR → RAG → Agent → HITL{title_suffix}",
        border_style="cyan",
    ))

    # -----------------------------------------------------------------------
    # Step 1: Initialize FHIR store
    # -----------------------------------------------------------------------
    section("Step 1: FHIR Store")
    store = FHIRStore()
    patient_ids = seed_demo_store(store)

    t = Table(show_header=True, header_style="bold blue")
    t.add_column("Patient", style="cyan")
    t.add_column("FHIR ID")
    t.add_column("Pre-loaded conditions")
    for name, pid in patient_ids.items():
        ctx = store.get_patient_context(pid)
        flags = [f.get("code", {}).get("text", "") for f in ctx["flags"]]
        meds  = [m.get("medicationCodeableConcept", {}).get("text", "")
                 for m in ctx["medications"]]
        notes = []
        if flags:
            notes.append(f"Flags: {', '.join(flags)}")
        if meds:
            notes.append(f"Meds: {', '.join(meds)}")
        t.add_row(name, pid, "; ".join(notes) or "None")
    console.print(t)

    # -----------------------------------------------------------------------
    # Step 2: Run signal pipeline for Patient B (AFib, on amiodarone)
    # -----------------------------------------------------------------------
    section("Step 2: Signal Intelligence Pipeline — Patient B (João Costa)")

    from omate.signal import get_ecg_classifier
    clf_name = opts.get("classifier", "clinical")
    clf = get_ecg_classifier(clf_name)
    clf_status = getattr(clf, "status", clf_name)
    console.print(f"[dim]Classifier: [bold]{clf_status}[/bold] · warming up...[/dim]")
    clf.warmup()

    ecg_fs = 250
    mitbih_rec = None
    if use_mitbih:
        console.print(f"[dim]Loading MIT-BIH record {mitbih_record} (real ECG)...[/dim]")
        try:
            raw_ecg, ecg_fs, mitbih_rec = _load_real_ecg(
                mitbih_record,
                start_s=opts.get("start_s"),
                duration_s=opts["duration_s"],
            )
            console.print(
                f"  [dim]Record {mitbih_record}: {mitbih_rec.label} · "
                f"{len(raw_ecg)} samples @ {ecg_fs} Hz · "
                f"{mitbih_rec.beat_count} beats · "
                f"dominant: {mitbih_rec.dominant_rhythm}[/dim]"
            )
        except (FileNotFoundError, ImportError) as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")
            console.print("[yellow]Falling back to synthetic ECG.[/yellow]")
            raw_ecg = generate_synthetic_ecg(
                duration_s=10.0, fs=250, heart_rate=82,
                noise_level=0.05, anomaly_type="afib"
            )
            use_mitbih = False
    else:
        console.print("[dim]Generating synthetic AFib ECG + running denoising pipeline...[/dim]")
        raw_ecg = generate_synthetic_ecg(
            duration_s=10.0, fs=250, heart_rate=82,
            noise_level=0.05, anomaly_type="afib"
        )

    denoised, anomaly, event = run_signal_pipeline(
        raw_ecg=raw_ecg,
        patient_id="patient-B",
        timestamp=_ts(),
        classifier=clf,
        fs=ecg_fs,
    )

    if mitbih_rec:
        console.print(f"  Ground truth:[dim] {mitbih_rec.dominant_rhythm} "
                      f"({mitbih_rec.beat_count} annotated beats)[/dim]")
    console.print(f"  Detected:    [yellow]{anomaly.predicted_class}[/yellow]")
    console.print(f"  Risk score:  [yellow]{event.risk_score:.3f}[/yellow]")
    console.print(f"  Confidence:  {anomaly.confidence:.3f}")
    console.print(f"  Heart rate:  {event.heart_rate_bpm:.0f} bpm")
    console.print(f"  Quality:     {denoised.quality_score:.2f}")
    console.print(f"  Escalate:    {'[red]YES[/red]' if event.requires_escalation else '[green]no[/green]'}")

    # Write observation to FHIR
    obs_id = store.create_resource("Observation", build_ecg_observation(
        patient_id="patient-B",
        value_note=f"AFib detected (confidence {anomaly.confidence:.2f})",
        timestamp=event.timestamp,
        anomaly_class=anomaly.predicted_class,
        risk_score=event.risk_score,
    ))
    flag_id = store.create_resource("Flag", build_anomaly_flag(
        patient_id="patient-B",
        anomaly_class=anomaly.predicted_class,
        risk_score=event.risk_score,
    ))
    console.print(f"  → FHIR Observation written: [cyan]{obs_id}[/cyan]")
    console.print(f"  → FHIR Flag written:        [cyan]{flag_id}[/cyan]")

    # -----------------------------------------------------------------------
    # Step 3: RAG report generation with hallucination guards
    # -----------------------------------------------------------------------
    section("Step 3: Clinical RAG Engine")
    console.print("[dim]Retrieving patient context + generating report with hallucination guards...[/dim]")

    rag = ClinicalRAGEngine(fhir_store=store)
    signal_events_list = [vars(event)]  # convert dataclass to dict

    rag_result = rag.generate_report(
        patient_id="patient-B",
        signal_events=signal_events_list,
    )

    status_color = "green" if rag_result.status == "READY_FOR_REVIEW" else "red"
    console.print(f"\n  Status:      [{status_color}]{rag_result.status}[/{status_color}]")
    console.print(f"  Confidence:  {rag_result.confidence:.3f}")

    if rag_result.guard_result:
        g = rag_result.guard_result
        console.print(f"  Citation score:    {g.citation_score:.3f}")
        console.print(f"  Consistency score: {g.consistency_score:.3f}")
        if g.flagged_sentences:
            console.print(f"  [yellow]Flagged sentences: {len(g.flagged_sentences)}[/yellow]")

    if rag_result.report:
        console.print("\n[bold]Draft Report:[/bold]")
        console.print(Panel(
            rag_result.report,
            border_style="dim",
            title="[dim]Pending physician review[/dim]",
        ))
    else:
        console.print(f"\n  [yellow]Message: {rag_result.message}[/yellow]")

    # -----------------------------------------------------------------------
    # Step 4: Agent graph — simulate full run for all 3 patients
    # -----------------------------------------------------------------------
    section("Step 4: Agent Graph (LangGraph) — All Patients")
    console.print(
        "[dim]Running supervisor + specialist agent graph with simulated HITL...[/dim]\n"
    )

    agent = OmateAgentGraph(fhir_store=store, rag_engine=rag)

    scenarios = [
        {
            "patient_id":   "patient-A",
            "description":  "Maria Santos — normal rhythm",
            "anomaly_type": "normal",
            "decision":     "approved",
        },
        {
            "patient_id":   "patient-B",
            "description":  "João Costa — AFib + amiodarone",
            "anomaly_type": "afib",
            "decision":     "approved",
        },
        {
            "patient_id":   "patient-C",
            "description":  "Ana Lima — ST elevation (urgent)",
            "anomaly_type": "st_elevation",
            "decision":     "escalated",
        },
    ]

    results_table = Table(
        title="Agent Run Results", show_header=True, header_style="bold magenta"
    )
    results_table.add_column("Patient", style="cyan", width=14)
    results_table.add_column("Description", width=30)
    results_table.add_column("Risk score", justify="right", width=11)
    results_table.add_column("QT meds", width=14)
    results_table.add_column("Outcome", width=18)
    results_table.add_column("Steps", justify="right", width=7)

    for sc in scenarios:
        raw = generate_synthetic_ecg(
            duration_s=10.0, fs=250, heart_rate=75,
            noise_level=0.04, anomaly_type=sc["anomaly_type"]
        )
        _, _, evt = run_signal_pipeline(
            raw_ecg=raw,
            patient_id=sc["patient_id"],
            timestamp=_ts(),
            classifier=clf,
        )
        signal_events = [vars(evt)]

        result = agent.run(
            patient_id=sc["patient_id"],
            signal_events=signal_events,
            physician_decision=sc["decision"],
        )

        outcome_color = {
            "report_approved": "green",
            "escalated":       "red",
            "rejected":        "yellow",
        }.get(result.outcome, "white")

        qt_meds = result.final_state.get("qt_medications", [])
        results_table.add_row(
            sc["patient_id"],
            sc["description"],
            f"{result.final_state.get('risk_score', 0):.3f}",
            ", ".join(qt_meds) if qt_meds else "none",
            f"[{outcome_color}]{result.outcome}[/{outcome_color}]",
            str(len(result.steps)),
        )

        # Print agent trace for patient-B
        if sc["patient_id"] == "patient-B":
            console.print("\n[bold]Agent trace — João Costa:[/bold]")
            for i, step in enumerate(result.steps, 1):
                console.print(f"  [dim]{i}.[/dim] {step}")

    console.print(results_table)

    # Print buffered escalation alerts after the table (not mid-loop)
    for alert in agent.tools.escalation_alerts:
        console.print(Panel(
            f"[bold]{alert}[/bold]",
            border_style="red",
            title="[bold red]ESCALATION[/bold red]",
        ))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("Summary")
    console.print("[bold green]✓[/bold green] Signal pipeline: denoising + anomaly detection")
    console.print("[bold green]✓[/bold green] FHIR mock store: Patient, Observation, Flag, DiagnosticReport")
    console.print("[bold green]✓[/bold green] Clinical RAG: citation grounding + SelfCheck consistency")
    console.print("[bold green]✓[/bold green] Agent graph: Supervisor → Signal Analyst → RAG Reporter → HITL")
    console.print("[bold green]✓[/bold green] Escalation path: ST elevation → immediate on-call alert")
    console.print()
    console.print("[dim]Production swaps:[/dim]")
    console.print("  MockLLM → BioMistral-7B via vLLM")
    console.print("  FHIRStore → AWS HealthLake")
    console.print("  ChromaDB → Pinecone")
    console.print("  PatchTST (CPU) → ONNX export → NVIDIA Triton")
    console.print("  In-memory state → LangGraph + Redis checkpointer")


if __name__ == "__main__":
    run_demo()

"""
Demo 1: Signal Intelligence Pipeline

Shows:
  - Synthetic ECG generation (normal, AFib, ST elevation)
  - Bandpass + wavelet denoising
  - PatchTST anomaly detection
  - Signal quality assessment

Run:
  python -m omate.demo_signal                          # synthetic ECG
  python -m omate.demo_signal --data mitbih            # all downloaded MIT-BIH records
  python -m omate.demo_signal --data mitbih --record 100   # specific record
  python -m omate.demo_signal --data mitbih --record 202   # AFib (best demo)
  python -m omate.demo_signal --data mitbih --list         # list available records
"""

import sys
import time
import warnings
import numpy as np
if "--no-warnings" in sys.argv:
    warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.progress import track

console = Console()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> dict:
    args = sys.argv[1:]
    result = {
        "data": None,
        "record": None,
        "list": False,
        "start_s": None,
        "duration_s": 10.0,
        "lead": 0,
        "classifier": "clinical",   # "clinical" | "huggingface" | "patchtst"
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--data" and i + 1 < len(args):
            result["data"] = args[i + 1]; i += 2
        elif a == "--record" and i + 1 < len(args):
            result["record"] = args[i + 1]; i += 2
        elif a == "--list":
            result["list"] = True; i += 1
        elif a == "--start" and i + 1 < len(args):
            result["start_s"] = float(args[i + 1]); i += 2
        elif a == "--duration" and i + 1 < len(args):
            result["duration_s"] = float(args[i + 1]); i += 2
        elif a == "--lead" and i + 1 < len(args):
            result["lead"] = int(args[i + 1]); i += 2
        elif a == "--classifier" and i + 1 < len(args):
            result["classifier"] = args[i + 1]; i += 2
        else:
            i += 1
    return result


# ---------------------------------------------------------------------------
# Synthetic demo (default)
# ---------------------------------------------------------------------------

def run_synthetic_demo(opts: dict) -> None:
    from omate.signal import (
        generate_synthetic_ecg, run_signal_pipeline, get_ecg_classifier,
    )

    clf_name = opts.get("classifier", "clinical")
    clf = get_ecg_classifier(clf_name)
    backend_label = getattr(clf, "status", clf_name)

    console.print(Panel.fit(
        "[bold cyan]Omate POC — Signal Intelligence Demo[/bold cyan]\n"
        f"Denoising pipeline + classifier: [bold]{clf_name}[/bold]\n"
        "[dim]Source: synthetic ECG generator[/dim]",
        border_style="cyan",
    ))

    scenarios = [
        ("patient-A", "normal",       72,  0.03, "Normal sinus rhythm"),
        ("patient-B", "afib",         85,  0.05, "Atrial Fibrillation"),
        ("patient-C", "st_elevation", 68,  0.04, "ST Elevation"),
        ("patient-D", "normal",       95,  0.15, "Normal + noise"),
    ]

    table = Table(title="Signal Analysis Results — Synthetic ECG",
                  show_header=True, header_style="bold magenta")
    table.add_column("Patient", style="cyan", width=12)
    table.add_column("Scenario", width=20)
    table.add_column("Detected", width=22)
    table.add_column("Risk", justify="right", width=8)
    table.add_column("Conf", justify="right", width=7)
    table.add_column("HR", justify="right", width=6)
    table.add_column("Quality", justify="right", width=9)
    table.add_column("Escalate?", width=10)

    for patient_id, anomaly_type, hr, noise, description in track(
        scenarios, description="Processing signals..."
    ):
        raw = generate_synthetic_ecg(
            duration_s=10.0, fs=250,
            heart_rate=hr,
            noise_level=noise,
            anomaly_type=anomaly_type,
        )

        t0 = time.perf_counter()
        denoised, anomaly, event = run_signal_pipeline(
            raw_ecg=raw,
            patient_id=patient_id,
            timestamp=_ts(),
            classifier=clf,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        escalate_marker = "[red bold]YES[/red bold]" if event.requires_escalation else "[green]no[/green]"
        risk_color = "red" if event.risk_score >= 0.9 else "yellow" if event.risk_score >= 0.5 else "green"

        table.add_row(
            patient_id,
            description[:20],
            anomaly.predicted_class,
            f"[{risk_color}]{event.risk_score:.3f}[/{risk_color}]",
            f"{anomaly.confidence:.3f}",
            f"{event.heart_rate_bpm:.0f}",
            f"{denoised.quality_score:.2f}",
            escalate_marker,
        )

    console.print(table)
    console.print("\n[bold]Pipeline latency:[/bold]")
    console.print("  Bandpass filter:    ~2–4ms (CPU)")
    console.print("  Wavelet denoise:    ~3–6ms (CPU)")
    console.print("  PatchTST inference: ~15–30ms (CPU, no GPU)")
    console.print("  [dim]Production target: <100ms p99 on A10G GPU via Triton[/dim]")
    console.print("\n[dim]Tip: run with real MIT-BIH data (requires: make download-mitbih):[/dim]")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 100[/cyan]  ← normal")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 202[/cyan]  ← AFib")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --list[/cyan]        ← all records\n")


# ---------------------------------------------------------------------------
# MIT-BIH real data demo
# ---------------------------------------------------------------------------

def _list_mitbih_records() -> None:
    from omate.signal import list_available_records, RECORD_CATALOGUE

    available = list_available_records()
    table = Table(title="MIT-BIH Records (data/mitbih/)",
                  show_header=True, header_style="bold blue")
    table.add_column("Record", style="cyan", width=8)
    table.add_column("Label", width=36)
    table.add_column("Description", width=46)
    table.add_column("Status", width=10)

    for record_id, info in RECORD_CATALOGUE.items():
        status = "[green]ready[/green]" if record_id in available else "[dim]missing[/dim]"
        table.add_row(record_id, info["label"], info.get("description", ""), status)

    console.print(table)
    if not available:
        console.print("\n[yellow]No records downloaded yet.[/yellow]")
        console.print("  [cyan]make download-mitbih[/cyan]  or  [cyan]python scripts/download_mitbih.py[/cyan]\n")
    else:
        console.print(f"\n[green]{len(available)} record(s) ready.[/green]  "
                      f"Run with: [cyan]python -m omate.demo_signal --data mitbih --record 100[/cyan]\n")


def _run_single_mitbih(record_id: str, opts: dict) -> None:
    """Run pipeline on a single MIT-BIH record and show detailed output."""
    from omate.signal import (
        load_mitbih_record, run_signal_pipeline, AnomalyDetector,
        bandpass_ecg, run_denoising_pipeline,
    )

    console.print(Panel.fit(
        f"[bold cyan]Omate POC — MIT-BIH Real ECG Demo[/bold cyan]\n"
        f"Record [bold]{record_id}[/bold]  ·  PhysioNet MIT-BIH Arrhythmia Database\n"
        "[dim]360 Hz · annotated by 2 cardiologists · 30-min recording[/dim]",
        border_style="cyan",
    ))

    # Load record
    console.print(f"\n[dim]Loading record {record_id}...[/dim]")
    try:
        rec = load_mitbih_record(
            record_id=record_id,
            lead=opts["lead"],
            start_s=opts["start_s"],
            duration_s=opts["duration_s"],
        )
    except FileNotFoundError as e:
        console.print(f"\n[red]{e}[/red]\n")
        return
    except ImportError as e:
        console.print(f"\n[red]{e}[/red]\n")
        return

    console.print(f"  Record:     [cyan]{rec.record_id}[/cyan]  —  {rec.label}")
    console.print(f"  Lead:       {rec.lead_name}")
    console.print(f"  Segment:    {rec.start_s:.1f}s – {rec.start_s + rec.duration_s:.1f}s  "
                  f"({rec.duration_s:.0f}s window)")
    console.print(f"  Samples:    {len(rec.signal)} @ {rec.fs} Hz")
    console.print(f"  ADC gain:   {rec.adc_gain:.0f} units/mV")
    console.print(f"  Beats:      {rec.beat_count} annotations in window")
    console.print(f"  Dominant:   [yellow]{rec.dominant_rhythm}[/yellow]")

    # Ground-truth annotation table
    if rec.annotations:
        console.print()
        console.print(Rule("[dim]Ground-Truth Annotations (cardiologist)[/dim]", style="dim"))
        ann_table = Table(show_header=True, header_style="bold blue",
                          box=None, padding=(0, 2))
        ann_table.add_column("#", justify="right", width=4, style="dim")
        ann_table.add_column("Time (s)", justify="right", width=10)
        ann_table.add_column("Symbol", width=8, style="cyan")
        ann_table.add_column("Label", width=36)

        shown = rec.annotations[:20]
        for i, ann in enumerate(shown, 1):
            ann_table.add_row(
                str(i),
                f"{ann['time_s']:.3f}",
                ann["symbol"],
                ann["label"],
            )
        console.print(ann_table)
        if len(rec.annotations) > 20:
            console.print(f"  [dim]... and {len(rec.annotations) - 20} more[/dim]")

    # Run denoising pipeline
    console.print()
    console.print(Rule("[dim]Denoising Pipeline[/dim]", style="dim"))
    console.print("[dim]Running bandpass + wavelet denoising at 360 Hz...[/dim]")

    from omate.signal import get_ecg_classifier
    clf_name = opts.get("classifier", "clinical")
    clf = get_ecg_classifier(clf_name)
    clf_status = getattr(clf, "status", clf_name)
    console.print(f"  Classifier: [bold]{clf_status}[/bold]")
    clf.warmup()

    t0 = time.perf_counter()
    denoised, anomaly, event = run_signal_pipeline(
        raw_ecg=rec.signal,
        patient_id=f"mitbih-{record_id}",
        timestamp=_ts(),
        classifier=clf,
        fs=rec.fs,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Results
    console.print()
    console.print(Rule("[dim]Pipeline Results[/dim]", style="dim"))

    res_table = Table(show_header=False, box=None, padding=(0, 2))
    res_table.add_column("Field", style="bold", width=22)
    res_table.add_column("Value", width=40)

    risk_color = "red" if event.risk_score >= 0.9 else "yellow" if event.risk_score >= 0.5 else "green"
    escalate_str = "[red bold]YES — escalate[/red bold]" if event.requires_escalation else "[green]no[/green]"

    # Estimate SNR improvement: signal power vs noise removed
    noise = denoised.raw - denoised.denoised
    signal_power = float(np.mean(denoised.denoised ** 2))
    noise_power = float(np.mean(noise ** 2))
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-12 else 0.0

    res_table.add_row("Detected class",    f"[yellow]{anomaly.predicted_class}[/yellow]")
    res_table.add_row("Confidence",        f"{anomaly.confidence:.4f}")
    res_table.add_row("Risk score",        f"[{risk_color}]{event.risk_score:.4f}[/{risk_color}]")
    res_table.add_row("Heart rate (est.)", f"{event.heart_rate_bpm:.1f} bpm")
    res_table.add_row("Signal quality",    f"{denoised.quality_score:.4f}")
    res_table.add_row("SNR (denoised)",    f"{snr_db:.1f} dB")
    res_table.add_row("Pipeline latency",  f"{latency_ms:.1f} ms")
    res_table.add_row("Requires escalation", escalate_str)

    console.print(res_table)

    # Agreement check (ground truth vs model)
    console.print()
    console.print(Rule("[dim]Ground Truth vs Model[/dim]", style="dim"))

    gt_rhythm = rec.dominant_rhythm
    model_class = anomaly.predicted_class
    console.print(f"  Ground truth (cardiologist): [bold]{gt_rhythm}[/bold]")
    console.print(f"  Model prediction:            [bold yellow]{model_class}[/bold yellow]")

    # Simple agreement heuristic
    gt_lower = gt_rhythm.lower()
    model_lower = model_class.lower()
    agree = (
        ("normal" in gt_lower and "normal" in model_lower) or
        ("atrial" in gt_lower and "afib" in model_lower) or
        ("fibrillation" in gt_lower and "afib" in model_lower) or
        ("ventricular" in gt_lower and ("pvc" in model_lower or "vt" in model_lower)) or
        ("tachycardia" in gt_lower and ("vt" in model_lower or "svt" in model_lower)) or
        ("bundle" in gt_lower and "block" in model_lower) or
        ("pacemaker" in gt_lower and "paced" in model_lower) or
        ("premature" in gt_lower and "pvc" in model_lower)
    )
    if agree:
        console.print("  Agreement:                   [green]✓ consistent[/green]")
    else:
        console.print("  Agreement:                   [yellow]⚠ review needed[/yellow]  "
                      "[dim](PatchTST trained on synthetic; production uses fine-tuned model)[/dim]")

    console.print()
    console.print("[dim]Tip: try other records:[/dim]")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 200[/cyan]  ← PVCs")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 207[/cyan]  ← V-tach")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 214[/cyan]  ← LBBB")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --list[/cyan]        ← all\n")


def _run_all_mitbih(opts: dict) -> None:
    """Run pipeline on all downloaded MIT-BIH records, show summary table."""
    from omate.signal import (
        load_mitbih_record, list_available_records, run_signal_pipeline,
        AnomalyDetector, RECORD_CATALOGUE,
    )

    available = list_available_records()
    if not available:
        console.print("\n[red]No MIT-BIH records found in data/mitbih/[/red]")
        console.print("  Run: [cyan]make download-mitbih[/cyan]\n")
        return

    console.print(Panel.fit(
        "[bold cyan]Omate POC — MIT-BIH Arrhythmia Database[/bold cyan]\n"
        f"Running pipeline on [bold]{len(available)}[/bold] downloaded records\n"
        "[dim]PhysioNet · 360 Hz · cardiologist-annotated[/dim]",
        border_style="cyan",
    ))

    from omate.signal import get_ecg_classifier
    clf_name = opts.get("classifier", "clinical")
    clf = get_ecg_classifier(clf_name)
    clf_status = getattr(clf, "status", clf_name)
    console.print(f"[dim]Classifier: {clf_status} · Warming up...[/dim]")
    clf.warmup()

    table = Table(title="MIT-BIH Real ECG — Pipeline Results",
                  show_header=True, header_style="bold magenta")
    table.add_column("Record", style="cyan", width=8)
    table.add_column("Condition", width=28)
    table.add_column("Ground truth", width=22)
    table.add_column("Detected", width=22)
    table.add_column("Risk", justify="right", width=8)
    table.add_column("Conf", justify="right", width=7)
    table.add_column("HR", justify="right", width=6)
    table.add_column("Beats", justify="right", width=6)
    table.add_column("Quality", justify="right", width=9)

    for record_id in track(available, description="Processing MIT-BIH records..."):
        cat = RECORD_CATALOGUE.get(record_id, {})
        try:
            rec = load_mitbih_record(
                record_id=record_id,
                lead=opts["lead"],
                duration_s=opts["duration_s"],
            )
            denoised, anomaly, event = run_signal_pipeline(
                raw_ecg=rec.signal,
                patient_id=f"mitbih-{record_id}",
                timestamp=_ts(),
                classifier=clf,
                fs=rec.fs,
            )
            risk_color = "red" if event.risk_score >= 0.9 else "yellow" if event.risk_score >= 0.5 else "green"
            table.add_row(
                record_id,
                cat.get("label", "Unknown")[:28],
                rec.dominant_rhythm[:22],
                anomaly.predicted_class[:22],
                f"[{risk_color}]{event.risk_score:.3f}[/{risk_color}]",
                f"{anomaly.confidence:.3f}",
                f"{event.heart_rate_bpm:.0f}",
                str(rec.beat_count),
                f"{denoised.quality_score:.2f}",
            )
        except Exception as e:
            table.add_row(record_id, cat.get("label", "Unknown")[:28],
                          "—", f"[red]Error[/red]", "—", "—", "—", "—", str(e)[:12])

    console.print(table)
    console.print(
        "\n[dim]Note: PatchTST model is trained on synthetic ECG — confidence "
        "scores reflect pattern similarity, not clinical validation.[/dim]"
    )
    console.print(
        "[dim]Production: fine-tune on labelled MIT-BIH segments per record type.[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_demo() -> None:
    opts = _parse_args()

    if opts["data"] == "mitbih":
        if opts["list"]:
            _list_mitbih_records()
        elif opts["record"]:
            _run_single_mitbih(opts["record"], opts)
        else:
            _run_all_mitbih(opts)
    else:
        run_synthetic_demo(opts)


if __name__ == "__main__":
    run_demo()
